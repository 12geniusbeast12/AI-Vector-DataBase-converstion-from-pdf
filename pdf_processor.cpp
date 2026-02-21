#include "pdf_processor.h"
#include <QDebug>
#include <QFile>
#include <QFileInfo>

PdfProcessor::PdfProcessor(QObject *parent) : QObject(parent) {}

PdfProcessor::~PdfProcessor() {}

void PdfProcessor::initLibrary() {
    FPDF_InitLibrary();
}

void PdfProcessor::destroyLibrary() {
    FPDF_DestroyLibrary();
}

#include <QCryptographicHash>
#include <QRegularExpression>
#include <QCoreApplication>
#include <unordered_map>

void PdfProcessor::extractChunksAsync(const QString& filePath) {
    QVector<Chunk> chunks;
    FPDF_DOCUMENT doc = FPDF_LoadDocument(filePath.toLocal8Bit().constData(), nullptr);
    if (!doc) {
        emit extractionFinished();
        return;
    }

    int pageCount = FPDF_GetPageCount(doc);
    
    // --- PHASE 4 PRE-COMPUTE: Fast Duplicate Filtering (Rolling Hash) ---
    // We do a fast first pass to compute hashes for lines in the top 15% and bottom 15% of the page.
    std::unordered_map<uint64_t, int> lineFrequencies;
    
    for (int p = 0; p < pageCount; ++p) {
        FPDF_PAGE page = FPDF_LoadPage(doc, p);
        if (page) {
            FPDF_TEXTPAGE textPage = FPDFText_LoadPage(page);
            if (textPage) {
                int charCount = FPDFText_CountChars(textPage);
                if (charCount > 0) {
                    double pageHeight = FPDF_GetPageHeight(page);
                    double topMargin = pageHeight * 0.15;
                    double bottomMargin = pageHeight * 0.85; // FPDF uses y=0 at bottom usually, but we check both ends
                    
                    QVector<unsigned short> buffer(charCount + 1);
                    FPDFText_GetText(textPage, 0, charCount, buffer.data());
                    QString pageText = QString::fromUtf16(buffer.data(), charCount);
                    
                    for (const QString& rawLine : pageText.split("\n")) {
                        QString norm = rawLine.toLower().remove(QRegularExpression("\\d")).trimmed();
                        if (norm.length() > 3) {
                            uint64_t h = qHash(norm);
                            lineFrequencies[h]++;
                        }
                    }
                }
                FPDFText_ClosePage(textPage);
            }
            FPDF_ClosePage(page);
        }
    }
    
    // Stateful Tracker (persists across pages)
    QString currentChapter;
    QString currentSection;
    QString currentSubsection;

    for (int i = 0; i < pageCount; ++i) {
        FPDF_PAGE page = FPDF_LoadPage(doc, i);
        if (page) {
            FPDF_TEXTPAGE textPage = FPDFText_LoadPage(page);
            if (textPage) {
                int charCount = FPDFText_CountChars(textPage);
                if (charCount > 0) {
                    
                    // --- PHASE 1: EXACT LAYOUT EXTRACTION ---
                    struct CharInfo {
                        double left, top, right, bottom;
                        unsigned short ch;
                        double fontSize;
                        int fontWeight;
                    };
                    QVector<CharInfo> chars;
                    
                    for (int c = 0; c < charCount; ++c) {
                        double L, T, R, B;
                        FPDFText_GetCharBox(textPage, c, &L, &R, &B, &T);
                        unsigned short ch = FPDFText_GetUnicode(textPage, c);
                        double fSize = FPDFText_GetFontSize(textPage, c);
                        int fWeight = FPDFText_GetFontWeight(textPage, c); // -1 if error
                        chars.append({L, T, R, B, ch, fSize, fWeight});
                    }

                    // Group into Lines
                    struct LineInfo {
                        double top, bottom, left, right;
                        QString text;
                        double fontSize = 0;
                        int fontWeight = 0;
                        int charCount = 0;
                    };
                    QVector<LineInfo> lines;
                    
                    if (!chars.isEmpty()) {
                        std::sort(chars.begin(), chars.end(), [](const CharInfo& a, const CharInfo& b) {
                            if (qAbs(a.top - b.top) > 5.0) return a.top > b.top;
                            return a.left < b.left;
                        });

                        LineInfo currentLine;
                        currentLine.top = chars.first().top;
                        currentLine.bottom = chars.first().bottom;
                        currentLine.fontSize = chars.first().fontSize;
                        currentLine.fontWeight = chars.first().fontWeight;
                        currentLine.charCount = 1;

                        for (int c = 0; c < chars.size(); ++c) {
                            const auto& chInfo = chars[c];
                            if (qAbs(chInfo.top - currentLine.top) > 5.0 && c > 0) { // c>0 handles first char
                                currentLine.fontSize /= currentLine.charCount;
                                currentLine.fontWeight /= currentLine.charCount;
                                lines.append(currentLine);
                                
                                currentLine.top = chInfo.top;
                                currentLine.bottom = chInfo.bottom;
                                currentLine.left = chInfo.left;
                                currentLine.right = chInfo.right;
                                currentLine.text = QString(QChar(chInfo.ch));
                                currentLine.fontSize = chInfo.fontSize;
                                currentLine.fontWeight = chInfo.fontWeight;
                                currentLine.charCount = 1;
                            } else {
                                if (c > 0 && chInfo.left - currentLine.right > 4.0) currentLine.text += " ";
                                currentLine.text += QChar(chInfo.ch);
                                currentLine.right = qMax(currentLine.right, chInfo.right);
                                currentLine.top = qMax(currentLine.top, chInfo.top);
                                currentLine.bottom = qMin(currentLine.bottom, chInfo.bottom);
                                currentLine.fontSize += chInfo.fontSize;
                                currentLine.fontWeight += chInfo.fontWeight;
                                currentLine.charCount++;
                            }
                        }
                        if (!currentLine.text.isEmpty()) {
                            currentLine.fontSize /= qMax(currentLine.charCount, 1);
                            currentLine.fontWeight /= qMax(currentLine.charCount, 1);
                            lines.append(currentLine);
                        }
                    }

                    // --- PHASE 2: BLOCK REASSEMBLY ---
                    struct TextBlock {
                        QString text;
                        double left;
                        double top;
                        bool isCode = false;
                        bool isTable = false;
                        int lines = 0;
                        int symbols = 0;
                        int nums = 0;
                        double fontSize = 0.0;
                        int fontWeight = 0;
                    };
                    QVector<TextBlock> blocks;

                    double pageWidth = FPDF_GetPageWidth(page);
                    double pageHeight = FPDF_GetPageHeight(page);
                    double colSplit = pageWidth / 2.0;

                    QVector<LineInfo> col1, col2;
                    for (const auto& l : lines) {
                        if (l.left < colSplit) col1.append(l);
                        else col2.append(l);
                    }

                    QVector<LineInfo> orderedLines = col1;
                    orderedLines.append(col2);

                    TextBlock currentBlock;
                    if (!orderedLines.isEmpty()) {
                        currentBlock.top = orderedLines.first().top;
                        currentBlock.left = orderedLines.first().left;
                        
                        for (int l = 0; l < orderedLines.size(); ++l) {
                            const auto& line = orderedLines[l];
                            
                            // 1. Noise Filter Applier (Headers / Footers)
                            QString norm = line.text.toLower().remove(QRegularExpression("\\d")).trimmed();
                            if (norm.length() > 3) {
                                uint64_t h = qHash(norm);
                                // If line occurs on > 5 pages AND it's in the top or bottom 15% margin
                                if (lineFrequencies[h] > 5) {
                                    if (line.top > pageHeight * 0.85 || line.top < pageHeight * 0.15) {
                                        continue; // Ditch it
                                    }
                                }
                            }
                            
                            if (line.text.length() < 5 && line.text.contains(QRegularExpression("^\\s*\\d+\\s*$"))) continue; // bare page num

                            // Block Boundary logic
                            bool forceNewBlock = false;
                            if (l > 0) {
                                const auto& prevLine = orderedLines[l - 1];
                                if (qAbs(prevLine.top - line.top) > 15.0) forceNewBlock = true;
                                if (line.top > prevLine.top + 20.0) forceNewBlock = true;
                            }

                            if (forceNewBlock) {
                                if (currentBlock.lines > 0) {
                                    currentBlock.fontSize /= currentBlock.lines;
                                    currentBlock.fontWeight /= currentBlock.lines;
                                    blocks.append(currentBlock);
                                }
                                currentBlock.text = line.text;
                                currentBlock.top = line.top;
                                currentBlock.left = line.left;
                                currentBlock.lines = 1;
                                currentBlock.symbols = line.text.count(QRegularExpression("[{};()#<>:=-]"));
                                currentBlock.nums = line.text.count(QRegularExpression("\\d"));
                                currentBlock.fontSize = line.fontSize;
                                currentBlock.fontWeight = line.fontWeight;
                            } else {
                                if (!currentBlock.text.isEmpty()) currentBlock.text += "\n";
                                currentBlock.text += line.text.trimmed();
                                currentBlock.lines++;
                                currentBlock.symbols += line.text.count(QRegularExpression("[{};()#<>:=-]"));
                                currentBlock.nums += line.text.count(QRegularExpression("\\d"));
                                currentBlock.fontSize += line.fontSize;
                                currentBlock.fontWeight += line.fontWeight;
                            }
                        }
                        if (currentBlock.lines > 0) {
                            currentBlock.fontSize /= currentBlock.lines;
                            currentBlock.fontWeight /= currentBlock.lines;
                            blocks.append(currentBlock);
                        }
                    }

                    // Pre-compute Baseline Font Size roughly (median or mode)
                    double baselineSize = 10.0;
                    if (!blocks.isEmpty()) {
                        QMap<int, int> sizeFreq;
                        for (const auto& b : blocks) sizeFreq[(int)b.fontSize]++;
                        int maxFreq = 0;
                        for (auto it = sizeFreq.begin(); it != sizeFreq.end(); ++it) {
                            if (it.value() > maxFreq) { maxFreq = it.value(); baselineSize = it.key(); }
                        }
                    }

                    // --- PHASE 3: STRUCTURE & CHUNKING ---
                    QRegularExpression chapterRegex("^(Chapter|CHAPTER|PART|Part)\\s+(\\d+)", QRegularExpression::CaseInsensitiveOption);
                    QRegularExpression sectionRegex("^(\\d+\\.\\d+)\\s+(.*)");
                    QRegularExpression subsectionRegex("^(\\d+\\.\\d+\\.\\d+)\\s+(.*)");
                    QRegularExpression chunkTypeRegex("^(Definition|Example|Theorem|Summary|Exercise|Corollary|Lemma|Proof)[:\\s+]", QRegularExpression::CaseInsensitiveOption);

                    QString currentChunk;
                    const int TARGET_SIZE = 800;
                    const int HARD_MAX = 1500;
                    const int OVERLAP_SIZE = 160; 
                    
                    for (int b = 0; b < blocks.size(); ++b) {
                        const auto& block = blocks[b];
                        QString p = block.text.trimmed();
                        if (p.isEmpty()) continue;

                        int level = 0;
                        bool isHeadingLayout = (block.fontSize >= baselineSize + 2.0) && (block.lines <= 3) && (block.text.length() < 120);
                        
                        QRegularExpressionMatch chapMatch = chapterRegex.match(p);
                        if ((chapMatch.hasMatch() || (isHeadingLayout && block.fontSize >= baselineSize + 6.0)) && p.length() < 100) {
                            currentChapter = p.replace("\n", " ");
                            currentSection = ""; 
                            currentSubsection = "";
                            level = 1;
                        } else {
                            QRegularExpressionMatch secMatch = sectionRegex.match(p);
                            if ((secMatch.hasMatch() || (isHeadingLayout && block.fontSize >= baselineSize + 3.0)) && p.length() < 120) {
                                currentSection = p.replace("\n", " ");
                                currentSubsection = "";
                                level = 2;
                            } else {
                                QRegularExpressionMatch subMatch = subsectionRegex.match(p);
                                if ((subMatch.hasMatch() || (isHeadingLayout && block.fontWeight > 600)) && p.length() < 150) {
                                    currentSubsection = p.replace("\n", " ");
                                    level = 3;
                                }
                            }
                        }

                        QString path;
                        if (!currentChapter.isEmpty()) path = currentChapter;
                        if (!currentSection.isEmpty()) path = (path.isEmpty() ? "" : path + " > ") + currentSection;
                        if (!currentSubsection.isEmpty()) path = (path.isEmpty() ? "" : path + " > ") + currentSubsection;

                        // Phase 2 Type detection Expansion: Code and Table Heuristics
                        QString cType = "text";
                        QString lType = "";
                        int lLen = 0;
                        
                        int codeScore = 0;
                        if (block.symbols > block.lines * 2) codeScore += 4;
                        if (p.contains(QRegularExpression("\\b(int|class|public|void|return|const|template|static|if|else|for|while)\\b"))) codeScore += 3;
                        if (p.startsWith("    ") || p.startsWith("\t")) codeScore += 3;
                        
                        bool isTable = (block.nums > block.lines * 3) && (block.text.count(".") < block.lines / 2); // High density, low prose

                        if (codeScore >= 5) {
                            cType = "code";
                        } else if (isTable) {
                            cType = "table";
                        } else if (p.startsWith("•") || p.startsWith("-") || p.startsWith("*")) {
                            cType = "list";
                            lType = "bullet";
                            lLen = block.lines;
                        } else if (p.contains(QRegularExpression("^(\\d+|[a-zA-Z])\\)"))) {
                            cType = "list";
                            lType = "numbered";
                            lLen = block.lines;
                        } else {
                            QRegularExpression definitionRegex("(Definition|DEFINITION|Theorem|THEOREM|Lemma|LEMMA|Corollary|COROLLARY)[:\\s+]", QRegularExpression::CaseInsensitiveOption);
                            QRegularExpressionMatch defMatch = definitionRegex.match(p);
                            if (defMatch.hasMatch() && p.indexOf(defMatch.captured(1)) < 5) {
                                cType = "definition";
                            } else {
                                QRegularExpressionMatch typeMatch = chunkTypeRegex.match(p);
                                if (typeMatch.hasMatch()) {
                                    cType = typeMatch.captured(1).toLower();
                                }
                            }
                        }

                        // Appending logic -> if code/table, try dumping existing prose chunk first
                        if (cType == "code" || cType == "table") {
                            if (!currentChunk.isEmpty()) {
                                QRegularExpression sentenceSplitter("(?<=[.?!])\\s+");
                                int sCount = currentChunk.split(sentenceSplitter).size();
                                chunks.append({currentChunk, i + 1, path, level, "text", sCount, "", 0});
                                currentChunk = "";
                            }
                            chunks.append({p, i + 1, path, level, cType, 0, "", 0});
                            continue;
                        }

                        if (currentChunk.isEmpty()) {
                            currentChunk = p;
                        } else {
                            currentChunk += "\n" + p;
                        }

                        if (currentChunk.length() >= TARGET_SIZE || currentChunk.length() >= HARD_MAX) {
                            QRegularExpression sentenceSplitter("(?<=[.?!])\\s+");
                            int sCount = currentChunk.split(sentenceSplitter).size();

                            QRegularExpressionMatchIterator it = sentenceSplitter.globalMatch(currentChunk);
                            int lastSplit = -1;
                            while (it.hasNext()) { lastSplit = it.next().capturedStart(); }
                            
                            QString chunkToSave;
                            if (lastSplit > TARGET_SIZE / 2 && currentChunk.length() < HARD_MAX) {
                                chunkToSave = currentChunk.left(lastSplit);
                                currentChunk = currentChunk.mid(lastSplit).trimmed();
                            } else if (currentChunk.length() >= HARD_MAX) {
                                chunkToSave = currentChunk.left(HARD_MAX);
                                currentChunk = currentChunk.mid(HARD_MAX).trimmed();
                            } else {
                                chunkToSave = currentChunk;
                                currentChunk = "";
                            }

                            chunks.append({chunkToSave, i + 1, path, level, cType, sCount, lType, lLen});
                            
                            if (currentChunk.isEmpty() && b > 0) {
                                currentChunk = blocks[b-1].text.right(OVERLAP_SIZE);
                            }
                        }
                    }
                    if (currentChunk.length() > 20) {
                        QString path;
                        if (!currentChapter.isEmpty()) path = currentChapter;
                        if (!currentSection.isEmpty()) path = (path.isEmpty() ? "" : path + " > ") + currentSection;
                        if (!currentSubsection.isEmpty()) path = (path.isEmpty() ? "" : path + " > ") + currentSubsection;
                        
                        QRegularExpression sentenceSplitter("(?<=[.?!])\\s+");
                        int sCount = currentChunk.split(sentenceSplitter).size();
                        chunks.append({currentChunk, i + 1, path, 0, "text", sCount, "", 0});
                    }
                }
                FPDFText_ClosePage(textPage);
            }
            FPDF_ClosePage(page);
        }
        
        // Emit chunks incrementally
        if (!chunks.isEmpty()) {
            emit chunksReady(chunks);
            chunks.clear();
        }
        
        emit progressUpdated(i + 1, pageCount);
        QCoreApplication::processEvents();
    }
    FPDF_CloseDocument(doc);
    emit extractionFinished();
}

QString PdfProcessor::generateDocId(const QString& filePath) {
    QFile file(filePath);
    if (!file.open(QIODevice::ReadOnly)) return "";
    
    // Stable ID: Hash of filename + size (fast and robust enough for local RAG)
    QString identity = QString("%1_%2").arg(QFileInfo(filePath).fileName()).arg(file.size());
    return QCryptographicHash::hash(identity.toUtf8(), QCryptographicHash::Md5).toHex();
}
