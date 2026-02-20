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

QVector<Chunk> PdfProcessor::extractChunks(const QString& filePath) {
    QVector<Chunk> chunks;
    FPDF_DOCUMENT doc = FPDF_LoadDocument(filePath.toLocal8Bit().constData(), nullptr);
    if (!doc) return chunks;

    int pageCount = FPDF_GetPageCount(doc);
    
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
                    QVector<unsigned short> buffer(charCount + 1);
                    FPDFText_GetText(textPage, 0, charCount, buffer.data());
                    QString pageText = QString::fromUtf16(buffer.data(), charCount);
                    
                    // Heading Detection Patterns
                    QRegularExpression chapterRegex("^(Chapter|CHAPTER|PART|Part)\\s+(\\d+)", QRegularExpression::CaseInsensitiveOption);
                    QRegularExpression sectionRegex("^(\\d+\\.\\d+)\\s+(.*)");
                    QRegularExpression subsectionRegex("^(\\d+\\.\\d+\\.\\d+)\\s+(.*)");
                    QRegularExpression chunkTypeRegex("^(Definition|Example|Theorem|Summary|Exercise|Corollary|Lemma|Proof)[:\\s+]", QRegularExpression::CaseInsensitiveOption);

                    // Sliding window over paragraphs with overlap
                    QStringList paragraphs = pageText.split("\n", Qt::SkipEmptyParts);
                    QString currentChunk;
                    const int TARGET_SIZE = 800;
                    const int HARD_MAX = 1500;
                    const int OVERLAP_SIZE = 160; 

                    for (int j = 0; j < paragraphs.size(); ++j) {
                        QString p = paragraphs[j].trimmed();
                        if (p.isEmpty()) continue;

                        // 1. Noise Safeguards
                        bool isLikelyNoise = p.length() < 15 && p.contains(QRegularExpression("^\\d+$|Page\\s+\\d+", QRegularExpression::CaseInsensitiveOption));
                        if (isLikelyNoise) continue;

                        int level = 0;
                        // 2. Detect Structure
                        QRegularExpressionMatch chapMatch = chapterRegex.match(p);
                        if (chapMatch.hasMatch() && p.length() < 100) {
                            currentChapter = p;
                            currentSection = ""; 
                            currentSubsection = "";
                            level = 1;
                        } else {
                            QRegularExpressionMatch secMatch = sectionRegex.match(p);
                            if (secMatch.hasMatch() && p.length() < 120) {
                                currentSection = p;
                                currentSubsection = "";
                                level = 2;
                            } else {
                                QRegularExpressionMatch subMatch = subsectionRegex.match(p);
                                if (subMatch.hasMatch() && p.length() < 150) {
                                    currentSubsection = p;
                                    level = 3;
                                }
                            }
                        }

                        QString path;
                        if (!currentChapter.isEmpty()) path = currentChapter;
                        if (!currentSection.isEmpty()) path = (path.isEmpty() ? "" : path + " > ") + currentSection;
                        if (!currentSubsection.isEmpty()) path = (path.isEmpty() ? "" : path + " > ") + currentSubsection;

                        // 3. Detect Chunk Type with List/Definition Metadata
                        QString cType = "text";
                        QString lType = "";
                        int lLen = 0;

                        if (p.startsWith("•") || p.startsWith("-") || p.startsWith("*")) {
                            cType = "list";
                            lType = "bullet";
                            lLen = 1;
                        } else if (p.contains(QRegularExpression("^(\\d+|[a-zA-Z])\\)"))) {
                            cType = "list";
                            lType = "numbered";
                            lLen = 1;
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

                        if (currentChunk.isEmpty()) {
                            currentChunk = p;
                        } else {
                            // If appending a list item to an existing list chunk, increment length
                            if (cType == "list" && currentChunk.contains("\n• ") || currentChunk.contains("\n- ")) {
                                lLen++;
                            }
                            currentChunk += "\n" + p;
                        }

                        // 4. Smart Chunking (Sentence-aware splitting + Hard Max)
                        if (currentChunk.length() >= TARGET_SIZE || currentChunk.length() >= HARD_MAX) {
                            // Count sentences in current chunk
                            QRegularExpression sentenceSplitter("(?<=[.?!])\\s+");
                            int sCount = currentChunk.split(sentenceSplitter).size();

                            QRegularExpressionMatchIterator it = sentenceSplitter.globalMatch(currentChunk);
                            int lastSplit = -1;
                            while (it.hasNext()) { lastSplit = it.next().capturedStart(); }
                            
                            QString chunkToSave;
                            // Enforce Hard Max if no split point found
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
                            
                            if (currentChunk.isEmpty()) {
                                QString overlap;
                                int backIdx = j;
                                while (backIdx >= 0 && overlap.length() < OVERLAP_SIZE) {
                                    if (overlap.isEmpty()) overlap = paragraphs[backIdx].trimmed();
                                    else overlap = paragraphs[backIdx].trimmed() + "\n" + overlap;
                                    backIdx--;
                                }
                                currentChunk = overlap;
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
        emit progressUpdated(i + 1, pageCount);
    }
    FPDF_CloseDocument(doc);
    return chunks;
}

QString PdfProcessor::generateDocId(const QString& filePath) {
    QFile file(filePath);
    if (!file.open(QIODevice::ReadOnly)) return "";
    
    // Stable ID: Hash of filename + size (fast and robust enough for local RAG)
    QString identity = QString("%1_%2").arg(QFileInfo(filePath).fileName()).arg(file.size());
    return QCryptographicHash::hash(identity.toUtf8(), QCryptographicHash::Md5).toHex();
}
