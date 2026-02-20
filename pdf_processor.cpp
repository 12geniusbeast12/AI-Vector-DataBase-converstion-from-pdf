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

QVector<Chunk> PdfProcessor::extractChunks(const QString& filePath) {
    QVector<Chunk> chunks;
    FPDF_DOCUMENT doc = FPDF_LoadDocument(filePath.toLocal8Bit().constData(), nullptr);
    if (!doc) return chunks;

    int pageCount = FPDF_GetPageCount(doc);
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
                    
                    // Sliding window over paragraphs with overlap
                    QStringList paragraphs = pageText.split("\n", Qt::SkipEmptyParts);
                    QString currentChunk;
                    const int TARGET_SIZE = 800;
                    const int OVERLAP_SIZE = 160; // 20% overlap

                    for (int j = 0; j < paragraphs.size(); ++j) {
                        QString p = paragraphs[j].trimmed();
                        if (p.isEmpty()) continue;

                        if (currentChunk.isEmpty()) {
                            currentChunk = p;
                        } else {
                            currentChunk += "\n" + p;
                        }

                        if (currentChunk.length() >= TARGET_SIZE) {
                            chunks.append({currentChunk, i + 1});
                            
                            // Re-initialize with overlap: find how many previous paragraphs to keep
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
                    if (currentChunk.length() > 20) {
                        chunks.append({currentChunk, i + 1});
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
