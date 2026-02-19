#include "pdf_processor.h"
#include <QDebug>
#include <QFile>

PdfProcessor::PdfProcessor(QObject *parent) : QObject(parent) {}

PdfProcessor::~PdfProcessor() {}

void PdfProcessor::initLibrary() {
    FPDF_InitLibrary();
}

void PdfProcessor::destroyLibrary() {
    FPDF_DestroyLibrary();
}

QString PdfProcessor::extractText(const QString& filePath) {
    FPDF_DOCUMENT doc = FPDF_LoadDocument(filePath.toLocal8Bit().constData(), nullptr);
    if (!doc) {
        qDebug() << "Failed to load document:" << filePath;
        return "";
    }

    int pageCount = FPDF_GetPageCount(doc);
    QString fullText;

    for (int i = 0; i < pageCount; ++i) {
        FPDF_PAGE page = FPDF_LoadPage(doc, i);
        if (page) {
            FPDF_TEXTPAGE textPage = FPDFText_LoadPage(page);
            if (textPage) {
                int charCount = FPDFText_CountChars(textPage);
                if (charCount > 0) {
                    QVector<unsigned short> buffer(charCount + 1);
                    FPDFText_GetText(textPage, 0, charCount, buffer.data());
                    fullText += QString::fromUtf16(buffer.data(), charCount);
                    fullText += "\n";
                }
                FPDFText_ClosePage(textPage);
            }
            FPDF_ClosePage(page);
        }
        emit progressUpdated(i + 1, pageCount);
    }

    FPDF_CloseDocument(doc);
    return fullText;
}
