#ifndef PDF_PROCESSOR_H
#define PDF_PROCESSOR_H

#include <QString>
#include <QObject>
#include <fpdfview.h>
#include <fpdf_text.h>

class PdfProcessor : public QObject {
    Q_OBJECT
public:
    explicit PdfProcessor(QObject *parent = nullptr);
    ~PdfProcessor();

    static void initLibrary();
    static void destroyLibrary();

    QString extractText(const QString& filePath);

signals:
    void progressUpdated(int page, int total);
};

#endif // PDF_PROCESSOR_H
