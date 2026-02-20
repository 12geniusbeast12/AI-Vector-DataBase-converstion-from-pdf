#ifndef PDF_PROCESSOR_H
#define PDF_PROCESSOR_H

#include <QString>
#include <QObject>
#include <fpdfview.h>
#include <fpdf_text.h>

struct Chunk {
    QString text;
    int pageNum;
};

class PdfProcessor : public QObject {
    Q_OBJECT
public:
    explicit PdfProcessor(QObject *parent = nullptr);
    ~PdfProcessor();

    static void initLibrary();
    static void destroyLibrary();

    QVector<Chunk> extractChunks(const QString& filePath);
    static QString generateDocId(const QString& filePath);

signals:
    void progressUpdated(int page, int total);
};

#endif // PDF_PROCESSOR_H
