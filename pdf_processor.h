#ifndef PDF_PROCESSOR_H
#define PDF_PROCESSOR_H

#include <QString>
#include <QObject>
#include <fpdfview.h>
#include <fpdf_text.h>

struct Chunk {
    QString text;
    int pageNum;
    QString headingPath;
    int headingLevel = 0;
    QString chunkType = "text";
    int sentenceCount = 0;
    QString listType = "";
    int listLength = 0;
};

Q_DECLARE_METATYPE(QVector<Chunk>)


class PdfProcessor : public QObject {
    Q_OBJECT
public:
    explicit PdfProcessor(QObject *parent = nullptr);
    ~PdfProcessor();

    static void initLibrary();
    static void destroyLibrary();

    void extractChunksAsync(const QString& filePath);
    static QString generateDocId(const QString& filePath);

signals:
    void progressUpdated(int page, int total);
    void chunksReady(QVector<Chunk> chunks);
    void extractionFinished();
};

#endif // PDF_PROCESSOR_H
