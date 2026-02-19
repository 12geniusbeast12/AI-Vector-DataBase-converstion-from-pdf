#ifndef VECTOR_STORE_H
#define VECTOR_STORE_H

#include <QString>
#include <QSqlDatabase>
#include <QVector>
#include <QByteArray>

struct VectorEntry {
    int id;
    QString text;
    QString sourceFile;
    QVector<float> embedding;
    double score; // For search results
};

class VectorStore {
public:
    VectorStore(const QString& dbPath = "vector_db.sqlite");
    ~VectorStore();

    bool init();
    bool addEntry(const QString& text, const QVector<float>& embedding, const QString& sourceFile = "");
    QVector<VectorEntry> search(const QVector<float>& queryEmbedding, int limit = 5);
    int count();
    void clear();
    void close();
    void setPath(const QString& name);
    bool exportToCsv(const QString& filePath);

private:
    QString m_dbPath;
    QSqlDatabase m_db;
    
    QByteArray vectorToBlob(const QVector<float>& vec);
    QVector<float> blobToVector(const QByteArray& blob);
    double cosineSimilarity(const QVector<float>& v1, const QVector<float>& v2);
};

#endif // VECTOR_STORE_H
