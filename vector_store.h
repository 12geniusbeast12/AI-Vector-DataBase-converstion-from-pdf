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
    QString docId;
    int pageNum;
    QString modelSig;
    QString headingPath;
    int headingLevel = 0;
    QString chunkType; // "text", "summary", "definition", "example", "list", etc.
    int sentenceCount = 0;
    QString listType = "";
    int listLength = 0;
    QVector<float> embedding;
    double score; // For search results
    int semanticRank = 0;
    int keywordRank = 0;
};

enum class IntentType { General, Definition, Summary, Procedure, Example };

class VectorStore {
public:
    VectorStore(const QString& dbPath = "vector_db.sqlite");
    ~VectorStore();

    bool init();
    bool addEntry(const QString& text, const QVector<float>& embedding, 
                  const QString& sourceFile, const QString& docId, 
                  int pageNum, int chunkIdx, const QString& modelSig,
                  const QString& path = "", int level = 0,
                  const QString& chunkType = "text",
                  int sCount = 0, const QString& lType = "", int lLen = 0);
                  
    QString getContext(const QString& docId, int currentIdx, int offset = 1);
                  
    QVector<VectorEntry> search(const QVector<float>& queryEmbedding, int limit = 5);
    QVector<VectorEntry> hybridSearch(const QString& queryText, const QVector<float>& queryEmbedding, int limit = 5);
    
    IntentType detectIntent(const QString& queryText, const QVector<float>& queryEmbedding);
    
    void logRetrieval(const QString& query, int sRank, int kRank, int fRank, 
                      int lEmbed, int lSearch, int lFusion, int lRerank, double topScore);
    
    int count();
    void clear();
    void close();
    QString m_mDbPath() const { return m_dbPath; }
    QSqlDatabase database() const { return m_db; }
    
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
