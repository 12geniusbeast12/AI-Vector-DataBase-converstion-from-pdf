#ifndef VECTOR_STORE_H
#define VECTOR_STORE_H

#include <QString>
#include <QSqlDatabase>
#include <QVector>
#include <QByteArray>
#include <QCache>
#include <QMutex>
#include <QThreadPool>
#include <QDateTime>

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
    int rerankRank = 0;
    QDateTime createdAt;
    float trustScore = 1.0f;
    bool isExploration = false;
    float stabilityIndex = 1.0f; // Phase 4.4: 1.0 = stable, 0.0 = volatile
};

struct SourceContext {
    int promptIndex = 0;
    QString chunkId;          
    QString docName;
    QString headingPath;
    int pageNumber = 0;
    QString chunkText;
    
    float semanticScore = 0.0f;
    QVector<float> embedding;
    float finalScore = 0.0f;
    int finalRank = 0;
    float trustScore = 1.0f;
    QString trustReason;
    
    QString retrievalMethod;
    QString retrievalStage;
    qint64 retrievalTime = 0;
};

struct ClaimNode {
    QString statement;
    QVector<int> sourceIndices;
    float confidence = 0.0f;
};


struct Contradiction {
    QString claim;
    QVector<int> conflictingIndices;
    float severity;
};

enum class IntentType { General, Definition, Summary, Procedure, Example };

struct SearchAudit {
    qint64 t_fts = 0;
    qint64 t_vector = 0;
    qint64 t_mmr = 0;
    qint64 t_rerank = 0;
    float queryStabilityScore = 1.0f; // Phase 4.4 global query signal
    qint64 t_context = 0;
    qint64 t_synthesis = 0;
};

struct SearchOptions {
    int limit = 5;
    bool enableStreaming = true;
    bool highPriority = true;
    float semanticThreshold = 0.95f;
    bool deterministic = false; // Benchmarking flag
    bool experimentalMmr = false; // Toggle for Phase 4.1 logic
    bool enableExploration = false; // Toggle for Phase 4.3 logic
    bool useRerank = false; // Added missing member
};

class VectorStore : public QObject {
    Q_OBJECT
public:
    VectorStore(const QString& dbPath = "vector_db.sqlite", QObject *parent = nullptr);
    ~VectorStore();

    bool init();
    bool addEntry(const QString& text, const QVector<float>& embedding, 
                  const QString& sourceFile, const QString& docId, 
                  int pageNum, int chunkIdx, const QString& modelSig,
                  const QString& path = "", int level = 0,
                  const QString& chunkType = "text",
                  int sCount = 0, const QString& lType = "", int lLen = 0);
                  
    void boostEntry(int entryId, float amount);
    void addInteraction(int entryId, const QString& query, bool isExploration = false);
                  
    QString getContext(const QString& docId, int currentIdx, int offset = 1);
    SourceContext getSourceContext(VectorEntry entry, int offset = 1, const QString& stage = "hybrid");
                  
    QVector<VectorEntry> search(const QVector<float>& queryEmbedding, int limit = 5);
    QVector<VectorEntry> ftsSearch(const QString& queryText, int limit = 5);
    QVector<VectorEntry> hybridSearch(const QString& queryText, const QVector<float>& queryEmbedding, const SearchOptions& options = SearchOptions());
    
    IntentType detectIntent(const QString& queryText, const QVector<float>& queryEmbedding);
    
    void logRetrieval(const QString& query, int sRank, int kRank, int fRank, 
                      int lEmbed, int lSearch, int lFusion, int lRerank, double topScore,
                      float mmrPenalty = 0.0f, bool isExploration = false, int rankDelta = 0,
                      float stability = 1.0f);
    
    // Phase 4.0: Observability & Benchmarking
    void setBenchmarkingMode(bool enabled) { m_benchmarkingMode = enabled; }
    void setGlobalSeed(int seed);
    
    void warmup(); // Low-priority pre-ping
    
    int count();
    void clear();
    void close();
    QString m_mDbPath() const { return m_dbPath; }
    QSqlDatabase database() const { return m_db; }
    
    void setPath(const QString& name);
    bool exportToCsv(const QString& filePath);
    
    // Workspace Metadata & Guardrails
    void setMetadata(const QString& key, const QString& value);
    QString getMetadata(const QString& key);
    int getRegisteredDimension();
    void setRegisteredDimension(int dim);
    
    // Phase 4.0 Observability State
    bool m_benchmarkingMode = false;
    int m_benchSeed = 42;
    
    // Phase 4.1 Adaptive State
    double m_avgDocEntropy = 0.0; // Smoothed via EMA
    int m_sessionSearchCount = 0;

private:
    QString m_dbPath;
    QSqlDatabase m_db;
    
    // Phase 3A: High-Performance Infrastructure
    QThreadPool* m_threadPool;
    QCache<QString, QVector<VectorEntry>> m_queryCache; // Layer 1: Exact
    
    struct SemanticCacheEntry {
        QVector<float> embedding;
        QVector<VectorEntry> results;
        QDateTime lastUsed;
    };
    QVector<SemanticCacheEntry> m_semanticCache; // Layer 2: Semantic Similarity
    QMutex m_cacheMutex;
    
    // Diagnostic stats
    int m_cacheHits = 0;
    int m_cacheMisses = 0;

    QByteArray vectorToBlob(const QVector<float>& vec);
    QVector<float> blobToVector(const QByteArray& blob);
    double cosineSimilarity(const QVector<float>& v1, const QVector<float>& v2);
};

#endif // VECTOR_STORE_H
