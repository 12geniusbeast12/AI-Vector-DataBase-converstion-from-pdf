#ifndef GEMINI_API_H
#define GEMINI_API_H

#include <QObject>
#include <QString>
#include <QNetworkAccessManager>
#include <QNetworkReply>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonArray>
#include <QVector>
#include <QMap>
#include <QVariant>
#include "vector_store.h"
#include <QSet>
#include <QFuture>
#include <memory>

enum class ModelCapability {
    Embedding,
    Chat,
    Rerank,
    Summary
};

// Required for QSet to hash the enum class
inline uint qHash(ModelCapability key, uint seed = 0) {
    return qHash(static_cast<uint>(key), seed);
}

struct ModelInfo {
    QString name;
    QString engine; // Gemini, Ollama, LMStudio
    QString endpoint;
    QString version;
    QSet<ModelCapability> capabilities;
    int maxTokens = 4096;
};

struct RerankResult {
    int chunkId;
    float score;
    int originalRank;
};

// Strategy Pattern Interface for Cross-Encoders
class IRerankClient {
public:
    virtual ~IRerankClient() = default;
    
    // Core synchronous batch execution
    virtual QVector<RerankResult> rerank(const QString& query, const QVector<VectorEntry>& candidates, int topK = 5) = 0;
    
    // Asynchronous batch request 
    virtual QFuture<QVector<RerankResult>> rerankAsync(const QString& query, const QVector<VectorEntry>& candidates, int topK = 5) = 0;
    
    // Phase 3B: Cross-Session Persistence
    virtual void loadStats(float mean, float stdDev) = 0;
    virtual void saveStats(float& mean, float& stdDev) = 0;
};

class GeminiApi : public QObject {
    Q_OBJECT
public:
    explicit GeminiApi(const QString& apiKey = "", QObject *parent = nullptr);
    static GeminiApi* instance() { return s_instance; }

    void setApiKey(const QString& key) { m_apiKey = key; }
    void setLocalMode(int mode);
    void setEmbeddingModel(const ModelInfo& model) { m_embedModel = model; }
    void setReasoningModel(const ModelInfo& model) { m_reasonModel = model; }
    void setRerankModel(const ModelInfo& model);
    void updateRerankerStats(float mean, float stdDev);
    
    void processPdf(const QString& filePath);
    void getEmbeddings(const QString& text, const QMap<QString, QVariant>& metadata = {});
    void generateSummary(const QString& text, const QMap<QString, QVariant>& metadata = {});
    void synthesizeResponse(const QString& query, const QVector<SourceContext>& contexts, const QMap<QString, QVariant>& metadata = {});
    void rerank(const QString& query, const QVector<VectorEntry>& candidates);
    void discoverModels();
    
signals:
    void pdfProcessed(const QString& text);
    void embeddingsReady(const QString& text, const QVector<float>& embedding, const QMap<QString, QVariant>& metadata = {});
    void summaryReady(const QString& summary, const QMap<QString, QVariant>& metadata = {});
    void synthesisReady(const QVector<ClaimNode>& claims, const QVector<SourceContext>& contexts, const QMap<QString, QVariant>& metadata = {});
    void rerankingReady(const QVector<VectorEntry>& rerankedResults);
    void partialResultsReady(const QVector<VectorEntry>& results, const QString& stage);
    void rerankerStatsUpdated(float mean, float stdDev);
    void anomalyDetected(const QString& title, const QString& message);
    void discoveredModelsReady(const QVector<ModelInfo>& models);
    void errorOccurred(const QString& error);

private slots:
    void onEmbeddingsReply(QNetworkReply* reply, const QString& originalText, const QMap<QString, QVariant>& metadata);
    void onPdfReply(QNetworkReply* reply);

private:
    QString m_apiKey;
    int m_localMode = 0; // 0: Gemini, 1: Ollama, 2: LM Studio
    
    ModelInfo m_embedModel;
    ModelInfo m_reasonModel;
    ModelInfo m_rerankModel;
    std::unique_ptr<IRerankClient> m_rerankClient;
    
    QNetworkAccessManager* m_networkManager;
    static GeminiApi* s_instance;
};

#endif // GEMINI_API_H
