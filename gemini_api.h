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

enum class ModelCapability {
    Embedding,
    Chat,
    Rerank,
    Summary
};

struct ModelInfo {
    QString name;
    QString engine; // Gemini, Ollama, LMStudio
    QString endpoint;
    QSet<ModelCapability> capabilities;
};

class GeminiApi : public QObject {
    Q_OBJECT
public:
    explicit GeminiApi(const QString& apiKey = "", QObject *parent = nullptr);

    void setApiKey(const QString& key) { m_apiKey = key; }
    void setLocalMode(int mode);
    void setEmbeddingModel(const ModelInfo& model) { m_embedModel = model; }
    void setReasoningModel(const ModelInfo& model) { m_reasonModel = model; }
    
    void processPdf(const QString& filePath);
    void getEmbeddings(const QString& text, const QMap<QString, QVariant>& metadata = {});
    void generateSummary(const QString& text, const QMap<QString, QVariant>& metadata = {});
    void synthesizeResponse(const QString& query, const QStringList& contexts, const QMap<QString, QVariant>& metadata = {});
    void rerank(const QString& query, const QVector<VectorEntry>& candidates);
    void discoverModels();
    
signals:
    void pdfProcessed(const QString& text);
    void embeddingsReady(const QString& text, const QVector<float>& embedding, const QMap<QString, QVariant>& metadata = {});
    void summaryReady(const QString& summary, const QMap<QString, QVariant>& metadata = {});
    void synthesisReady(const QString& report, const QMap<QString, QVariant>& metadata = {});
    void rerankingReady(const QVector<VectorEntry>& rerankedResults);
    void discoveredModelsReady(const QVector<ModelInfo>& models);
    void errorOccurred(const QString& error);

private slots:
    void onEmbeddingsReply(QNetworkReply* reply, const QString& originalText, const QMap<QString, QVariant>& metadata);
    void onPdfReply(QNetworkReply* reply);

private:
    QString m_apiKey;
    int m_localMode = 0; 
    ModelInfo m_embedModel;
    ModelInfo m_reasonModel;
    QNetworkAccessManager* m_networkManager;
};

#endif // GEMINI_API_H
