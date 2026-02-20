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

struct ModelInfo {
    QString name;
    QString engine; // Gemini, Ollama, LMStudio
    QString endpoint;
};

class GeminiApi : public QObject {
    Q_OBJECT
public:
    explicit GeminiApi(const QString& apiKey = "", QObject *parent = nullptr);

    void setApiKey(const QString& key) { m_apiKey = key; }
    void setLocalMode(int mode);
    void setSelectedModel(const ModelInfo& model) { m_selectedModel = model; }
    
    void processPdf(const QString& filePath);
    void getEmbeddings(const QString& text, const QMap<QString, QVariant>& metadata = {});
    void rerank(const QString& query, const QVector<VectorEntry>& candidates);
    void discoverModels();
    
signals:
    void pdfProcessed(const QString& text);
    void embeddingsReady(const QString& text, const QVector<float>& embedding, const QMap<QString, QVariant>& metadata = {});
    void rerankingReady(const QVector<VectorEntry>& rerankedResults);
    void discoveredModelsReady(const QVector<ModelInfo>& models);
    void errorOccurred(const QString& error);

private slots:
    void onEmbeddingsReply(QNetworkReply* reply, const QString& originalText, const QMap<QString, QVariant>& metadata);
    void onRerankReply(QNetworkReply* reply, QVector<VectorEntry> candidates);
    void onPdfReply(QNetworkReply* reply);

private:
    QString m_apiKey;
    int m_localMode = 0; 
    ModelInfo m_selectedModel;
    QNetworkAccessManager* m_networkManager;
};

#endif // GEMINI_API_H
