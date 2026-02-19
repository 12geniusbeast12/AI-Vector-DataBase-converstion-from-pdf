#include "gemini_api.h"
#include <QUrl>
#include <QNetworkRequest>
#include <QFile>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonArray>
#include <QDebug>
#include <memory>

GeminiApi::GeminiApi(const QString& apiKey, QObject *parent) 
    : QObject(parent), m_apiKey(apiKey) {
    m_networkManager = new QNetworkAccessManager(this);
}

void GeminiApi::setLocalMode(int mode) {
    m_localMode = mode;
    qDebug() << "GeminiApi provider changed to mode:" << m_localMode;
}

void GeminiApi::getEmbeddings(const QString& text) {
    QUrl url;
    QJsonObject json;

    if (m_localMode == 1) { // Ollama
        url = QUrl("http://localhost:11434/api/embeddings");
        json["model"] = m_selectedModel.name.isEmpty() ? "nomic-embed-text" : m_selectedModel.name;
        json["prompt"] = text;
    } else if (m_localMode == 2) { // LM Studio
        url = QUrl("http://localhost:1234/v1/embeddings");
        json["model"] = m_selectedModel.name;
        json["input"] = text;
    } else { // Gemini
        url = QUrl("https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:embedContent?key=" + m_apiKey);
        QJsonObject content;
        QJsonArray parts;
        parts.append(QJsonObject{{"text", text}});
        content["parts"] = parts;
        json["content"] = content;
        json["task_type"] = "RETRIEVAL_DOCUMENT";
    }

    QNetworkRequest request(url);
    request.setHeader(QNetworkRequest::ContentTypeHeader, "application/json");

    qDebug() << "Requesting embeddings for text (length):" << text.length() << "using url:" << url.toString();
    QNetworkReply* reply = m_networkManager->post(request, QJsonDocument(json).toJson());
    connect(reply, &QNetworkReply::finished, this, [this, reply, text]() {
        onEmbeddingsReply(reply, text);
    });
}

void GeminiApi::processPdf(const QString& filePath) {
    if (m_localMode > 0) {
        emit errorOccurred("Local PDF OCR is not yet implemented. Please use Gemini for PDF extraction, then switch to Local for offline search.");
        return;
    }

    QUrl url("https://generativelanguage.googleapis.com/v1beta/models/gemini-flash-latest:generateContent?key=" + m_apiKey);
    QNetworkRequest request(url);
    request.setHeader(QNetworkRequest::ContentTypeHeader, "application/json");

    QFile file(filePath);
    if (!file.open(QIODevice::ReadOnly)) {
        emit errorOccurred("Could not open file: " + filePath);
        return;
    }
    QByteArray fileData = file.readAll();
    file.close();

    QJsonObject json;
    QJsonArray contents;
    QJsonObject contentObj;
    QJsonArray parts;
    
    QJsonObject inlineData;
    inlineData["mime_type"] = "application/pdf";
    inlineData["data"] = QString(fileData.toBase64());
    
    parts.append(QJsonObject{{"inline_data", inlineData}});
    parts.append(QJsonObject{{"text", "Extract all text from this PDF exactly as it is."}});
    
    contentObj["parts"] = parts;
    contents.append(contentObj);
    json["contents"] = contents;

    qDebug() << "Sending PDF to Gemini for extraction...";
    QNetworkReply* reply = m_networkManager->post(request, QJsonDocument(json).toJson());
    connect(reply, &QNetworkReply::finished, this, [this, reply]() {
        onPdfReply(reply);
    });
}

void GeminiApi::onEmbeddingsReply(QNetworkReply* reply, const QString& originalText) {
    if (reply->error() != QNetworkReply::NoError) {
        emit errorOccurred("Embedding error: " + reply->errorString());
        reply->deleteLater();
        return;
    }

    QByteArray data = reply->readAll();
    QJsonDocument doc = QJsonDocument::fromJson(data);
    QJsonObject obj = doc.object();
    QVector<float> embedding;

    if (m_localMode > 0) {
        QJsonArray values;
        if (m_localMode == 2) { // LM Studio (OpenAI format)
            values = obj["data"].toArray()[0].toObject()["embedding"].toArray();
        } else { // Ollama
            values = obj["embedding"].toArray();
        }
        
        for (const QJsonValue& val : values) {
            embedding.append(static_cast<float>(val.toDouble()));
        }
    } else {
        QJsonObject embeddingObj = obj["embedding"].toObject();
        if (embeddingObj.isEmpty() && obj.contains("embeddings")) {
            embeddingObj = obj["embeddings"].toArray().first().toObject();
        }
        
        QJsonArray values = embeddingObj["values"].toArray();
        for (const QJsonValue& val : values) {
            embedding.append(static_cast<float>(val.toDouble()));
        }
    }

    if (!embedding.isEmpty()) {
        emit embeddingsReady(originalText, embedding);
    }
    reply->deleteLater();
}

void GeminiApi::onPdfReply(QNetworkReply* reply) {
    if (reply->error() != QNetworkReply::NoError) {
        int statusCode = reply->attribute(QNetworkRequest::HttpStatusCodeAttribute).toInt();
        if (statusCode == 429) {
            emit errorOccurred("Rate limit hit (429). PDF might be too large. Try a shorter file.");
        } else {
            emit errorOccurred("PDF Processing error: " + reply->errorString());
        }
        reply->deleteLater();
        return;
    }

    QByteArray data = reply->readAll();
    QJsonDocument doc = QJsonDocument::fromJson(data);
    QJsonObject obj = doc.object();
    
    QString extractedText;
    QJsonArray candidates = obj["candidates"].toArray();
    if (!candidates.isEmpty()) {
        QJsonArray parts = candidates[0].toObject()["content"].toObject()["parts"].toArray();
        if (!parts.isEmpty()) {
            extractedText = parts[0].toObject()["text"].toString();
        }
    }

    qDebug() << "Extracted text length:" << extractedText.length();
    if (extractedText.length() > 0) {
        qDebug() << "Extracted snippet:" << extractedText.left(100);
    }

    if (!extractedText.isEmpty()) {
        emit pdfProcessed(extractedText);
    } else {
        qDebug() << "Extraction failed. Response:" << data;
        emit errorOccurred("No text extracted from PDF. Check if the PDF is password-protected or scanned without OCR.");
    }
    reply->deleteLater();
}

void GeminiApi::discoverModels() {
    // 1. Try Ollama
    QNetworkRequest ollamaReq(QUrl("http://localhost:11434/api/tags"));
    QNetworkReply* oReply = m_networkManager->get(ollamaReq);
    
    // 2. Try LM Studio
    QNetworkRequest lmsReq(QUrl("http://localhost:1234/v1/models"));
    QNetworkReply* lReply = m_networkManager->get(lmsReq);

    struct State {
        QVector<ModelInfo> models;
        int count = 2;
    };
    auto state = std::make_shared<State>();

    auto check = [this, state]() {
        if (--state->count == 0) {
            emit discoveredModelsReady(state->models);
        }
    };

    connect(oReply, &QNetworkReply::finished, this, [this, oReply, state, check]() {
        if (oReply->error() == QNetworkReply::NoError) {
            QJsonDocument doc = QJsonDocument::fromJson(oReply->readAll());
            QJsonArray arr = doc.object()["models"].toArray();
            for (const QJsonValue& v : arr) {
                QString name = v.toObject()["name"].toString();
                if (name.contains("embed")) {
                    state->models.append({name, "Ollama", "http://localhost:11434"});
                }
            }
        }
        oReply->deleteLater();
        check();
    });

    connect(lReply, &QNetworkReply::finished, this, [this, lReply, state, check]() {
        if (lReply->error() == QNetworkReply::NoError) {
            QJsonDocument doc = QJsonDocument::fromJson(lReply->readAll());
            QJsonArray data = doc.object()["data"].toArray();
            for (const QJsonValue& v : data) {
                QString id = v.toObject()["id"].toString();
                state->models.append({id, "LMStudio", "http://localhost:1234"});
            }
        }
        lReply->deleteLater();
        check();
    });
}
