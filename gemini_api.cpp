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

void GeminiApi::getEmbeddings(const QString& text, const QMap<QString, QVariant>& metadata) {
    QUrl url;
    QJsonObject json;

    if (m_localMode == 1) { // Ollama
        url = QUrl("http://127.0.0.1:11434/api/embeddings");
        json["model"] = m_selectedModel.name.isEmpty() ? "nomic-embed-text" : m_selectedModel.name;
        json["prompt"] = text;
    } else if (m_localMode == 2) { // LM Studio
        url = QUrl("http://127.0.0.1:1234/v1/embeddings");
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
    connect(reply, &QNetworkReply::finished, this, [this, reply, text, metadata]() {
        onEmbeddingsReply(reply, text, metadata);
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

void GeminiApi::generateSummary(const QString& text, const QMap<QString, QVariant>& metadata) {
    QUrl url;
    if (m_localMode == 1) { // Ollama
        url = QUrl("http://127.0.0.1:11434/api/generate");
    } else if (m_localMode == 2) { // LM Studio
        url = QUrl("http://127.0.0.1:1234/v1/chat/completions");
    } else { // Gemini
        url = QUrl("https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key=" + m_apiKey);
    }

    QJsonObject json;
    QString prompt = QString("Summarize the following textbook section into a single concise paragraph (max 3 sentences). Focus on core concepts and terminology. \n\n Content: %1").arg(text);

    if (m_localMode == 0) { // Gemini
        QJsonObject content;
        QJsonArray parts;
        parts.append(QJsonObject{{"text", prompt}});
        content["parts"] = parts;
        QJsonArray contents;
        contents.append(content);
        json["contents"] = contents;
    } else if (m_localMode == 1) { // Ollama
        json["model"] = m_selectedModel.name.isEmpty() ? "llama3" : m_selectedModel.name;
        json["prompt"] = prompt;
        json["stream"] = false;
    } else { // LM Studio
        QString modelStr = m_selectedModel.name;
        if (modelStr.isEmpty() || modelStr.toLower().contains("embed") || modelStr.toLower().contains("nomic")) {
            modelStr = "local-model";
        }
        json["model"] = modelStr;
        QJsonArray messages;
        messages.append(QJsonObject{{"role", "user"}, {"content", prompt}});
        json["messages"] = messages;
    }

    QNetworkRequest request(url);
    request.setHeader(QNetworkRequest::ContentTypeHeader, "application/json");

    QNetworkReply* reply = m_networkManager->post(request, QJsonDocument(json).toJson());
    connect(reply, &QNetworkReply::finished, this, [this, reply, metadata]() {
        if (reply->error() != QNetworkReply::NoError) {
            emit errorOccurred("Summary error: " + reply->errorString());
            reply->deleteLater();
            return;
        }

        QByteArray data = reply->readAll();
        QJsonDocument doc = QJsonDocument::fromJson(data);
        QString summary;

        if (m_localMode == 0) { // Gemini
            summary = doc.object()["candidates"].toArray()[0].toObject()["content"].toObject()["parts"].toArray()[0].toObject()["text"].toString();
        } else if (m_localMode == 1) { // Ollama
            summary = doc.object()["response"].toString();
        } else { // LM Studio
            summary = doc.object()["choices"].toArray()[0].toObject()["message"].toObject()["content"].toString();
        }

        emit summaryReady(summary.trimmed(), metadata);
        reply->deleteLater();
    });
}

void GeminiApi::synthesizeResponse(const QString& query, const QStringList& contexts, const QMap<QString, QVariant>& metadata) {
    QUrl url;
    if (m_localMode == 1) { // Ollama
        url = QUrl("http://127.0.0.1:11434/api/generate");
    } else if (m_localMode == 2) { // LM Studio
        url = QUrl("http://127.0.0.1:1234/v1/chat/completions");
    } else { // Gemini
        url = QUrl("https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key=" + m_apiKey);
    }

    QString contextBlock = contexts.join("\n\n---\n\n");
    QString prompt = QString("You are a technical document assistant. Based on the following textbook snippets from different sections, provide a comprehensive synthesized answer to the user's query. Bridge concepts across sections if necessary. \n\n"
                             "USER QUERY: %1 \n\n"
                             "DOCUMENT CONTEXT: \n %2 \n\n"
                             "SYNTHESIS REPORT:").arg(query).arg(contextBlock);

    QJsonObject json;
    if (m_localMode == 0) { // Gemini
        QJsonObject content;
        QJsonArray parts;
        parts.append(QJsonObject{{"text", prompt}});
        content["parts"] = parts;
        QJsonArray contents;
        contents.append(content);
        json["contents"] = contents;
    } else if (m_localMode == 1) { // Ollama
        json["model"] = m_selectedModel.name.isEmpty() ? "llama3" : m_selectedModel.name;
        json["prompt"] = prompt;
        json["stream"] = false;
    } else { // LM Studio
        QString modelStr = m_selectedModel.name;
        if (modelStr.isEmpty() || modelStr.toLower().contains("embed") || modelStr.toLower().contains("nomic")) {
            modelStr = "local-model";
        }
        json["model"] = modelStr;
        QJsonArray messages;
        messages.append(QJsonObject{{"role", "user"}, {"content", prompt}});
        json["messages"] = messages;
    }

    QNetworkRequest request(url);
    request.setHeader(QNetworkRequest::ContentTypeHeader, "application/json");

    QNetworkReply* reply = m_networkManager->post(request, QJsonDocument(json).toJson());
    connect(reply, &QNetworkReply::finished, this, [this, reply, metadata]() {
        if (reply->error() != QNetworkReply::NoError) {
            emit errorOccurred("Synthesis error: " + reply->errorString());
            reply->deleteLater();
            return;
        }

        QByteArray data = reply->readAll();
        QJsonDocument doc = QJsonDocument::fromJson(data);
        QString report;

        if (m_localMode == 0) { // Gemini
            report = doc.object()["candidates"].toArray()[0].toObject()["content"].toObject()["parts"].toArray()[0].toObject()["text"].toString();
        } else if (m_localMode == 1) { // Ollama
            report = doc.object()["response"].toString();
        } else { // LM Studio
            report = doc.object()["choices"].toArray()[0].toObject()["message"].toObject()["content"].toString();
        }

        emit synthesisReady(report.trimmed(), metadata);
        reply->deleteLater();
    });
}

void GeminiApi::onEmbeddingsReply(QNetworkReply* reply, const QString& originalText, const QMap<QString, QVariant>& metadata) {
    if (reply->error() != QNetworkReply::NoError) {
        QString errorMsg = reply->errorString();
        int statusCode = reply->attribute(QNetworkRequest::HttpStatusCodeAttribute).toInt();
        
        if (statusCode == 400 || errorMsg.contains("Bad Request")) {
            errorMsg = "Bad Request (400): Ensure you have loaded an EMBEDDING model (like 'nomic-embed-text') in LM Studio. Note: General chat models (like Qwen or Gemma) usually fail to generate embeddings on this endpoint.";
        }
        
        emit errorOccurred("Embedding error: " + errorMsg);
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
            QJsonArray dataArr = obj["data"].toArray();
            if (!dataArr.isEmpty()) {
                values = dataArr[0].toObject()["embedding"].toArray();
            }
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

    if (embedding.isEmpty()) {
        emit errorOccurred("Embeddings returned empty. Please verify you are using an embedding-compatible model in your local AI server.");
    } else {
        QMap<QString, QVariant> finalMetadata = metadata;
        finalMetadata["model_sig"] = m_selectedModel.name.isEmpty() ? (m_localMode == 1 ? "nomic-embed-text" : "gemini-embedding-001") : m_selectedModel.name;
        emit embeddingsReady(originalText, embedding, finalMetadata);
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
    // 1. Try Ollama (using 127.0.0.1 to be more reliable on Windows)
    QNetworkRequest ollamaReq(QUrl("http://127.0.0.1:11434/api/tags"));
    QNetworkReply* oReply = m_networkManager->get(ollamaReq);
    
    // 2. Try LM Studio
    QNetworkRequest lmsReq(QUrl("http://127.0.0.1:1234/v1/models"));
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
            qDebug() << "Ollama discovery found" << arr.size() << "models";
            for (const QJsonValue& v : arr) {
                QString name = v.toObject()["name"].toString();
                if (name.contains("embed")) {
                    state->models.append({name, "Ollama", "http://localhost:11434"});
                }
            }
        } else {
            qDebug() << "Ollama discovery failed:" << oReply->errorString();
        }
        oReply->deleteLater();
        check();
    });

    connect(lReply, &QNetworkReply::finished, this, [this, lReply, state, check]() {
        if (lReply->error() == QNetworkReply::NoError) {
            QJsonDocument doc = QJsonDocument::fromJson(lReply->readAll());
            QJsonArray data = doc.object()["data"].toArray();
            qDebug() << "LM Studio discovery found" << data.size() << "models";
            for (const QJsonValue& v : data) {
                QString id = v.toObject()["id"].toString();
                state->models.append({id, "LMStudio", "http://127.0.0.1:1234"});
            }
        } else {
            qDebug() << "LM Studio discovery failed (Port 1234):" << lReply->errorString();
            qDebug() << "Note: Make sure LM Studio Local Server is STARTED on port 1234.";
        }
        lReply->deleteLater();
        check();
    });
}
void GeminiApi::rerank(const QString& query, const QVector<VectorEntry>& candidates) {
    if (m_localMode == 0 || candidates.isEmpty()) {
        emit rerankingReady(candidates);
        return;
    }

    struct RerankState {
        QVector<VectorEntry> results;
        int remaining;
    };
    auto state = std::make_shared<RerankState>();
    state->results = candidates;
    state->remaining = candidates.size();

    for (int i = 0; i < candidates.size(); ++i) {
        QUrl url;
        QJsonObject json;
        if (m_localMode == 1) { // Ollama
            url = QUrl("http://127.0.0.1:11434/api/chat");
            QJsonArray messages;
            messages.append(QJsonObject{{"role", "user"}, {"content", QString("On a scale of 0 to 100, how relevant is this text to the query: '%1'?\nText: %2\nReply only with the number.").arg(query).arg(candidates[i].text)}});
            json["model"] = m_selectedModel.name.isEmpty() ? "llama3" : m_selectedModel.name;
            json["messages"] = messages;
            json["stream"] = false;
        } else { // LM Studio
            url = QUrl("http://127.0.0.1:1234/v1/chat/completions");
            QJsonArray messages;
            messages.append(QJsonObject{{"role", "user"}, {"content", QString("Rate relevance (0-100) of this text to query: '%1'\nText: %2\nOutput ONLY number.").arg(query).arg(candidates[i].text)}});
            QString model = m_selectedModel.name;
            if (model.isEmpty() || model.toLower().contains("embed") || model.toLower().contains("nomic")) {
                model = "local-model";
            }
            json["model"] = model;
            json["messages"] = messages;
        }

        QNetworkRequest request(url);
        request.setHeader(QNetworkRequest::ContentTypeHeader, "application/json");
        QNetworkReply* reply = m_networkManager->post(request, QJsonDocument(json).toJson());
        
        connect(reply, &QNetworkReply::finished, this, [this, reply, state, i]() {
            if (reply->error() == QNetworkReply::NoError) {
                QJsonObject obj = QJsonDocument::fromJson(reply->readAll()).object();
                QString response;
                if (m_localMode == 1) response = obj["message"].toObject()["content"].toString();
                else response = obj["choices"].toArray()[0].toObject()["message"].toObject()["content"].toString();
                
                // Extract number
                bool ok;
                double score = response.trimmed().split(QRegularExpression("[^0-9.]")).first().toDouble(&ok);
                if (ok) {
                    state->results[i].score = (state->results[i].score * 0.3) + (score / 100.0 * 0.7); // Weighted blend
                }
            }
            reply->deleteLater();
            if (--state->remaining == 0) {
                std::sort(state->results.begin(), state->results.end(), [](const VectorEntry& a, const VectorEntry& b) {
                    return a.score > b.score;
                });
                emit rerankingReady(state->results);
            }
        });
    }
}
