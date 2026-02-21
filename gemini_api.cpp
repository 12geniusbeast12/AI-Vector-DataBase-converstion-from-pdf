#include "gemini_api.h"
#include <QtConcurrent/QtConcurrent>
#include <QFuture>
#include <cmath>
#include <QEventLoop>
#include <algorithm>

// Concrete strategy for local cross-encoders (LM Studio/Ollama)
class LocalRerankClient : public IRerankClient {
    QNetworkAccessManager* m_manager;
    ModelInfo m_model;
    
    // Rolling statistics for score calibration
    float m_mean = 0.5f;
    float m_stdDev = 0.15f;
    int m_sampleCount = 0;
    
    // Cross-session keys
    QString m_meanKey;
    QString m_stdKey;
    
    void updateStats(const QVector<float>& batchScores) {
        if (batchScores.isEmpty()) return;
        
        float sum = 0;
        for (float s : batchScores) sum += s;
        float batchMean = sum / batchScores.size();
        
        // Drift Detection & Recovery Loop
        if (m_sampleCount > 5) { // Wait for initial stability
            float drift = std::abs(batchMean - m_mean);
            if (drift > 0.4f) { // Significant drift threshold
                qDebug() << "⚠️ Reranker Drift Detected (" << drift << "). Resetting stats.";
                m_sampleCount = 0; // Trigger reset
            }
        }

        // Welford's algorithm or simple rolling average
        float alpha = 0.15f; 
        if (m_sampleCount == 0) {
            m_mean = batchMean;
        } else {
            m_mean = (1.0f - alpha) * m_mean + alpha * batchMean;
        }
        
        float sqSum = 0;
        for (float s : batchScores) sqSum += (s - m_mean) * (s - m_mean);
        float batchStd = std::sqrt(sqSum / batchScores.size());
        
        if (m_sampleCount == 0) {
            m_stdDev = qMax(0.01f, batchStd);
        } else {
            m_stdDev = (1.0f - alpha) * m_stdDev + alpha * qMax(0.01f, batchStd);
        }
        
        m_sampleCount++;
    }
    
    float normalize(float raw) {
        // 1. Z-Score
        float z = (raw - m_mean) / m_stdDev;
        
        // Phase 3B: Clamping & Outlier Rejection
        if (std::abs(z) > 5.0) return -1.0f; // Reject outlier
        z = std::clamp(z, -3.0f, 3.0f);      // Finalize Clamp
        
        // 2. Sigmoid to map to 0-1
        return 1.0f / (1.0f + std::exp(-z));
    }
    
public:
public:
    LocalRerankClient(QNetworkAccessManager* /* unused */, const ModelInfo& model) 
        : m_model(model) {
        // We do not store the passed manager. We instantiate thread-local managers
        // because this client executes on QtConcurrent background threads.
    }
        
    QVector<RerankResult> rerank(const QString& query, const QVector<VectorEntry>& candidates, int topK = 5) override {
        if (candidates.isEmpty()) return {};
        
        QVector<RerankResult> results;
        QString documentsBlock;
        for (int i = 0; i < candidates.size(); ++i) {
            documentsBlock += QString("[%1] %2\n").arg(i).arg(candidates[i].text.left(500));
        }
        
        QString prompt = QString("You are a relevance scoring engine. Given the query: \"%1\"\n"
                                 "Score each of the following documents from 0.0 (Irrelevant) to 1.0 (Highly Relevant) based on how well they answer the query.\n"
                                 "Return ONLY a JSON array of scores in the order provided.\n"
                                 "Example: [0.85, 0.12, 0.95]\n\n"
                                 "Documents:\n%2").arg(query).arg(documentsBlock);
                                 
        QJsonObject json;
        QUrl url(m_model.endpoint);
        
        if (m_model.engine == "Ollama") {
            json["model"] = m_model.name;
            json["prompt"] = prompt;
            json["stream"] = false;
            QJsonObject options;
            options["temperature"] = 0;
            json["options"] = options;
        } else { // LM Studio (OpenAI format)
            json["model"] = m_model.name;
            QJsonArray messages;
            messages.append(QJsonObject{{"role", "system"}, {"content", "You are a scoring engine. Return only JSON arrays."}});
            messages.append(QJsonObject{{"role", "user"}, {"content", prompt}});
            json["messages"] = messages;
            json["temperature"] = 0;
            if (url.isEmpty()) url = QUrl("http://127.0.0.1:1234/v1/chat/completions");
        }
        
        QNetworkRequest request(url);
        request.setHeader(QNetworkRequest::ContentTypeHeader, "application/json");
        
        // Phase 5 Readiness: Thread-safe local NetworkManager 
        // Must be constructed on the executing thread (worker thread)
        QNetworkAccessManager localManager;
        
        QEventLoop loop;
        QNetworkReply* reply = localManager.post(request, QJsonDocument(json).toJson());
        QObject::connect(reply, &QNetworkReply::finished, &loop, &QEventLoop::quit);
        loop.exec();
        
        if (reply->error() == QNetworkReply::NoError) {
            QByteArray data = reply->readAll();
            QJsonDocument doc = QJsonDocument::fromJson(data);
            QString responseText;
            
            if (m_model.engine == "Ollama") {
                responseText = doc.object()["response"].toString();
            } else {
                responseText = doc.object()["choices"].toArray()[0].toObject()["message"].toObject()["content"].toString();
            }
            
            int start = responseText.indexOf('[');
            int end = responseText.lastIndexOf(']');
            
            if (start != -1 && end != -1) {
                QString arrayStr = responseText.mid(start, end - start + 1);
                QJsonDocument arrayDoc = QJsonDocument::fromJson(arrayStr.toUtf8());
                QJsonArray scoresArr = arrayDoc.array();
                
                QVector<float> rawScores;
                for (int i = 0; i < scoresArr.size(); ++i) rawScores.append((float)scoresArr[i].toDouble());
                
                if (!checkConsistency(rawScores)) {
                    qDebug() << "⚠️ Reranker Consistency Failure: Low variance in batch scores. Skipping calibration update.";
                    if (GeminiApi::instance()) {
                        emit GeminiApi::instance()->anomalyDetected("Reranker Anomaly", 
                            "The model is producing highly uniform scores. This may indicate a 'frozen' state. Recalibration recommended.");
                    }
                } else {
                    updateStats(rawScores);
                }
                
                for (int i = 0; i < qMin((int)rawScores.size(), candidates.size()); ++i) {
                    float score = normalize(rawScores[i]);
                    if (score < 0) continue; // Skip outliers
                    
                    RerankResult res;
                    res.chunkId = candidates[i].id;
                    res.score = score;
                    res.originalRank = i;
                    results.append(res);
                }
            }
        }
        reply->deleteLater();
        
        std::sort(results.begin(), results.end(), [](const RerankResult& a, const RerankResult& b) {
            return a.score > b.score;
        });
        
        if (results.size() > topK) results.resize(topK);
        return results;
    }
    
    QFuture<QVector<RerankResult>> rerankAsync(const QString& query, const QVector<VectorEntry>& candidates, int topK = 5) override {
        return QtConcurrent::run([this, query, candidates, topK]() {
            return rerank(query, candidates, topK);
        });
    }
    
    // Phase 3B: Cross-Session Persistence
    void loadStats(float mean, float stdDev) override {
        if (stdDev > 0) {
            m_mean = mean;
            m_stdDev = stdDev;
            m_sampleCount = 10; // Assume stable start
            qDebug() << "📊 Reranker stats loaded: Mean=" << m_mean << "StdDev=" << m_stdDev;
        }
    }
    
    void saveStats(float& mean, float& stdDev) override {
        mean = m_mean;
        stdDev = m_stdDev;
    }
    
    // Phase 3B: Consistency Check
    bool checkConsistency(const QVector<float>& scores) {
        if (scores.isEmpty()) return true;
        float var = 0;
        for (float s : scores) var += (s - 0.5f) * (s - 0.5f);
        if (var < 0.001f) return false; // Model is outputting identical values
        return true;
    }
};

GeminiApi* GeminiApi::s_instance = nullptr;

GeminiApi::GeminiApi(const QString& apiKey, QObject *parent) 
    : QObject(parent), m_apiKey(apiKey) {
    s_instance = this;
    m_networkManager = new QNetworkAccessManager(this);
}

void GeminiApi::setRerankModel(const ModelInfo& model) {
    m_rerankModel = model;
    // Instantiate the appropriate strategy
    if (model.engine == "Ollama" || model.engine == "LMStudio") {
        m_rerankClient = std::make_unique<LocalRerankClient>(m_networkManager, model);
    }
}

void GeminiApi::updateRerankerStats(float mean, float stdDev) {
    if (m_rerankClient) {
        m_rerankClient->loadStats(mean, stdDev);
    }
}

void GeminiApi::setLocalMode(int mode) {
    m_localMode = mode;
    qDebug() << "GeminiApi provider changed to mode:" << m_localMode;
}

void GeminiApi::getEmbeddings(const QString& text, const QMap<QString, QVariant>& metadata) {
    if (text.trimmed().isEmpty()) {
        emit embeddingsReady(text, QVector<float>(), metadata);
        return;
    }

    QUrl url;
    QJsonObject json;

    if (m_embedModel.engine == "Ollama") { // Ollama
        url = QUrl("http://127.0.0.1:11434/api/embeddings");
        json["model"] = m_embedModel.name.isEmpty() ? "nomic-embed-text" : m_embedModel.name;
        json["prompt"] = text;
    } else if (m_embedModel.engine == "LMStudio") { // LM Studio
        url = QUrl("http://127.0.0.1:1234/v1/embeddings");
        json["model"] = m_embedModel.name;
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
    if (m_reasonModel.engine == "Ollama") { // Ollama
        url = QUrl("http://127.0.0.1:11434/api/generate");
    } else if (m_reasonModel.engine == "LMStudio") { // LM Studio
        url = QUrl("http://127.0.0.1:1234/v1/chat/completions");
    } else { // Gemini
        QString cleanId = m_reasonModel.name.isEmpty() ? "models/gemini-1.5-flash" : m_reasonModel.name;
        if (!cleanId.startsWith("models/")) cleanId = "models/" + cleanId;
        url = QUrl("https://generativelanguage.googleapis.com/v1beta/" + cleanId + ":generateContent?key=" + m_apiKey);
    }

    QJsonObject json;
    QString prompt = QString("Summarize the following textbook section into a single concise paragraph (max 3 sentences). Focus on core concepts and terminology. \n\n Content: %1").arg(text);

    if (m_reasonModel.engine == "Gemini" || m_reasonModel.engine.isEmpty()) { // Gemini
        QJsonObject content;
        QJsonArray parts;
        parts.append(QJsonObject{{"text", prompt}});
        content["parts"] = parts;
        QJsonArray contents;
        contents.append(content);
        json["contents"] = contents;
    } else if (m_reasonModel.engine == "Ollama") { // Ollama
        json["model"] = m_reasonModel.name.isEmpty() ? "llama3" : m_reasonModel.name;
        json["prompt"] = prompt;
        json["stream"] = false;
    } else { // LM Studio
        json["model"] = m_reasonModel.name.isEmpty() ? "local-model" : m_reasonModel.name;
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
            emit summaryReady("", metadata); // Unblock queue
            reply->deleteLater();
            return;
        }

        QByteArray data = reply->readAll();
        QJsonDocument doc = QJsonDocument::fromJson(data);
        QString summary;

        if (m_reasonModel.engine == "Gemini" || m_reasonModel.engine.isEmpty()) { // Gemini
            summary = doc.object()["candidates"].toArray()[0].toObject()["content"].toObject()["parts"].toArray()[0].toObject()["text"].toString();
        } else if (m_reasonModel.engine == "Ollama") { // Ollama
            summary = doc.object()["response"].toString();
        } else { // LM Studio
            summary = doc.object()["choices"].toArray()[0].toObject()["message"].toObject()["content"].toString();
        }

        emit summaryReady(summary.trimmed(), metadata);
    reply->deleteLater();
    });
}

void GeminiApi::synthesizeResponse(const QString& query, const QVector<SourceContext>& contexts, const QMap<QString, QVariant>& metadata) {
    auto cosineSim = [](const QVector<float>& v1, const QVector<float>& v2) -> float {
        if (v1.isEmpty() || v2.isEmpty() || v1.size() != v2.size()) return 0.0f;
        double dot = 0.0, n1 = 0.0, n2 = 0.0;
        for (int i = 0; i < v1.size(); ++i) {
            dot += v1[i] * v2[i];
            n1 += v1[i] * v1[i];
            n2 += v2[i] * v2[i];
        }
        return (n1 > 0 && n2 > 0) ? (float)(dot / (sqrt(n1) * sqrt(n2))) : 0.0f;
    };

    // Phase 4.2: Semantic Fact Clustering
    QVector<QVector<int>> clusters; // Indices of contexts
    QSet<int> assigned;
    
    for (int i = 0; i < contexts.size(); ++i) {
        if (assigned.contains(i)) continue;
        QVector<int> currentCluster;
        currentCluster.append(i);
        assigned.insert(i);
        
        for (int j = i + 1; j < contexts.size(); ++j) {
            if (assigned.contains(j)) continue;
            if (cosineSim(contexts[i].embedding, contexts[j].embedding) > 0.85f) {
                currentCluster.append(j);
                assigned.insert(j);
            }
        }
        clusters.append(currentCluster);
    }

    QUrl url;
    if (m_localMode == 1) url = QUrl("http://127.0.0.1:11434/api/generate");
    else if (m_localMode == 2) url = QUrl("http://127.0.0.1:1234/v1/chat/completions");
    else {
        QString cleanId = m_reasonModel.name.isEmpty() ? "models/gemini-1.5-flash" : m_reasonModel.name;
        if (!cleanId.startsWith("models/")) cleanId = "models/" + cleanId;
        url = QUrl("https://generativelanguage.googleapis.com/v1beta/" + cleanId + ":generateContent?key=" + m_apiKey);
    }

    QString contextBlock;
    for (int i = 0; i < clusters.size(); ++i) {
        const auto& cluster = clusters[i];
        contextBlock += QString("[FACT UNIT %1]\n").arg(i + 1);
        for (int idx : cluster) {
            const SourceContext& ctx = contexts[idx];
            contextBlock += QString("- Source [%1] (%2, Trust: %3): %4\n")
                                .arg(ctx.promptIndex)
                                .arg(ctx.docName)
                                .arg(ctx.trustScore, 0, 'f', 2)
                                .arg(ctx.chunkText);
        }
        contextBlock += "\n";
    }

    QString prompt = QString("You are a high-trust research synthesis engine. Based ONLY on the following FACT UNITS, provide a grounded answer.\n"
                             "Each fact unit contains multiple supporting sources. Use Source [ID] for citations.\n"
                             "If fact units conflict (e.g. different dates or opposing claims), YOU MUST mention the conflict.\n"
                             "Return your answer ONLY as valid JSON.\n\n"
                             "Format:\n"
                             "{\n"
                             "  \"answer\": [\n"
                             "    {\"statement\": \"<claim text here>\", \"sources\": [<source_id1>, <source_id2>]}\n"
                             "  ]\n"
                             "}\n\n"
                             "Context:\n%1\n\nQuery: %2")
                             .arg(contextBlock).arg(query);

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
        json["model"] = m_reasonModel.name.isEmpty() ? "llama3" : m_reasonModel.name;
        json["prompt"] = prompt;
        json["stream"] = false;
    } else { // LM Studio
        json["model"] = m_reasonModel.name.isEmpty() ? "local-model" : m_reasonModel.name;
        QJsonArray messages;
        messages.append(QJsonObject{{"role", "user"}, {"content", prompt}});
        json["messages"] = messages;
    }

    QNetworkRequest request(url);
    request.setHeader(QNetworkRequest::ContentTypeHeader, "application/json");

    QNetworkReply* reply = m_networkManager->post(request, QJsonDocument(json).toJson());
    connect(reply, &QNetworkReply::finished, this, [this, reply, contexts, metadata]() {
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
        
        QVector<ClaimNode> claims;
        
        if (report.contains("No grounded answer found", Qt::CaseInsensitive)) {
            emit synthesisReady(claims, contexts, metadata);
            reply->deleteLater();
            return;
        }
        
        int startIdx = report.indexOf('{');
        int endIdx = report.lastIndexOf('}');
        
        if (startIdx != -1 && endIdx != -1 && endIdx > startIdx) {
            QString jsonStr = report.mid(startIdx, endIdx - startIdx + 1);
            
            int depth = 0;
            for (QChar c : jsonStr) {
                if (c == '{') depth++;
                else if (c == '}') depth--;
            }
            
            if (depth == 0) {
                QJsonDocument outDoc = QJsonDocument::fromJson(jsonStr.toUtf8());
                QJsonArray arr = outDoc.object()["answer"].toArray();
                for (int i = 0; i < arr.size(); ++i) {
                    QJsonObject item = arr[i].toObject();
                    ClaimNode claim;
                    claim.statement = item["statement"].toString();
                    
                    QVector<int> validSources;
                    float totalConfidence = 0;
                    
                    if (item.contains("sources") && item["sources"].isArray()) {
                        QJsonArray srcArr = item["sources"].toArray();
                        for (int j = 0; j < srcArr.size(); ++j) {
                            int srcIdx = srcArr[j].toInt();
                            bool found = false;
                            float cScore = 0.0f;
                            for (const SourceContext& ctx : contexts) {
                                if (ctx.promptIndex == srcIdx) {
                                    found = true;
                                    cScore = ctx.finalScore;
                                    break;
                                }
                            }
                            if (found) {
                                validSources.append(srcIdx);
                                totalConfidence += cScore;
                            }
                        }
                    }
                    
                    claim.sourceIndices = validSources;
                    if (!validSources.isEmpty()) {
                        claim.confidence = totalConfidence / validSources.size();
                    } else if (!contexts.isEmpty()){
                         claim.confidence = contexts[0].finalScore * 0.5f; // Fallback so it doesn't render completely low confidence if model missed citing it
                    }
                    
                    if (!claim.statement.isEmpty()) {
                        claims.append(claim);
                    }
                }
            } else {
                qDebug() << "JSON Payload from model lacked balanced braces:\n" << report;
            }
        } else {
            qDebug() << "No JSON structural wrapper found in LLM payload:\n" << report;
        }

        emit synthesisReady(claims, contexts, metadata);
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

    if (m_embedModel.engine != "Gemini" && !m_embedModel.engine.isEmpty()) {
        QJsonArray values;
        if (m_embedModel.engine == "LMStudio") { // LM Studio (OpenAI format)
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
        finalMetadata["model_sig"] = m_embedModel.name.isEmpty() ? (m_localMode == 1 ? "nomic-embed-text" : "gemini-embedding-001") : m_embedModel.name;
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
                ModelInfo info{name, "Ollama", "http://localhost:11434", "(Ollama Native)", {}};
                if (name.toLower().contains("embed") || name.toLower().contains("nomic")) {
                    info.capabilities.insert(ModelCapability::Embedding);
                } else if (name.toLower().contains("rerank") || name.toLower().contains("bge")) {
                    info.capabilities.insert(ModelCapability::Rerank);
                } else {
                    info.capabilities.insert(ModelCapability::Chat);
                    info.capabilities.insert(ModelCapability::Summary);
                }
                state->models.append(info);
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
                ModelInfo info{id, "LMStudio", "http://127.0.0.1:1234", "(Local Shared)", {}};
                if (id.toLower().contains("embed") || id.toLower().contains("nomic")) {
                    info.capabilities.insert(ModelCapability::Embedding);
                } else if (id.toLower().contains("rerank") || id.toLower().contains("bge")) {
                    info.capabilities.insert(ModelCapability::Rerank);
                } else {
                    info.capabilities.insert(ModelCapability::Chat);
                    info.capabilities.insert(ModelCapability::Summary);
                }
                state->models.append(info);
            }
        } else {
            qDebug() << "LM Studio discovery failed (Port 1234):" << lReply->errorString();
            qDebug() << "Note: Make sure LM Studio Local Server is STARTED on port 1234.";
        }
        lReply->deleteLater();
        check();
    });
}

#include <QFutureWatcher>

void GeminiApi::rerank(const QString& query, const QVector<VectorEntry>& candidates) {
    if (!m_rerankClient || candidates.isEmpty()) {
        emit rerankingReady(candidates);
        return;
    }

    QFuture<QVector<RerankResult>> future = m_rerankClient->rerankAsync(query, candidates);
    auto* watcher = new QFutureWatcher<QVector<RerankResult>>(this);
    
    connect(watcher, &QFutureWatcher<QVector<RerankResult>>::finished, this, [this, watcher, candidates]() {
        QVector<RerankResult> results = watcher->result();
        QVector<VectorEntry> rerankedResults;
        
        for (const auto& res : results) {
            for (const auto& entry : candidates) {
                if (entry.id == res.chunkId) {
                    VectorEntry updated = entry;
                    updated.score = res.score;
                    updated.rerankRank = res.originalRank;
                    rerankedResults.append(updated);
                    break;
                }
            }
        }
        
        // Final sanity check: if results is empty due to error, return original
        if (rerankedResults.isEmpty() && !candidates.isEmpty()) {
            emit rerankingReady(candidates);
        } else {
            // Phase 3B: Broadcast updated stats for persistence
            float m, s;
            m_rerankClient->saveStats(m, s);
            emit rerankerStatsUpdated(m, s);
            
            emit rerankingReady(rerankedResults);
        }
        watcher->deleteLater();
    });
    
    watcher->setFuture(future);
}
