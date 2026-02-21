#include "vector_store.h"
#include <QSqlQuery>
#include <QRegularExpression>
#include <QMessageBox>
#include <QHeaderView>
#include <QVariant>
#include <QDebug>
#include <QCoreApplication>
#include <QSqlError>
#include <QFile>
#include <QDir>
#include <QTextStream>
#include <QtMath>
#include <QStandardPaths>
#include <QElapsedTimer>
#include <algorithm>

VectorStore::VectorStore(const QString& dbPath) : m_dbPath(dbPath) {}

VectorStore::~VectorStore() {
    if (m_db.isOpen()) {
        m_db.close();
    }
}

bool VectorStore::init() {
    QString connectionName = "VectorDBConnection";
    if (QSqlDatabase::contains(connectionName)) {
        m_db = QSqlDatabase::database(connectionName);
    } else {
        m_db = QSqlDatabase::addDatabase("QSQLITE", connectionName);
    }
    
    QString dataDir = QStandardPaths::writableLocation(QStandardPaths::AppDataLocation);
    QDir().mkpath(dataDir);
    
    if (m_dbPath == "vector_db.sqlite" || !m_dbPath.contains("/") && !m_dbPath.contains("\\")) {
        m_dbPath = dataDir + "/" + m_dbPath;
    }
    
    m_db.setDatabaseName(m_dbPath);
    qDebug() << "Initializing database at:" << m_dbPath;

    if (!m_db.open()) {
        qDebug() << "CRITICAL: Database open failed at" << m_dbPath << "Error:" << m_db.lastError().text();
        return false;
    }

    QSqlQuery q(m_db);
    
    // Check schema version
    q.exec("PRAGMA user_version");
    int version = 0;
    if (q.next()) version = q.value(0).toInt();
    qDebug() << "Current Schema Version:" << version;

    // Initialization (v0-v10 handled incrementally)
    if (version < 1) {
        q.exec("CREATE TABLE IF NOT EXISTS embeddings ("
               "id INTEGER PRIMARY KEY AUTOINCREMENT, "
               "source_file TEXT, "
               "text_chunk TEXT, "
               "vector_blob BLOB)");
        q.exec("PRAGMA user_version = 1");
    }

    if (version < 2) {
        q.exec("ALTER TABLE embeddings ADD COLUMN doc_id TEXT");
        q.exec("ALTER TABLE embeddings ADD COLUMN page_num INTEGER");
        q.exec("ALTER TABLE embeddings ADD COLUMN model_sig TEXT");
        q.exec("ALTER TABLE embeddings ADD COLUMN created_at DATETIME DEFAULT CURRENT_TIMESTAMP");
        q.exec("PRAGMA user_version = 2");
    }

    if (version < 6) {
        q.exec("ALTER TABLE embeddings ADD COLUMN chunk_idx INTEGER");
        q.exec("ALTER TABLE embeddings ADD COLUMN model_dim INTEGER");
        q.exec("ALTER TABLE embeddings ADD COLUMN token_count INTEGER");
        q.exec("ALTER TABLE embeddings ADD COLUMN doc_version TEXT");
        q.exec("CREATE VIRTUAL TABLE IF NOT EXISTS embeddings_fts USING fts5(text_chunk, content='embeddings', content_rowid='id')");
        q.exec("CREATE TABLE IF NOT EXISTS retrieval_logs ("
               "id INTEGER PRIMARY KEY AUTOINCREMENT, "
               "query TEXT, "
               "semantic_rank INTEGER, "
               "keyword_rank INTEGER, "
               "final_rank INTEGER, "
               "latency_embedding INTEGER, "
               "latency_search INTEGER, "
               "latency_fusion INTEGER, "
               "latency_rerank INTEGER, "
               "top_score REAL, "
               "created_at DATETIME DEFAULT CURRENT_TIMESTAMP)");
        q.exec("PRAGMA user_version = 6");
    }

    if (version < 7) {
        q.exec("ALTER TABLE embeddings ADD COLUMN chapter_title TEXT");
        q.exec("ALTER TABLE embeddings ADD COLUMN section_title TEXT");
        q.exec("ALTER TABLE embeddings ADD COLUMN chunk_type TEXT DEFAULT 'text'");
        q.exec("PRAGMA user_version = 7");
    }

    if (version < 8) {
        q.exec("ALTER TABLE embeddings ADD COLUMN heading_path TEXT");
        q.exec("ALTER TABLE embeddings ADD COLUMN heading_level INTEGER DEFAULT 0");
        q.exec("PRAGMA user_version = 8");
    }

    if (version < 9) {
        q.exec("ALTER TABLE embeddings ADD COLUMN heading_vec_blob BLOB");
        q.exec("PRAGMA user_version = 9");
    }

    if (version < 10) {
        q.exec("ALTER TABLE embeddings ADD COLUMN sentence_count INTEGER DEFAULT 0");
        q.exec("ALTER TABLE embeddings ADD COLUMN list_type TEXT");
        q.exec("ALTER TABLE embeddings ADD COLUMN list_length INTEGER DEFAULT 0");
        q.exec("PRAGMA user_version = 10");
    }

    // Migration to v11: Workspace Metadata & Guardrails
    if (version < 11) {
        q.exec("CREATE TABLE IF NOT EXISTS workspace_metadata ("
               "key TEXT PRIMARY KEY, "
               "value TEXT)");
        q.exec("PRAGMA user_version = 11");
        qDebug() << "Migrated database to v11 (Workspace Metadata).";
    }

    return true;
}

bool VectorStore::addEntry(const QString& text, const QVector<float>& embedding, 
                           const QString& sourceFile, const QString& docId, 
                           int pageNum, int chunkIdx, const QString& modelSig,
                           const QString& path, int level, const QString& chunkType,
                           int sCount, const QString& lType, int lLen) {
    if (!m_db.isOpen()) {
        qDebug() << "Cannot add entry: Database is not open!";
        return false;
    }
    QSqlQuery query(m_db);
    query.prepare("INSERT INTO embeddings (source_file, text_chunk, vector_blob, doc_id, page_num, chunk_idx, model_sig, model_dim, heading_path, heading_level, chunk_type, sentence_count, list_type, list_length) "
                  "VALUES (:source, :text, :blob, :docid, :page, :index, :sig, :dim, :path, :level, :type, :scount, :ltype, :llen)");
    
    query.bindValue(":source", sourceFile);
    query.bindValue(":text", text);
    query.bindValue(":blob", vectorToBlob(embedding));
    query.bindValue(":docid", docId);
    query.bindValue(":page", pageNum);
    query.bindValue(":index", chunkIdx);
    query.bindValue(":sig", modelSig);
    query.bindValue(":dim", embedding.size());
    query.bindValue(":path", path);
    query.bindValue(":level", level);
    query.bindValue(":type", chunkType);
    query.bindValue(":scount", sCount);
    query.bindValue(":ltype", lType);
    query.bindValue(":llen", lLen);

    if (!query.exec()) {
        qDebug() << "Insert failed:" << query.lastError().text();
        return false;
    }
    
    // Update Registered Dimension if this is the first entry
    if (getRegisteredDimension() == 0) {
        setRegisteredDimension(embedding.size());
    }

    qlonglong lastId = query.lastInsertId().toLongLong();
    QSqlQuery ftsQuery(m_db);
    ftsQuery.prepare("INSERT INTO embeddings_fts(rowid, text_chunk) VALUES (:id, :text)");
    
    QString headingTokens = path;
    headingTokens.replace(QRegularExpression("[^a-zA-Z0-9\\s]"), " ");
    QString indexedText = QString("[CONTEXT: %1] %2").arg(headingTokens).arg(text);
    
    ftsQuery.bindValue(":id", lastId);
    ftsQuery.bindValue(":text", indexedText);
    ftsQuery.exec();
    
    return true;
}

QVector<VectorEntry> VectorStore::search(const QVector<float>& queryEmbedding, int limit) {
    QVector<VectorEntry> semanticResults;
    QSqlQuery query("SELECT id, text_chunk, vector_blob, source_file, doc_id, page_num, model_sig FROM embeddings", m_db);

    while (query.next()) {
        VectorEntry entry;
        entry.id = query.value(0).toInt();
        entry.text = query.value(1).toString();
        entry.embedding = blobToVector(query.value(2).toByteArray());
        entry.sourceFile = query.value(3).toString();
        entry.docId = query.value(4).toString();
        entry.pageNum = query.value(5).toInt();
        entry.modelSig = query.value(6).toString();
        
        entry.score = cosineSimilarity(queryEmbedding, entry.embedding);
        semanticResults.append(entry);
    }

    std::sort(semanticResults.begin(), semanticResults.end(), [](const VectorEntry& a, const VectorEntry& b) {
        return a.score > b.score;
    });

    if (semanticResults.size() > limit) semanticResults.resize(limit);
    return semanticResults;
}

IntentType VectorStore::detectIntent(const QString& queryText, const QVector<float>& queryEmbedding) {
    QString q = queryText.toLower();
    if (q.contains(QRegularExpression("\\b(what is|define|definition of|meaning of|theorem|lemma)\\b"))) return IntentType::Definition;
    if (q.contains(QRegularExpression("\\b(how to|steps to|procedure for|process of)\\b"))) return IntentType::Procedure;
    if (q.contains(QRegularExpression("\\b(summary|overview|explain chapter|summarize)\\b"))) return IntentType::Summary;
    if (q.contains(QRegularExpression("\\b(example|illustration|case study|walkthrough)\\b"))) return IntentType::Example;
    return IntentType::General;
}

QVector<VectorEntry> VectorStore::hybridSearch(const QString& queryText, const QVector<float>& queryEmbedding, int limit) {
    QElapsedTimer timer;
    timer.start();

    IntentType intent = detectIntent(queryText, queryEmbedding);
    QVector<VectorEntry> semanticRes = search(queryEmbedding, limit * 3); 
    qint64 tSearch = timer.elapsed();

    QVector<VectorEntry> keywordRes;
    QSqlQuery q(m_db);
    q.prepare("SELECT rowid, text_chunk, source_file, page_num, heading_path, heading_level, chunk_type, doc_id, sentence_count, list_type, list_length FROM embeddings "
              "WHERE id IN (SELECT rowid FROM embeddings_fts WHERE embeddings_fts MATCH :query) LIMIT :limit");
    q.bindValue(":query", queryText);
    q.bindValue(":limit", limit * 3);
    q.exec();
    
    while(q.next()) {
        VectorEntry e;
        e.id = q.value(0).toInt();
        e.text = q.value(1).toString();
        e.sourceFile = q.value(2).toString();
        e.pageNum = q.value(3).toInt();
        e.headingPath = q.value(4).toString();
        e.headingLevel = q.value(5).toInt();
        e.chunkType = q.value(6).toString();
        e.docId = q.value(7).toString();
        e.sentenceCount = q.value(8).toInt();
        e.listType = q.value(9).toString();
        e.listLength = q.value(10).toInt();
        keywordRes.append(e);
    }

    QMap<int, double> rrfScores;
    QMap<int, VectorEntry> entryMap;
    QMap<int, int> semanticRanks;
    QMap<int, int> keywordRanks;
    
    const double K = 60.0;
    const double weightSemantic = 0.5;
    const double weightKeyword = 0.5;

    for (int i = 0; i < semanticRes.size(); ++i) {
        int id = semanticRes[i].id;
        entryMap[id] = semanticRes[i];
        semanticRanks[id] = i + 1;
        rrfScores[id] = weightSemantic * (1.0 / (K + i + 1));
        
        double intentBoost = 0.0;
        const auto& entry = semanticRes[i];
        if (intent == IntentType::Definition && entry.chunkType == "definition") intentBoost = 0.5;
        else if (intent == IntentType::Summary && entry.chunkType == "summary") intentBoost = 0.5;
        else if (intent == IntentType::Procedure && entry.chunkType == "list") intentBoost = 0.3;
        else if (intent == IntentType::Example && entry.chunkType == "example") intentBoost = 0.4;
        
        if (intent == IntentType::Summary && entry.headingLevel == 1) intentBoost += 0.2;
        if (intent == IntentType::Definition && entry.headingLevel > 1) intentBoost += 0.1;
        rrfScores[id] += intentBoost;
    }

    for (int i = 0; i < keywordRes.size(); ++i) {
        int id = keywordRes[i].id;
        keywordRanks[id] = i + 1;
        if (!entryMap.contains(id)) entryMap[id] = keywordRes[i];
        
        rrfScores[id] += weightKeyword * (1.0 / (K + i + 1));
        double intentBoost = 0.0;
        const auto& entry = entryMap[id];
        if (intent == IntentType::Definition && entry.chunkType == "definition") intentBoost = 0.3;
        else if (intent == IntentType::Summary && entry.chunkType == "summary") intentBoost = 0.3;
        rrfScores[id] += intentBoost;
    }

    QVector<VectorEntry> finalResults;
    for (auto it = rrfScores.begin(); it != rrfScores.end(); ++it) {
        int id = it.key();
        VectorEntry e = entryMap[id];
        e.score = it.value();
        e.semanticRank = semanticRanks.value(id, 0);
        e.keywordRank = keywordRanks.value(id, 0);
        finalResults.append(e);
    }

    std::sort(finalResults.begin(), finalResults.end(), [](const VectorEntry& a, const VectorEntry& b) {
        return a.score > b.score;
    });

    if (finalResults.size() > limit) finalResults.resize(limit);
    
    qint64 tFusion = timer.elapsed() - tSearch;
    logRetrieval(queryText, 0, 0, finalResults.size(), 0, tSearch, tFusion, 0, finalResults.isEmpty() ? 0 : finalResults[0].score);
    return finalResults;
}

void VectorStore::logRetrieval(const QString& query, int sRank, int kRank, int fRank, 
                               int lEmbed, int lSearch, int lFusion, int lRerank, double topScore) {
    QSqlQuery q(m_db);
    q.prepare("INSERT INTO retrieval_logs (query, semantic_rank, keyword_rank, final_rank, "
              "latency_embedding, latency_search, latency_fusion, latency_rerank, top_score) "
              "VALUES (:query, :sr, :kr, :fr, :le, :ls, :lf, :lr, :score)");
    q.bindValue(":query", query);
    q.bindValue(":sr", sRank);
    q.bindValue(":kr", kRank);
    q.bindValue(":fr", fRank);
    q.bindValue(":le", lEmbed);
    q.bindValue(":ls", lSearch);
    q.bindValue(":lf", lFusion);
    q.bindValue(":lr", lRerank);
    q.bindValue(":score", topScore);
    q.exec();
}

int VectorStore::count() {
    QSqlQuery query("SELECT COUNT(*) FROM embeddings", m_db);
    if (query.next()) return query.value(0).toInt();
    return 0;
}

void VectorStore::clear() {
    QSqlQuery query(m_db);
    query.exec("DELETE FROM embeddings");
    query.exec("DELETE FROM workspace_metadata WHERE key = 'embedding_dimension'");
}

void VectorStore::close() {
    if (m_db.isOpen()) m_db.close();
    QString connectionName = m_db.connectionName();
    m_db = QSqlDatabase(); 
    QSqlDatabase::removeDatabase(connectionName);
}

void VectorStore::setPath(const QString& name) { m_dbPath = name; }

bool VectorStore::exportToCsv(const QString& filePath) {
    QFile file(filePath);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Text)) return false;
    QTextStream out(&file);
    out << "ID,Source File,Text Chunk\n";
    QSqlQuery query("SELECT id, source_file, text_chunk FROM embeddings", m_db);
    while (query.next()) {
        int id = query.value(0).toInt();
        QString source = query.value(1).toString().replace("\"", "\"\"");
        QString text = query.value(2).toString().replace("\"", "\"\"");
        out << id << ",\"" << source << "\",\"" << text << "\"\n";
    }
    file.close();
    return true;
}

QByteArray VectorStore::vectorToBlob(const QVector<float>& vec) {
    QByteArray blob;
    blob.append(reinterpret_cast<const char*>(vec.data()), vec.size() * sizeof(float));
    return blob;
}

QVector<float> VectorStore::blobToVector(const QByteArray& blob) {
    QVector<float> vec;
    int count = blob.size() / sizeof(float);
    vec.resize(count);
    memcpy(vec.data(), blob.constData(), blob.size());
    return vec;
}

QString VectorStore::getContext(const QString& docId, int currentIdx, int offset) {
    QSqlQuery q(m_db);
    q.prepare("SELECT text_chunk FROM embeddings WHERE doc_id = :doc AND chunk_idx = :idx");
    q.bindValue(":doc", docId);
    q.bindValue(":idx", currentIdx + offset);
    if (q.exec() && q.next()) return q.value(0).toString();
    return "[No further context]";
}

double VectorStore::cosineSimilarity(const QVector<float>& v1, const QVector<float>& v2) {
    if (v1.size() != v2.size() || v1.isEmpty()) return 0.0;
    double dotProduct = 0.0;
    double norm1 = 0.0;
    double norm2 = 0.0;
    for (int i = 0; i < v1.size(); ++i) {
        dotProduct += v1[i] * v2[i];
        norm1 += v1[i] * v1[i];
        norm2 += v2[i] * v2[i];
    }
    if (norm1 == 0.0 || norm2 == 0.0) return 0.0;
    return dotProduct / (qSqrt(norm1) * qSqrt(norm2));
}

void VectorStore::setMetadata(const QString& key, const QString& value) {
    QSqlQuery q(m_db);
    q.prepare("INSERT OR REPLACE INTO workspace_metadata (key, value) VALUES (:key, :value)");
    q.bindValue(":key", key);
    q.bindValue(":value", value);
    q.exec();
}

QString VectorStore::getMetadata(const QString& key) {
    QSqlQuery q(m_db);
    q.prepare("SELECT value FROM workspace_metadata WHERE key = :key");
    q.bindValue(":key", key);
    if (q.exec() && q.next()) return q.value(0).toString();
    return "";
}

int VectorStore::getRegisteredDimension() {
    QString dim = getMetadata("embedding_dimension");
    return dim.isEmpty() ? 0 : dim.toInt();
}

void VectorStore::setRegisteredDimension(int dim) {
    setMetadata("embedding_dimension", QString::number(dim));
}
