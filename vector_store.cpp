#include "vector_store.h"
#include <QSqlQuery>
#include <QVariant>
#include <QDebug>
#include <QCoreApplication>
#include <QSqlError>
#include <QFile>
#include <QDir>
#include <QTextStream>
#include <QtMath>
#include <QStandardPaths>
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

    // Initialization (v0)
    if (version < 1) {
        q.exec("CREATE TABLE IF NOT EXISTS embeddings ("
               "id INTEGER PRIMARY KEY AUTOINCREMENT, "
               "source_file TEXT, "
               "text_chunk TEXT, "
               "vector_blob BLOB)");
        q.exec("PRAGMA user_version = 1");
    }

    // Migration to v2: Basic metadata
    if (version < 2) {
        q.exec("ALTER TABLE embeddings ADD COLUMN doc_id TEXT");
        q.exec("ALTER TABLE embeddings ADD COLUMN page_num INTEGER");
        q.exec("ALTER TABLE embeddings ADD COLUMN model_sig TEXT");
        q.exec("ALTER TABLE embeddings ADD COLUMN created_at DATETIME DEFAULT CURRENT_TIMESTAMP");
        q.exec("PRAGMA user_version = 2");
        qDebug() << "Migrated database to v2 (Core Metadata).";
    }

    // Migration to v5: Granular Ranks and Latencies
    if (version < 5) {
        q.exec("DROP TABLE IF EXISTS retrieval_logs"); // Safe to reset logs for diagnostic upgrade
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
        q.exec("PRAGMA user_version = 5");
        qDebug() << "Migrated database to v5 (Advanced Diagnostics).";
    }

    return true;
}

bool VectorStore::addEntry(const QString& text, const QVector<float>& embedding, 
                           const QString& sourceFile, const QString& docId, 
                           int pageNum, int chunkIdx, const QString& modelSig) {
    if (!m_db.isOpen()) {
        qDebug() << "Cannot add entry: Database is not open!";
        return false;
    }
    QSqlQuery query(m_db);
    query.prepare("INSERT INTO embeddings (source_file, text_chunk, vector_blob, doc_id, page_num, chunk_idx, model_sig, model_dim) "
                  "VALUES (:source, :text, :blob, :docid, :page, :index, :sig, :dim)");
    
    query.bindValue(":source", sourceFile);
    query.bindValue(":text", text);
    query.bindValue(":blob", vectorToBlob(embedding));
    query.bindValue(":docid", docId);
    query.bindValue(":page", pageNum);
    query.bindValue(":index", chunkIdx);
    query.bindValue(":sig", modelSig);
    query.bindValue(":dim", embedding.size());

    if (!query.exec()) {
        qDebug() << "Insert failed:" << query.lastError().text();
        return false;
    }
    
    // Index into FTS5
    qlonglong lastId = query.lastInsertId().toLongLong();
    QSqlQuery ftsQuery(m_db);
    ftsQuery.prepare("INSERT INTO embeddings_fts(rowid, text_chunk) VALUES (:id, :text)");
    ftsQuery.bindValue(":id", lastId);
    ftsQuery.bindValue(":text", text);
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

QVector<VectorEntry> VectorStore::hybridSearch(const QString& queryText, const QVector<float>& queryEmbedding, int limit) {
    QElapsedTimer timer;
    timer.start();

    // 1. Query Classification (Heuristic)
    QStringList words = queryText.split(" ", Qt::SkipEmptyParts);
    bool isShortQuery = words.size() <= 3;
    
    // 2. Semantic Search
    auto semanticRes = search(queryEmbedding, limit * 3); 
    qint64 tSearch = timer.elapsed();

    // 3. Keyword Search (FTS5)
    QVector<VectorEntry> keywordRes;
    QSqlQuery q(m_db);
    q.prepare("SELECT rowid, text_chunk, source_file, page_num FROM embeddings "
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
        keywordRes.append(e);
    }

    // 4. Reciprocal Rank Fusion (RRF) with Query-Aware Weights
    QMap<int, double> rrfScores;
    QMap<int, VectorEntry> entryMap;
    QMap<int, int> semanticRanks;
    QMap<int, int> keywordRanks;
    
    const double K = 60.0;
    const double weightSemantic = isShortQuery ? 0.4 : 1.0;
    const double weightKeyword = isShortQuery ? 1.0 : 0.6;

    for (int i = 0; i < semanticRes.size(); ++i) {
        int id = semanticRes[i].id;
        semanticRanks[id] = i + 1;
        rrfScores[id] += weightSemantic * (1.0 / (K + i + 1));
        entryMap[id] = semanticRes[i];
    }

    for (int i = 0; i < keywordRes.size(); ++i) {
        int id = keywordRes[i].id;
        keywordRanks[id] = i + 1;
        rrfScores[id] += weightKeyword * (1.0 / (K + i + 1));
        if (!entryMap.contains(id)) {
            QSqlQuery fetch(m_db);
            fetch.prepare("SELECT text_chunk, source_file, page_num, model_sig FROM embeddings WHERE id = :id");
            fetch.bindValue(":id", id);
            if (fetch.exec() && fetch.next()) {
                VectorEntry fe;
                fe.id = id;
                fe.text = fetch.value(0).toString();
                fe.sourceFile = fetch.value(1).toString();
                fe.pageNum = fetch.value(2).toInt();
                fe.modelSig = fetch.value(3).toString();
                entryMap[id] = fe;
            }
        }
    }

    QVector<VectorEntry> finalResults;
    for (auto it = rrfScores.begin(); it != rrfScores.end(); ++it) {
        int id = it.key();
        VectorEntry e = entryMap[id];
        e.score = it.value(); // RRF score
        e.semanticRank = semanticRanks.value(id, 0);
        e.keywordRank = keywordRanks.value(id, 0);
        finalResults.append(e);
    }

    std::sort(finalResults.begin(), finalResults.end(), [](const VectorEntry& a, const VectorEntry& b) {
        return a.score > b.score;
    });

    if (finalResults.size() > limit) finalResults.resize(limit);
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
    if (query.next()) {
        return query.value(0).toInt();
    }
    return 0;
}

void VectorStore::clear() {
    QSqlQuery query(m_db);
    query.exec("DELETE FROM embeddings");
}

void VectorStore::close() {
    if (m_db.isOpen()) {
        m_db.close();
    }
    // Remove the database connection to allow re-creating it if needed
    QString connectionName = m_db.connectionName();
    m_db = QSqlDatabase(); 
    QSqlDatabase::removeDatabase(connectionName);
}

void VectorStore::setPath(const QString& name) {
    m_dbPath = name;
}

bool VectorStore::exportToCsv(const QString& filePath) {
    QFile file(filePath);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Text)) {
        return false;
    }

    QTextStream out(&file);
    out << "ID,Source File,Text Chunk\n";

    QSqlQuery query("SELECT id, source_file, text_chunk FROM embeddings", m_db);
    while (query.next()) {
        int id = query.value(0).toInt();
        QString source = query.value(1).toString().replace("\"", "\"\"");
        QString text = query.value(2).toString().replace("\"", "\"\"");
        
        // Basic CSV escaping
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
