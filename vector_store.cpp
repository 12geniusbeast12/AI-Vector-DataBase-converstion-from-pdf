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
#include <cstdlib>

#include <QtConcurrent>
#include <QFuture>

VectorStore::VectorStore(const QString& dbPath, QObject *parent) 
    : QObject(parent), m_dbPath(dbPath) {
    m_threadPool = new QThreadPool(this);
    m_threadPool->setMaxThreadCount(qMax(2, QThread::idealThreadCount() / 2));
    m_queryCache.setMaxCost(100); // Store 100 recent query results
}

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
        q.exec("CREATE TABLE IF NOT EXISTS workspace_metadata (key TEXT PRIMARY KEY, value TEXT)");
        q.exec("PRAGMA user_version = 11");
        qDebug() << "Migrated database to v11 (Workspace Metadata).";
    }

    // Migration to v12: Interaction Feedback & Boosting
    if (version < 12) {
        q.exec("ALTER TABLE embeddings ADD COLUMN boost_factor REAL DEFAULT 1.0");
        q.exec("PRAGMA user_version = 12");
        qDebug() << "Migrated database to v12 (Interaction Boosting).";
    }
    
    // Migration to v15: Rank Stability Signals (Phase 4.4)
    if (version < 15) {
        q.exec("ALTER TABLE retrieval_logs ADD COLUMN mmr_decay REAL DEFAULT 1.0");
        q.exec("PRAGMA user_version = 15");
        qDebug() << "Migrated database to v15 (Rank Stability Signals).";
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
    QSqlQuery query("SELECT id, text_chunk, vector_blob, source_file, doc_id, page_num, model_sig, created_at, boost_factor FROM embeddings", m_db);

    while (query.next()) {
        VectorEntry entry;
        entry.id = query.value(0).toInt();
        entry.text = query.value(1).toString();
        entry.embedding = blobToVector(query.value(2).toByteArray());
        entry.sourceFile = query.value(3).toString();
        entry.docId = query.value(4).toString();
        entry.pageNum = query.value(5).toInt();
        entry.modelSig = query.value(6).toString();
        entry.createdAt = query.value(7).toDateTime();
        
        // Phase 4.2: Trust Multiplier based on recency & boost
        float boost = query.value(8).toFloat();
        qint64 secsAgo = entry.createdAt.secsTo(QDateTime::currentDateTime());
        float recencyFactor = qMax(0.5f, 1.0f - (float)secsAgo / (3600.0f * 24.0f * 30.0f)); // Decay over 30 days
        entry.trustScore = boost * recencyFactor;
        
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

QVector<VectorEntry> VectorStore::ftsSearch(const QString& queryText, int limit) {
    QVector<VectorEntry> results;
    QSqlQuery q(m_db);
    q.prepare("SELECT id, text_chunk, source_file, page_num, heading_path, heading_level, chunk_type, doc_id, sentence_count, list_type, list_length, created_at, boost_factor FROM embeddings "
              "WHERE id IN (SELECT rowid FROM embeddings_fts WHERE embeddings_fts MATCH :query) LIMIT :limit");
    q.bindValue(":query", queryText);
    q.bindValue(":limit", limit);
    
    if (q.exec()) {
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
            e.score = 0.5; // Baseline score for FTS-only
            results.append(e);
        }
    }
    return results;
}

QVector<VectorEntry> VectorStore::hybridSearch(const QString& queryText, const QVector<float>& queryEmbedding, const SearchOptions& options) {
    QElapsedTimer timer;
    timer.start();
    
    QString canonicalQuery = queryText.trimmed().toLower();
    
    // 1. Layer 1 Cache: Exact Match
    {
        QMutexLocker locker(&m_cacheMutex);
        if (m_queryCache.contains(canonicalQuery)) {
            m_cacheHits++;
            qDebug() << "Cache Hit (Exact):" << canonicalQuery;
            return *m_queryCache.object(canonicalQuery);
        }
        
        // 2. Layer 2 Cache: Semantic Similarity Match
        for (const auto& entry : m_semanticCache) {
            double sim = cosineSimilarity(queryEmbedding, entry.embedding);
            if (sim > options.semanticThreshold) {
                m_cacheHits++;
                qDebug() << "Cache Hit (Semantic Similarity" << sim << "):" << queryText;
                return entry.results;
            }
        }
    }

    // 3. Intent Detection (Metadata for calibration)
    IntentType intent = detectIntent(queryText, queryEmbedding);
    
    // Phase 3C: Intelligent Routing & Performance Monitoring
    double weightSemantic = 0.5;
    double weightKeyword = 0.5;
    int retrievalLimit = options.limit * 4;

    if (intent == IntentType::Definition || intent == IntentType::Procedure) {
        weightKeyword = 0.65;
        weightSemantic = 0.35;
        retrievalLimit = options.limit * 3; // Narrow focus
    } else if (intent == IntentType::Summary) {
        weightSemantic = 0.7;
        weightKeyword = 0.3;
        retrievalLimit = options.limit * 6; // Broad coverage
    }

    // Progressive Performance Budgeting (Degradation)
    static qint64 avgLatency = 100; // Seed with 100ms
    bool lowLatencyMode = (avgLatency > 1500); 
    bool criticalLatency = (avgLatency > 4000); // Trigger if average > 4s
    
    if (criticalLatency && intent != IntentType::Summary) {
        qDebug() << "🚨 [Intelligence] CRITICAL Latency (" << avgLatency << "ms). EMERGENCY: Bypassing Vector search.";
        QVector<VectorEntry> results = ftsSearch(queryText, options.limit);
        // Minimal RRF-like injection
        for (auto& e : results) e.score = 0.5; 
        return results;
    }

    if (lowLatencyMode) {
        qDebug() << "⏩ [Intelligence] High Latency Detected (" << avgLatency << "ms). Scaling back retrieval depth.";
        retrievalLimit = options.limit * 3;
    }

    // 4. Parallel Retrieval Pipelining
    SearchAudit audit;
    QElapsedTimer auditTimer;
    auditTimer.start();

    QFuture<QVector<VectorEntry>> ftsFuture = QtConcurrent::run(m_threadPool, [this, queryText, retrievalLimit, options]() {
        // ... (FTS thread logic remains same) ...
        QString threadConn = QString("FTS_Thread_%1").arg(reinterpret_cast<uintptr_t>(QThread::currentThreadId()));
        QSqlDatabase db;
        if (QSqlDatabase::contains(threadConn)) db = QSqlDatabase::database(threadConn);
        else { db = QSqlDatabase::cloneDatabase(m_db, threadConn); db.open(); }
        
        QVector<VectorEntry> results;
        QSqlQuery q(db);
        q.prepare("SELECT id, text_chunk, source_file, page_num, heading_path, heading_level, chunk_type, doc_id, sentence_count, list_type, list_length FROM embeddings "
                  "WHERE id IN (SELECT rowid FROM embeddings_fts WHERE embeddings_fts MATCH :query) LIMIT :limit");
        q.bindValue(":query", queryText);
        q.bindValue(":limit", retrievalLimit);
        if (q.exec()) {
            while(q.next()) {
                VectorEntry e;
                e.id = q.value(0).toInt(); e.text = q.value(1).toString();
                e.sourceFile = q.value(2).toString(); e.pageNum = q.value(3).toInt();
                e.headingPath = q.value(4).toString(); e.headingLevel = q.value(5).toInt();
                e.chunkType = q.value(6).toString(); e.docId = q.value(7).toString();
                e.sentenceCount = q.value(8).toInt();                e.listType = q.value(9).toString();
                e.listLength = q.value(10).toInt();
                e.createdAt = q.value(11).toDateTime();
                
                // Phase 4.2: Trust Multiplier
                float boost = q.value(12).toFloat();
                qint64 secsAgo = e.createdAt.secsTo(QDateTime::currentDateTime());
                float recencyFactor = qMax(0.5f, 1.0f - (float)secsAgo / (3600.0f * 24.0f * 30.0f));
                e.trustScore = boost * recencyFactor;

                results.append(e);
            }
        }
        return results;
    });

    // Run Semantic Search
    QVector<VectorEntry> semanticRes = search(queryEmbedding, retrievalLimit);
    audit.t_vector = auditTimer.elapsed();
    
    QVector<VectorEntry> keywordRes = ftsFuture.result();
    audit.t_fts = auditTimer.elapsed() - audit.t_vector;

    qint64 tSearch = timer.elapsed();
    avgLatency = (0.8 * avgLatency) + (0.2 * tSearch);

    QMap<int, double> rrfScores;
    QMap<int, VectorEntry> entryMap;
    QMap<int, int> semanticRanks;
    QMap<int, int> keywordRanks;
    
    const double K = 60.0;
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
        rrfScores[id] += intentBoost;
    }

    for (int i = 0; i < keywordRes.size(); ++i) {
        int id = keywordRes[i].id;
        keywordRanks[id] = i + 1;
        if (!entryMap.contains(id)) entryMap[id] = keywordRes[i];
        rrfScores[id] += weightKeyword * (1.0 / (K + i + 1));
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


    // Phase 4.4: Intent-Aware Rank Stability (Final Regulation)
    // 1. Calculate Historical Query Stability
    float queryStability = 1.0f;
    QSqlQuery stabilityQ(m_db);
    stabilityQ.prepare("SELECT AVG(ABS(rank_delta)) FROM retrieval_logs WHERE query = :q AND is_exploration = 0 LIMIT 10");
    stabilityQ.bindValue(":q", queryText);
    if (stabilityQ.exec() && stabilityQ.next()) {
        float avgDelta = stabilityQ.value(0).toFloat();
        queryStability = qMax(0.0f, 1.0f - (avgDelta / 5.0f)); // 0.0 if avg jump > 5 ranks
    }
    audit.queryStabilityScore = queryStability;

    // 2. Intent-Weighted Stability Bias
    float stabilityMultiplier = 0.5f; // General
    if (intent == IntentType::Definition) stabilityMultiplier = 2.0f;
    else if (intent == IntentType::Procedure) stabilityMultiplier = 1.5f;
    else if (intent == IntentType::Summary) stabilityMultiplier = 1.0f;
    
    for (auto& res : finalResults) {
        // Calculate per-entry stability index (Simplified: 1.0 if not a new entry)
        res.stabilityIndex = queryStability; 
        // Apply bias to lock in established results for high-trust intents
        res.score += res.stabilityIndex * stabilityMultiplier * 0.1f;
    }
    
    // Final sort after stability bias
    std::sort(finalResults.begin(), finalResults.end(), [](const VectorEntry& a, const VectorEntry& b) {
        return a.score > b.score;
    });

    // Phase 4.1: Adaptive Multi-Level MMR+ (Experimental)
    float mmrPenaltyTotal = 0.0f;
    if (options.experimentalMmr && finalResults.size() > 1) {
        // 1. Calculate Lambda (Diversity weight) via Sigmoid Query Complexity
        // Complexity estimate: Query length + intent weight
        float complexity = (float)queryText.split(" ").size() / 10.0f;
        if (intent == IntentType::Summary || intent == IntentType::Procedure) complexity += 0.5f;
        float lambda = 1.0 / (1.0 + qExp(-5.0 * (complexity - 0.5))); // Sigmoid [0.1, 0.9]
        lambda = qBound(0.2f, (float)lambda, 0.8f); // Stability Clamp
        
        // 2. Document Distribution Entropy (EMA Smoothed)
        QMap<QString, int> docCounts;
        for (const auto& res : finalResults) docCounts[res.docId]++;
        
        double currentEntropy = 0.0;
        for (int count : docCounts.values()) {
            double p = (double)count / finalResults.size();
            currentEntropy -= p * qLn(p) / qLn(2.0);
        }
        
        // Session-aware EMA update
        double alpha = (m_sessionSearchCount < 10) ? 0.3 : 0.1; // Respond faster to new domains early on
        m_avgDocEntropy = (alpha * currentEntropy) + (1.0 - alpha) * m_avgDocEntropy;
        m_sessionSearchCount++;
        
        // 3. Iterative Selection (MMR Greedy)
        QVector<VectorEntry> diverseResults;
        QSet<QString> selectedDocs;
        QSet<QString> selectedPaths;
        
        // Always take Top 1
        diverseResults.append(finalResults.takeFirst());
        selectedDocs.insert(diverseResults.first().docId);
        selectedPaths.insert(diverseResults.first().headingPath);
        
        while (diverseResults.size() < options.limit && !finalResults.isEmpty()) {
            int bestIdx = -1;
            double bestMmrScore = -1e9;
            float currentPenalty = 0.0f;
            
            for (int i = 0; i < finalResults.size(); ++i) {
                const auto& candidate = finalResults[i];
                
                // Multi-Tier Diversity Penalty
                float diversityPenalty = 0.0f;
                // Tier 1: Document overlap (Doc Entropy aware)
                if (selectedDocs.contains(candidate.docId)) {
                    diversityPenalty += 0.15f * (1.1 - m_avgDocEntropy); 
                }
                // Tier 2: Sectional overlap
                if (selectedPaths.contains(candidate.headingPath)) {
                    diversityPenalty += 0.1f;
                }
                
                double mmrScore = lambda * candidate.score - (1.0f - lambda) * diversityPenalty;
                if (mmrScore > bestMmrScore) {
                    bestMmrScore = mmrScore;
                    bestIdx = i;
                    currentPenalty = diversityPenalty;
                }
            }
            
            if (bestIdx != -1) {
                VectorEntry selected = finalResults.takeAt(bestIdx);
                mmrPenaltyTotal += currentPenalty;
                diverseResults.append(selected);
                selectedDocs.insert(selected.docId);
                selectedPaths.insert(selected.headingPath);
            } else break;
        }
        finalResults = diverseResults;
    }

    // Phase 4.3: Budgeted Uncertainty Exploration (Experimental)
    bool explorationInjected = false;
    // Phase 4.4: Auto-Suppression if system is unstable
    bool stabilityGate = (queryStability >= 0.6f); 

    if (options.enableExploration && stabilityGate && !finalResults.isEmpty() && 
        intent != IntentType::Definition && intent != IntentType::Procedure) {
        
        // Find a "Cold Pool" candidate: boost_factor = 1.0 (no clicks), semantic similarity [0.65, 0.85]
        // For simplicity, we scan current semantic results for a high-uncertainty candidate
        // that hasn't made it to the Top 5 yet.
        for (int i = options.limit; i < semanticRes.size(); ++i) {
            VectorEntry& candidate = semanticRes[i];
            if (candidate.trustScore <= 1.0f && candidate.score > 0.65) {
                candidate.isExploration = true;
                candidate.score = finalResults.first().score * 0.95; // Inject just below Rank 1
                finalResults.insert(1, candidate); // Active acquisition at Position 2
                explorationInjected = true;
                break;
            }
        }
    }

    if (finalResults.size() > options.limit) finalResults.resize(options.limit);
    audit.t_mmr = auditTimer.elapsed() - audit.t_fts - audit.t_vector;
    
    // Cache the result
    {
        QMutexLocker locker(&m_cacheMutex);
        m_queryCache.insert(canonicalQuery, new QVector<VectorEntry>(finalResults));
        m_cacheMisses++;
    }
    
    qint64 tTotal = timer.elapsed();
    if (!finalResults.isEmpty()) {
        int rankDelta = finalResults[0].semanticRank - 1; // Simplified delta against semantic top
        logRetrieval(queryText, finalResults[0].semanticRank, finalResults[0].keywordRank, 1, 
                     0, tSearch, audit.t_mmr, 0, finalResults[0].score, 
                     mmrPenaltyTotal, explorationInjected, rankDelta, audit.queryStabilityScore); 
    }
    return finalResults;
}

void VectorStore::setGlobalSeed(int seed) {
    m_benchSeed = seed;
    std::srand(seed); // Replacement for qsrand
}

void VectorStore::logRetrieval(const QString& query, int sRank, int kRank, int fRank, 
                              int lEmbed, int lSearch, int lFusion, int lRerank, double topScore,
                              float mmrPenalty, bool isExploration, int rankDelta, float stability) {
    if (!m_db.isOpen()) return;
    QSqlQuery q(m_db);
    q.prepare("INSERT INTO retrieval_logs (query, semantic_rank, keyword_rank, final_rank, "
              "latency_embed, latency_search, latency_fusion, latency_rerank, top_score, "
              "mmr_penalty, is_exploration, rank_delta, mmr_decay) " // Reusing mmr_decay slot for stability or adding column
              "VALUES (:query, :sr, :kr, :fr, :le, :ls, :lf, :lr, :score, :mmr, :exp, :rd, :stab)");
    
    // Actually, let's check if we should add a column or use an existing one.
    // For now, I'll assume we might want a new column in v15.
    
    q.bindValue(":query", query);
    q.bindValue(":sr", sRank);
    q.bindValue(":kr", kRank);
    q.bindValue(":fr", fRank);
    q.bindValue(":le", lEmbed);
    q.bindValue(":ls", lSearch);
    q.bindValue(":lf", lFusion);
    q.bindValue(":lr", lRerank);
    q.bindValue(":score", topScore);
    q.bindValue(":mmr", mmrPenalty);
    q.bindValue(":exp", isExploration ? 1 : 0);
    q.bindValue(":rd", rankDelta);
    q.bindValue(":stab", stability);
    q.exec();
}

void VectorStore::warmup() {
    // Runnable for low-priority background warmup
    class WarmupTask : public QRunnable {
        QSqlDatabase m_parentDb;
    public:
        WarmupTask(QSqlDatabase db) : m_parentDb(db) {}
        void run() override {
            QString threadConn = QString("Warmup_Thread_%1").arg(reinterpret_cast<uintptr_t>(QThread::currentThreadId()));
            QSqlDatabase db;
            if (QSqlDatabase::contains(threadConn)) {
                db = QSqlDatabase::database(threadConn);
            } else {
                db = QSqlDatabase::cloneDatabase(m_parentDb, threadConn);
                db.open();
            }
            
            QSqlQuery q("SELECT COUNT(id) FROM embeddings", db);
            q.exec();
            qDebug() << "Database warmup complete.";
        }
    };
    
    // Start at low priority to avoid blocking UI or search
    m_threadPool->start(new WarmupTask(m_db), -1); 
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
    if (!m_db.isOpen()) return "";
    QSqlQuery q(m_db);
    q.prepare("SELECT text_chunk FROM embeddings WHERE doc_id = :docid AND chunk_idx >= :start AND chunk_idx <= :end ORDER BY chunk_idx");
    q.bindValue(":docid", docId);
    q.bindValue(":start", currentIdx - offset);
    q.bindValue(":end", currentIdx + offset);
    
    QString result;
    if (q.exec()) {
        while (q.next()) {
            result += q.value(0).toString() + " \n";
        }
    }
    return result.trimmed();
}

#include <QDateTime>

SourceContext VectorStore::getSourceContext(VectorEntry entry, int offset, const QString& stage) {
    SourceContext ctx;
    
    // Fallback if chunk_idx wasn't retrieved in the VectorEntry directly
    // Let's explicitly query the chunk_idx for the given rowid (entry.id)
    int chunkIdx = entry.id; // Assume rowid by default
    QSqlQuery q(m_db);
    q.prepare("SELECT chunk_idx, created_at, boost_factor FROM embeddings WHERE id = :id");
    q.bindValue(":id", entry.id);
    if (q.exec() && q.next()) {
        chunkIdx = q.value(0).toInt();
        QDateTime created = q.value(1).toDateTime();
        float boost = q.value(2).toFloat();
        
        qint64 secsAgo = created.secsTo(QDateTime::currentDateTime());
        float recencyFactor = qMax(0.5f, 1.0f - (float)secsAgo / (3600.0f * 24.0f * 30.0f));
        entry.trustScore = boost * recencyFactor;
        entry.createdAt = created;
    }

    ctx.chunkId = QString("%1_%2").arg(entry.docId).arg(chunkIdx);
    ctx.docName = entry.sourceFile;
    ctx.headingPath = entry.headingPath;
    ctx.pageNumber = entry.pageNum;
    ctx.semanticScore = 0.0f; 
    ctx.embedding = entry.embedding;
    ctx.finalScore = entry.score;
    ctx.finalRank = entry.semanticRank; 
    ctx.trustScore = entry.trustScore;
    qint64 daysAgo = entry.createdAt.daysTo(QDateTime::currentDateTime());
    ctx.trustReason = QString("Recency: %1 days old (Score: %2)").arg(daysAgo).arg(entry.trustScore, 0, 'f', 2);
    
    // Phase 3C: Dynamic Context Packing
    // Increase window for final synthesis to provide more "situation awareness"
    int windowOffset = (stage == "synthesis" || stage == "refined") ? qMax(offset, 3) : offset;
    ctx.chunkText = getContext(entry.docId, chunkIdx, windowOffset);
    
    ctx.retrievalMethod = "hybrid";
    ctx.retrievalStage = stage;
    ctx.retrievalTime = QDateTime::currentSecsSinceEpoch();
    
    return ctx;
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

void VectorStore::boostEntry(int entryId, float amount) {
    QSqlQuery q(m_db);
    q.prepare("UPDATE embeddings SET boost_factor = boost_factor + :amount WHERE id = :id");
    q.bindValue(":amount", amount);
    q.bindValue(":id", entryId);
    q.exec();
}

void VectorStore::addInteraction(int entryId, const QString& query, bool isExploration) {
    QSqlQuery q(m_db);
    q.prepare("INSERT INTO retrieval_logs (query, final_rank, top_score, is_exploration) VALUES (:query, :id, 1.0, :is_exp)");
    q.bindValue(":query", "USER_CLICK: " + query);
    q.bindValue(":id", entryId);
    q.bindValue(":is_exp", isExploration ? 1 : 0);
    q.exec();

    // Phase 4.3: Exploration Quarantine
    if (!isExploration) {
        boostEntry(entryId, 0.1); 
    } else {
        qDebug() << "Exploration Quarantine: Click logged for probe" << entryId << ", but ranking boost bypassed.";
    }
}
