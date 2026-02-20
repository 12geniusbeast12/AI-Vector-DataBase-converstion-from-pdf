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
    
    // Ensure we use a dedicated 'data' folder in AppData
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

    QSqlQuery query(m_db);
    bool success = query.exec("CREATE TABLE IF NOT EXISTS embeddings ("
                              "id INTEGER PRIMARY KEY AUTOINCREMENT, "
                              "source_file TEXT, "
                              "text_chunk TEXT, "
                              "vector_blob BLOB)");
    
    if (!success) {
        qDebug() << "Error creating table:" << query.lastError().text();
    } else {
        // Auto-migration: check if source_file exists
        query.exec("PRAGMA table_info(embeddings)");
        bool hasSource = false;
        while(query.next()) {
            if (query.value(1).toString() == "source_file") hasSource = true;
        }
        if (!hasSource) {
            query.exec("ALTER TABLE embeddings ADD COLUMN source_file TEXT");
            qDebug() << "Database migrated: Added 'source_file' column.";
        }
        qDebug() << "Database table 'embeddings' verified successfully.";
    }
    return success;
}

bool VectorStore::addEntry(const QString& text, const QVector<float>& embedding, const QString& sourceFile) {
    if (!m_db.isOpen()) {
        qDebug() << "Cannot add entry: Database is not open!";
        return false;
    }
    QSqlQuery query(m_db);
    query.prepare("INSERT INTO embeddings (source_file, text_chunk, vector_blob) VALUES (:source, :text, :blob)");
    query.bindValue(":source", sourceFile);
    query.bindValue(":text", text);
    query.bindValue(":blob", vectorToBlob(embedding));

    if (!query.exec()) {
        qDebug() << "Insert failed:" << query.lastError().text();
        return false;
    }
    return true;
}

QVector<VectorEntry> VectorStore::search(const QVector<float>& queryEmbedding, int limit) {
    QVector<VectorEntry> results;
    QSqlQuery query("SELECT id, text_chunk, vector_blob, source_file FROM embeddings", m_db);

    if (query.lastError().isValid()) {
        qDebug() << "Search query error:" << query.lastError().text();
    }

    while (query.next()) {
        VectorEntry entry;
        entry.id = query.value(0).toInt();
        entry.text = query.value(1).toString();
        entry.embedding = blobToVector(query.value(2).toByteArray());
        entry.sourceFile = query.value(3).toString();
        entry.score = cosineSimilarity(queryEmbedding, entry.embedding);
        results.append(entry);
    }

    // Sort by score descending
    std::sort(results.begin(), results.end(), [](const VectorEntry& a, const VectorEntry& b) {
        return a.score > b.score;
    });

    if (results.size() > limit) {
        results.resize(limit);
    }

    return results;
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
