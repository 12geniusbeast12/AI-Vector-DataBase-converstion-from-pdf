#include "mainwindow.h"
#include "pdf_processor.h"
#include "gemini_api.h"
#include "vector_store.h"
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QCoreApplication>
#include <QPushButton>
#include <QLineEdit>
#include <QComboBox>
#include <QTextEdit>
#include <QFileDialog>
#include <QNetworkReply>
#include <QRegularExpression>
#include <QSqlQuery>
#include <QProgressBar>
#include <QLabel>
#include <QMessageBox>
#include <QFileInfo>
#include <QDir>
#include <QHeaderView>
#include <QTableWidget>
#include <QDesktopServices>
#include <QInputDialog>
#include <QUrl>
#include <QDebug>
#include <QStandardPaths>
#include <QScreen>
#include <QSqlDatabase>
#include <QTimer>

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent), m_store(new VectorStore()), m_pdfProcessor(new PdfProcessor(this)), m_statusLabel(nullptr),
      m_embedCombo(new QComboBox(this)), m_reasonCombo(new QComboBox(this)), 
      m_embedHealth(new QLabel("🔴 Embedding", this)), m_reasonHealth(new QLabel("🔴 Reasoning", this)) {
    
    // ... UI Setup ...
    QWidget *centralWidget = new QWidget(this);
    setCentralWidget(centralWidget);
    QVBoxLayout *layout = new QVBoxLayout(centralWidget);

    QLabel *titleLabel = new QLabel("PDF Vector DB Converter (v3.9.0 - Structured Knowledge System)", this);
    titleLabel->setStyleSheet("font-size: 22px; font-weight: bold; color: #e67e22; margin-bottom: 5px;");
    layout->addWidget(titleLabel);

    QHBoxLayout *settingsLayout = new QHBoxLayout();
    QComboBox *providerCombo = new QComboBox(this);
    providerCombo->addItems({"Google Gemini (Cloud)", "Local (Ollama)", "Local (LM Studio)"});
    QLineEdit *apiKeyEdit = new QLineEdit(this);
    apiKeyEdit->setPlaceholderText("Enter API Key...");
    apiKeyEdit->setEchoMode(QLineEdit::Password);
    settingsLayout->addWidget(new QLabel("Provider:", this));
    settingsLayout->addWidget(providerCombo);
    
    QPushButton *refreshBtn = new QPushButton("🔄 Refresh Local AI", this);
    refreshBtn->setFixedWidth(120);
    settingsLayout->addWidget(refreshBtn);

    QPushButton *troubleBtn = new QPushButton("❓ Troubleshoot", this);
    troubleBtn->setFixedWidth(100);
    settingsLayout->addWidget(troubleBtn);

    settingsLayout->addWidget(new QLabel("API Key:", this));
    settingsLayout->addWidget(apiKeyEdit);
    layout->addLayout(settingsLayout);

    // Dual-Engine Control Row
    QHBoxLayout *engineRow = new QHBoxLayout();
    m_embedCombo->setMinimumWidth(250);
    m_reasonCombo->setMinimumWidth(250);
    
    m_embedHealth->setStyleSheet("color: #e74c3c; font-weight: bold; margin-right: 10px;");
    m_reasonHealth->setStyleSheet("color: #e74c3c; font-weight: bold;");

    engineRow->addWidget(new QLabel("⚡ Embedding Engine:", this));
    engineRow->addWidget(m_embedCombo);
    engineRow->addWidget(m_embedHealth);
    engineRow->addSpacing(20);
    engineRow->addWidget(new QLabel("🧠 Reasoning Engine:", this));
    engineRow->addWidget(m_reasonCombo);
    engineRow->addWidget(m_reasonHealth);
    engineRow->addStretch();
    layout->addLayout(engineRow);

    QHBoxLayout *dbRow = new QHBoxLayout();
    m_workspaceCombo = new QComboBox(this);
    m_workspaceCombo->setMinimumWidth(200);
    refreshWorkspaces();
    
    QPushButton *addDbBtn = new QPushButton("+ New Workspace", this);
    addDbBtn->setStyleSheet("background-color: #3498db; color: white; font-weight: bold;");
    
    dbRow->addWidget(new QLabel("Workspace:", this));
    dbRow->addWidget(m_workspaceCombo);
    dbRow->addWidget(addDbBtn);
    dbRow->addStretch();
    layout->addLayout(dbRow);

    connect(m_workspaceCombo, &QComboBox::currentTextChanged, [this](const QString& dbName) {
        if (!dbName.isEmpty()) {
            m_store->close();
            m_store->setPath(dbName);
            m_store->init();
            
            // Restore Engine Selections
            QString savedEmbed = m_store->getMetadata("embed_engine");
            QString savedReason = m_store->getMetadata("reason_engine");
            
            if (!savedEmbed.isEmpty()) m_embedCombo->setCurrentText(savedEmbed);
            if (!savedReason.isEmpty()) m_reasonCombo->setCurrentText(savedReason);
            
            m_statusLabel->setText(QString("Switched to workspace: %1 [Dim: %2]").arg(dbName).arg(m_store->getRegisteredDimension()));
        }
    });

    connect(addDbBtn, &QPushButton::clicked, [this]() {
        bool ok;
        QString name = QInputDialog::getText(this, "New Workspace", "Enter name (e.g. Finance):", QLineEdit::Normal, "", &ok);
        if (ok && !name.isEmpty()) {
            if (!name.endsWith(".sqlite")) name += ".sqlite";
            m_store->close();
            m_store->setPath(name);
            m_store->init();
            refreshWorkspaces();
            m_workspaceCombo->setCurrentText(name);
        }
    });

    QHBoxLayout *fileLayout = new QHBoxLayout();
    QPushButton *selectBtn = new QPushButton("Step 1: Select PDF to Index", this);
    selectBtn->setMinimumHeight(40);
    QLabel *fileLabel = new QLabel("No file selected", this);
    fileLayout->addWidget(selectBtn);
    fileLayout->addWidget(fileLabel);
    layout->addLayout(fileLayout);

    QHBoxLayout *actionLayout = new QHBoxLayout();
    QPushButton *clearBtn = new QPushButton("Danger Zone: Clear Local Database", this);
    clearBtn->setStyleSheet("background-color: #c0392b; color: white; font-weight: bold; padding: 5px;");
    clearBtn->setMinimumWidth(250);
    
    QPushButton *exportBtn = new QPushButton("Step 2: Save Database to CSV", this);
    exportBtn->setStyleSheet("background-color: #27ae60; color: white; font-weight: bold; padding: 5px;");
    exportBtn->setMinimumWidth(250);

    QPushButton *folderBtn = new QPushButton("📂 Open DB Folder", this);
    folderBtn->setStyleSheet("background-color: #f39c12; color: white; font-weight: bold; padding: 5px;");
    folderBtn->setMinimumWidth(180);

    actionLayout->addWidget(clearBtn);
    actionLayout->addWidget(exportBtn);
    actionLayout->addWidget(folderBtn);
    actionLayout->addStretch();
    layout->addLayout(actionLayout);

    QProgressBar *progressBar = new QProgressBar(this);
    layout->addWidget(progressBar);

    QHBoxLayout *searchLayout = new QHBoxLayout();
    m_searchEdit = new QLineEdit(this);
    m_searchEdit->setPlaceholderText("Search your local vector database...");
    
    m_hybridCheck = new QCheckBox("Hybrid Search (RRF)", this);
    m_hybridCheck->setChecked(true);
    
    m_rerankCheck = new QCheckBox("Stage 2: Rerank", this);
    m_rerankCheck->setChecked(false);
    
    QPushButton *searchBtn = new QPushButton("Search", this);
    searchLayout->addWidget(m_searchEdit);
    searchLayout->addWidget(m_hybridCheck);
    searchLayout->addWidget(m_rerankCheck);
    searchLayout->addWidget(searchBtn);
    
    m_deepDiveBtn = new QPushButton("🧪 Deep Dive Synthesis");
    m_deepDiveBtn->setStyleSheet("background-color: #6200ee; color: white; font-weight: bold; border-radius: 4px; padding: 4px 8px;");
    connect(m_deepDiveBtn, &QPushButton::clicked, this, &MainWindow::onDeepDiveRequested);
    searchLayout->addWidget(m_deepDiveBtn);

    layout->addLayout(searchLayout);

    // Results
    QTableWidget *resultsTable = new QTableWidget(0, 4, this);
    resultsTable->setHorizontalHeaderLabels({"Text Chunk", "Source File", "Page", "Relevance"});
    resultsTable->horizontalHeader()->setSectionResizeMode(0, QHeaderView::Stretch);
    resultsTable->horizontalHeader()->setSectionResizeMode(1, QHeaderView::ResizeToContents);
    resultsTable->horizontalHeader()->setSectionResizeMode(2, QHeaderView::ResizeToContents);
    layout->addWidget(resultsTable);

    connect(resultsTable, &QTableWidget::cellDoubleClicked, [this, resultsTable](int row, int col) {
        QTableWidgetItem *item = resultsTable->item(row, 0);
        if (!item) return;
        
        QString docId = item->data(Qt::UserRole).toString();
        // Since we didn't store chunk_idx directly in VectorEntry yet (it was local to addEntry),
        // let's fetch it from the database based on the entry ID (Qt::UserRole + 1)
        int entryId = item->data(Qt::UserRole + 1).toInt();
        
        QSqlQuery q(m_store->database()); // Access the db
        q.prepare("SELECT chunk_idx, text_chunk FROM embeddings WHERE id = :id");
        q.bindValue(":id", entryId);
        if (q.exec() && q.next()) {
            int idx = q.value(0).toInt();
            QString currentText = q.value(1).toString();
            
            QString prev = m_store->getContext(docId, idx, -1);
            QString next = m_store->getContext(docId, idx, 1);
            
            QMessageBox msg;
            msg.setWindowTitle("Context Peek (Situation Awareness)");
            msg.setText(QString("<b>Previous:</b><br>%1<hr><b>Current:</b><br>%2<hr><b>Next:</b><br>%3")
                        .arg(prev.left(300)).arg(currentText).arg(next.left(300)));
            msg.exec();
        }
    });

    connect(m_pdfProcessor, &PdfProcessor::progressUpdated, this, [progressBar](int page, int total) {
        if (progressBar->value() == 0) {
            progressBar->setFormat(QString("Extracting Page %1/%2").arg(page).arg(total));
        }
    });

    connect(m_pdfProcessor, &PdfProcessor::chunksReady, this, [this, progressBar](QVector<Chunk> chunks) {
        bool wasEmpty = m_chunkQueue.isEmpty() && m_totalChunks > 0 && m_processedChunks > 0;
        if (m_totalChunks == 0) wasEmpty = true;
        
        m_chunkQueue.append(chunks);
        m_totalChunks += chunks.size();
        
        for (const auto& c : chunks) {
            if (!c.headingPath.isEmpty() && c.text.length() > 5) {
                m_sectionBuffer[c.headingPath] += c.text + "\n";
            }
        }
        
        progressBar->setMaximum(m_totalChunks);
        m_statusLabel->setText(QString("Database: Indexing (%1/%2)...").arg(m_processedChunks).arg(m_totalChunks));
        
        // If we were waiting for chunks (the API is idle and the queue was empty), kick it off
        if (m_isIndexing && wasEmpty) {
            processNextChunk(progressBar);
        }
    });
    
    connect(m_pdfProcessor, &PdfProcessor::extractionFinished, this, [this, progressBar]() {
        m_extractionComplete = true;
        // If we finish extracting and the queue is already empty, we might need to finalize
        if (m_isIndexing && m_chunkQueue.isEmpty()) {
            processNextChunk(progressBar); // This will handle the completion logic
        }
    });

    m_statusLabel = new QLabel("Database: Initializing...", this);
    m_statusLabel->setStyleSheet("color: #7f8c8d; font-style: italic;");
    
    m_latencyLabel = new QLabel("", this);
    m_latencyLabel->setStyleSheet("color: #27ae60; font-weight: bold;");
    
    QHBoxLayout *footerLayout = new QHBoxLayout();
    footerLayout->addWidget(m_statusLabel);
    footerLayout->addStretch();
    footerLayout->addWidget(m_latencyLabel);
    layout->addLayout(footerLayout);

    // Backend
    m_api = new GeminiApi(apiKeyEdit->text(), this);
    if (m_store->init()) {
        m_statusLabel->setText(QString("Database Loaded: %1 chunks available.").arg(m_store->count()));
    } else {
        m_statusLabel->setText("Database Error: Failed to open vector_db.sqlite");
    }

    // Discovery logic
    connect(m_api, &GeminiApi::discoveredModelsReady, this, [this](const QVector<ModelInfo>& models) {
        m_embedCombo->clear();
        m_reasonCombo->clear();
        
        m_embedCombo->addItem("Gemini-Embedding-001 (Cloud)");
        m_reasonCombo->addItem("Gemini-1.5-Flash (Cloud)");
        
        m_lastDiscoveredModels = models;
        int embedCount = 0;
        int reasonCount = 0;

        for (const auto& m : models) {
            QString displayName = QString("[%1] %2").arg(m.engine).arg(m.name);
            if (m.capabilities.contains(ModelCapability::Embedding)) {
                m_embedCombo->addItem(displayName);
                embedCount++;
            }
            if (m.capabilities.contains(ModelCapability::Chat)) {
                m_reasonCombo->addItem(displayName);
                reasonCount++;
            }
        }
        
        // Update Health Indicators
        m_embedHealth->setText(embedCount > 0 ? "🟢 Embedding Ready" : "🟡 Embedding (Cloud Only)");
        m_embedHealth->setStyleSheet(embedCount > 0 ? "color: #27ae60; font-weight: bold;" : "color: #f39c12; font-weight: bold;");
        
        m_reasonHealth->setText(reasonCount > 0 ? "🟢 Reasoning Ready" : "🟡 Reasoning (Cloud Only)");
        m_reasonHealth->setStyleSheet(reasonCount > 0 ? "color: #27ae60; font-weight: bold;" : "color: #f39c12; font-weight: bold;");

        m_statusLabel->setText(QString("Discovery: %1 embedding, %2 reasoning models found.").arg(embedCount).arg(reasonCount));
    });

    // Dual-Engine Selection Logic
    auto updateEngines = [this]() {
        // Embedding Engine
        if (m_embedCombo->currentIndex() == 0) {
            m_api->setEmbeddingModel({"gemini-embedding-001", "Gemini", "", {ModelCapability::Embedding}});
        } else {
            // ... (existing local lookup)
            int localIdx = m_embedCombo->currentIndex() - 1;
            int currentMatch = 0;
            for (const auto& m : m_lastDiscoveredModels) {
                if (m.capabilities.contains(ModelCapability::Embedding)) {
                    if (currentMatch == localIdx) {
                        m_api->setEmbeddingModel(m);
                        break;
                    }
                    currentMatch++;
                }
            }
        }
        m_store->setMetadata("embed_engine", m_embedCombo->currentText());

        // Reasoning Engine
        if (m_reasonCombo->currentIndex() == 0) {
            m_api->setReasoningModel({"gemini-1.5-flash", "Gemini", "", {ModelCapability::Chat, ModelCapability::Summary, ModelCapability::Rerank}});
        } else {
            // ... (existing local lookup)
            int localIdx = m_reasonCombo->currentIndex() - 1;
            int currentMatch = 0;
            for (const auto& m : m_lastDiscoveredModels) {
                if (m.capabilities.contains(ModelCapability::Chat)) {
                    if (currentMatch == localIdx) {
                        m_api->setReasoningModel(m);
                        break;
                    }
                    currentMatch++;
                }
            }
        }
        m_store->setMetadata("reason_engine", m_reasonCombo->currentText());
    };

    connect(m_embedCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), updateEngines);
    connect(m_reasonCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), updateEngines);

    // Connections
    connect(providerCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), [this](int index) {
        if (index <= 0) {
            m_api->setLocalMode(0); // Gemini
        } else {
            int mode = (index == 1) ? 1 : 2; // 1 = Ollama, 2 = LM Studio
            m_api->setLocalMode(mode);
        }
        m_api->discoverModels();
    });

    connect(refreshBtn, &QPushButton::clicked, m_api, &GeminiApi::discoverModels);
    
    connect(apiKeyEdit, &QLineEdit::textChanged, m_api, &GeminiApi::setApiKey);
    
    connect(troubleBtn, &QPushButton::clicked, [this]() {
        QMessageBox::information(this, "Local AI Troubleshooting",
            "To use LM Studio with this app:\n\n"
            "1. Open LM Studio.\n"
            "2. Go to 'Local Server' tab (icon on the left sidebar).\n"
            "3. Select a model (e.g., Qwen 2.5) in the top dropdown.\n"
            "4. Click 'START SERVER' (Default Port: 1234).\n\n"
            "If it still isn't detected, make sure 'CORS' is enabled in LM Studio settings.");
    });
    
    // Initial discovery
    m_api->discoverModels();

    connect(apiKeyEdit, &QLineEdit::textChanged, [this](const QString& text) {
        m_api->setApiKey(text);
    });


    connect(selectBtn, &QPushButton::clicked, [this, fileLabel, progressBar]() {
        QString fileName = QFileDialog::getOpenFileName(this, "Open PDF", "", "PDF Files (*.pdf)");
        if (!fileName.isEmpty()) {
            m_currentFileName = QFileInfo(fileName).fileName();
            m_currentDocId = PdfProcessor::generateDocId(fileName);
            fileLabel->setText(fileName);
            progressBar->setValue(0);
            progressBar->setMaximum(1);
            progressBar->setFormat("Extracting chunks...");
            m_isIndexing = true;
            m_extractionComplete = false;
            m_totalChunks = 0;
            m_processedChunks = 0;
            m_chunkQueue.clear();
            
            m_pdfProcessor->extractChunksAsync(fileName);
        }
    });

    connect(clearBtn, &QPushButton::clicked, [this, resultsTable, progressBar]() {
        if (QMessageBox::question(this, "Clear Index", "Delete all indexed chunks from the database?") == QMessageBox::Yes) {
            m_store->clear();
            resultsTable->setRowCount(0);
            progressBar->setValue(0);
            m_statusLabel->setText("Database cleared. Indexing required.");
        }
    });

    connect(exportBtn, &QPushButton::clicked, [this]() {
        if (m_store->count() == 0) {
            QMessageBox::warning(this, "Empty Database", "There is no data to export yet.");
            return;
        }

        QString defaultPath = QCoreApplication::applicationDirPath() + "/data/export.csv";
        QString fileName = QFileDialog::getSaveFileName(this, "Save Database to CSV", defaultPath, "CSV Files (*.csv)");
        if (!fileName.isEmpty()) {
            if (m_store->exportToCsv(fileName)) {
                QMessageBox::information(this, "Export Success", "Database exported successfully to:\n" + fileName);
            } else {
                QMessageBox::critical(this, "Export Failed", "Could not write to file.");
            }
        }
    });

    connect(folderBtn, &QPushButton::clicked, [this]() {
        QString path = QStandardPaths::writableLocation(QStandardPaths::AppDataLocation);
        QDir().mkpath(path); // Ensure it exists
        QDesktopServices::openUrl(QUrl::fromLocalFile(path));
    });

    connect(m_api, &GeminiApi::errorOccurred, this, &MainWindow::handleError);
    connect(m_api, &GeminiApi::summaryReady, this, &MainWindow::handleSummaryReady);
    connect(m_api, &GeminiApi::synthesisReady, this, &MainWindow::handleSynthesisReady);
    
    connect(searchBtn, &QPushButton::clicked, [this]() {
        if (!m_searchEdit->text().isEmpty()) {
            m_isIndexing = false;
            m_searchTimer.start();
            m_statusLabel->setText("Generating query embedding...");
            m_api->getEmbeddings(m_searchEdit->text());
        }
    });

    connect(m_api, &GeminiApi::embeddingsReady, this, [this, resultsTable, progressBar](const QString& text, const QVector<float>& embedding, const QMap<QString, QVariant>& metadata) {
        int regDim = m_store->getRegisteredDimension();
        if (regDim > 0 && embedding.size() != regDim) {
            m_isIndexing = false;
            m_statusLabel->setText("Error: Dimension Guardrail Triggered.");
            QMessageBox::critical(this, "Dimension Mismatch", 
                QString("The selected Embedding Engine produces %1-dimensional vectors, "
                        "but this workspace requires %2-dimensional vectors.\n\n"
                        "Please select the correct embedding model or switch to a new workspace.")
                .arg(embedding.size()).arg(regDim));
            return;
        }

        if (!m_isIndexing) { 
            // SEARCH MODE
            m_tEmbed = m_searchTimer.elapsed();
            
            QVector<VectorEntry> results;
            if (m_hybridCheck->isChecked()) {
                results = m_store->hybridSearch(m_searchEdit->text(), embedding);
            } else {
                results = m_store->search(embedding);
            }
            
            m_tSearch = m_searchTimer.elapsed() - m_tEmbed;
            
            if (m_rerankCheck->isChecked() && !results.isEmpty()) {
                m_statusLabel->setText(QString("Hybrid found %1 potential hits. Reranking top 10...").arg(results.size()));
                m_api->rerank(m_searchEdit->text(), results.mid(0, 10));
            } else {
                // Skip reranking
                m_tRerank = 0;
                updateResultsTable(results);
            }
        } else {
            // INDEXING MODE
            int pageNum = metadata.value("page").toInt();
            int chunkIdx = metadata.value("index").toInt();
            QString modelSig = metadata.value("model_sig").toString();
            QString headingPath = metadata.value("path").toString();
            int headingLevel = metadata.value("level").toInt();
            QString chunkType = metadata.value("type").toString();
            int sCount = metadata.value("scount").toInt();
            QString lType = metadata.value("ltype").toString();
            int lLen = metadata.value("llen").toInt();
            
            m_store->addEntry(text, embedding, m_currentFileName, m_currentDocId, pageNum, chunkIdx, modelSig, headingPath, headingLevel, chunkType, sCount, lType, lLen);
            m_processedChunks++;
            progressBar->setValue(m_processedChunks);
            
            m_statusLabel->setText(QString("Database: Indexing (%1/%2)...").arg(m_processedChunks).arg(m_totalChunks));
            processNextChunk(progressBar);
        }
    });

    connect(m_api, &GeminiApi::rerankingReady, this, [this](const QVector<VectorEntry>& results) {
        m_tRerank = m_searchTimer.elapsed() - m_tEmbed - m_tSearch;
        updateResultsTable(results);
    });

    resize(900, 650);
}

MainWindow::~MainWindow() {
    delete m_store;
}

void MainWindow::updateResultsTable(const QVector<VectorEntry>& results) {
    QTableWidget *resultsTable = findChild<QTableWidget*>();
    if (!resultsTable) return;
    
    int totalLatency = m_searchTimer.elapsed();
    m_latencyLabel->setText(QString("Total: %1ms (Embed: %2ms, Search: %3ms, Rerank: %4ms)")
                            .arg(totalLatency).arg(m_tEmbed).arg(m_tSearch).arg(m_tRerank));
    
    m_statusLabel->setText(QString("Found %1 results.").arg(results.size()));
    
    if (!results.isEmpty()) {
        m_store->logRetrieval(m_searchEdit->text(), results[0].semanticRank, results[0].keywordRank, 1, 
                              m_tEmbed, m_tSearch, 0, m_tRerank, results[0].score);
    }

    resultsTable->setRowCount(0);
    for (const auto& entry : results) {
        int row = resultsTable->rowCount();
        resultsTable->insertRow(row);
        
        // Show stylized breadcrumb Heading Path
        QString displayContent = entry.text;
        QString intentLabel;
        
        if (!entry.headingPath.isEmpty()) {
            QString pathHeader = "📍 " + entry.headingPath;
            QString typeTag = entry.chunkType.toUpper();
            
            if (typeTag == "DEFINITION") {
                pathHeader += " • 📘 DEF";
                intentLabel = "Matched Definition Intent";
            } else if (typeTag == "EXAMPLE") {
                pathHeader += " • 💡 EX";
                intentLabel = "Matched Example Intent";
            } else if (typeTag == "SUMMARY") {
                pathHeader += " • 📝 SUMMARY";
                intentLabel = "Matched Overview Intent";
            } else if (typeTag == "LIST") {
                pathHeader += " • 📝 LIST (" + QString::number(entry.listLength) + ")";
                if (entry.listType == "numbered") intentLabel = "Matched Procedure Intent";
            } else {
                pathHeader += " • " + typeTag;
            }
            displayContent = pathHeader + "\n" + QString(50, '-') + "\n" + displayContent;
        }

        QTableWidgetItem *textItem = new QTableWidgetItem(displayContent);
        if (!intentLabel.isEmpty()) {
            textItem->setToolTip("<b>Adaptive Retrieval Boost:</b><br>" + intentLabel);
            // Add a subtle star or badge indicator if possible (using text prefix for now)
            if (displayContent.startsWith("📍")) {
                textItem->setText("✨ " + displayContent);
            }
        }
        textItem->setData(Qt::UserRole, entry.docId);
        textItem->setData(Qt::UserRole + 1, entry.id); // Using rowid since chunk_idx might vary
        
        resultsTable->setItem(row, 0, textItem);
        resultsTable->setItem(row, 1, new QTableWidgetItem(entry.sourceFile));
        resultsTable->setItem(row, 2, new QTableWidgetItem(QString::number(entry.pageNum)));
        
        QString label = m_rerankCheck->isChecked() ? " (Refined)" : (m_hybridCheck->isChecked() ? " (Hybrid)" : "");
        resultsTable->setItem(row, 3, new QTableWidgetItem(QString::number(entry.score, 'f', 4) + label));
    }
}

void MainWindow::chunkAndProcess(const QVector<Chunk>& chunks, QProgressBar* progressBar) {
    // Obsolete with incremental streaming - kept for backward compatibility if ever needed.
    m_chunkQueue = chunks;
    m_totalChunks = chunks.size();
    m_processedChunks = 0;
    progressBar->setMaximum(m_totalChunks);
    progressBar->setValue(0);
    processNextChunk(progressBar);
}

void MainWindow::handlePdfProcessed(const QString& text) { }

void MainWindow::handleError(const QString& error) {
    QMessageBox::critical(this, "Error", error);
}

void MainWindow::processNextChunk(QProgressBar* progressBar) {
    if (m_chunkQueue.isEmpty()) {
        if (!m_extractionComplete) {
            // Still extracting from PDF, wait for the next chunksReady signal
            return;
        }
        
        // Extraction is finished and chunks are processed. Start summaries.
        if (m_summaryQueue.isEmpty() && !m_sectionBuffer.isEmpty()) {
            m_summaryQueue = m_sectionBuffer.keys();
            m_sectionBuffer.clear();
        }
        
        if (!m_summaryQueue.isEmpty()) {
            progressBar->setFormat("Generating Section Summaries: %v / %m");
            progressBar->setRange(0, m_summaryQueue.size());
            progressBar->setValue(0);
            processNextSummary(progressBar);
            return;
        }
        m_isIndexing = false;
        progressBar->setFormat("Indexing Complete!");
        m_statusLabel->setText(QString("Database Ready: %1 chunks indexed.").arg(m_store->count()));
        return;
    }

    Chunk next = m_chunkQueue.takeFirst();
    
    // Skip extremely short chunks that are likely noise or parsing artifacts
    if (next.text.trimmed().length() <= 3) {
        m_processedChunks++;
        progressBar->setValue(m_processedChunks);
        m_statusLabel->setText(QString("Database: Indexing (%1/%2)...").arg(m_processedChunks).arg(m_totalChunks));
        QTimer::singleShot(0, this, [this, progressBar]() { processNextChunk(progressBar); });
        return;
    }

    QMap<QString, QVariant> metadata;
    metadata["page"] = next.pageNum;
    metadata["index"] = m_processedChunks;
    metadata["path"] = next.headingPath;
    metadata["level"] = next.headingLevel;
    metadata["type"] = next.chunkType;
    metadata["scount"] = next.sentenceCount;
    metadata["ltype"] = next.listType;
    metadata["llen"] = next.listLength;
    
    m_api->getEmbeddings(next.text, metadata);
}

void MainWindow::processNextSummary(QProgressBar* progressBar) {
    if (m_summaryQueue.isEmpty()) {
        m_isIndexing = false;
        progressBar->setFormat("Indexing Complete (with Summaries)!");
        m_statusLabel->setText(QString("Database Ready: %1 chunks indexed.").arg(m_store->count()));
        return;
    }

    QString path = m_summaryQueue.takeFirst();
    QString text = m_sectionBuffer[path];
    
    QMap<QString, QVariant> metadata;
    metadata["path"] = path;
    metadata["type"] = "summary";
    metadata["level"] = 1; // Summaries are top-level
    
    // Limit summary input to avoid prompt overflows (take first 5000 chars)
    m_api->generateSummary(text.left(5000), metadata);
}

void MainWindow::handleSummaryReady(const QString& summary, const QMap<QString, QVariant>& metadata) {
    if (summary.trimmed().isEmpty()) {
        QProgressBar* pb = findChild<QProgressBar*>();
        if (pb) {
            pb->setValue(pb->value() + 1);
            processNextSummary(pb);
        }
        return;
    }
    
    // Once summary is ready, get its embedding to index it
    m_api->getEmbeddings(summary, metadata);
    
    // Update progress in processNextSummary via callback or status
    QProgressBar* pb = findChild<QProgressBar*>();
    if (pb) {
        pb->setValue(pb->value() + 1);
        processNextSummary(pb);
    }
}

void MainWindow::onDeepDiveRequested() {
    QString query = m_searchEdit->text();
    if (query.isEmpty()) return;

    // 1. Logic: Collect top 5 results and their context islands
    QTableWidget *resultsTable = findChild<QTableWidget*>();
    if (!resultsTable || resultsTable->rowCount() == 0) {
        QMessageBox::information(this, "Deep Dive", "Please perform a search first to provide context for the synthesis.");
        return;
    }

    m_statusLabel->setText("Deep Dive: Synthesizing reasoning...");
    m_deepDiveBtn->setEnabled(false);

    QStringList contextIslands;
    int limit = qMin(resultsTable->rowCount(), 5);
    
    for (int i = 0; i < limit; ++i) {
        QTableWidgetItem *item = resultsTable->item(i, 0);
        if (!item) continue;
        
        QString docId = item->data(Qt::UserRole).toString();
        int chunkId = item->data(Qt::UserRole + 1).toInt();
        
        // Extended context (±2 offsets)
        QString context = m_store->getContext(docId, chunkId, 2);
        if (!contextIslands.contains(context)) {
            contextIslands.append(context);
        }
    }

    m_api->synthesizeResponse(query, contextIslands);
}

void MainWindow::handleSynthesisReady(const QString& report) {
    m_statusLabel->setText("Deep Dive: Synthesis Complete.");
    m_deepDiveBtn->setEnabled(true);

    // 2. UI: Luxury Reasoning Dialog
    QDialog *reportDlg = new QDialog(this);
    reportDlg->setWindowTitle("🧠 Synthesis Report: Deep Dive Reasoning");
    reportDlg->resize(700, 500);
    
    QVBoxLayout *layout = new QVBoxLayout(reportDlg);
    QTextEdit *reportText = new QTextEdit();
    reportText->setReadOnly(true);
    reportText->setHtml("<div style='line-height: 1.6; font-family: Inter, Segoe UI, sans-serif;'>" + report.toHtmlEscaped().replace("\n", "<br>") + "</div>");
    
    // Stylized report view
    reportText->setStyleSheet("background-color: #fcfcfc; border: 1px solid #ddd; padding: 15px; font-size: 14px;");
    
    layout->addWidget(reportText);
    
    QPushButton *closeBtn = new QPushButton("Close");
    closeBtn->setStyleSheet("padding: 8px 16px; background-color: #6200ee; color: white; border-radius: 4px;");
    connect(closeBtn, &QPushButton::clicked, reportDlg, &QDialog::accept);
    layout->addWidget(closeBtn, 0, Qt::AlignRight);
    
    reportDlg->exec();
}

void MainWindow::refreshWorkspaces() {
    m_workspaceCombo->blockSignals(true);
    m_workspaceCombo->clear();
    
    QString dataDir = QStandardPaths::writableLocation(QStandardPaths::AppDataLocation);
    QDir dir(dataDir);
    QStringList filters;
    filters << "*.sqlite";
    
    QStringList files = dir.entryList(filters, QDir::Files, QDir::Name);
    if (files.isEmpty()) {
        files << "vector_db.sqlite"; // Default
    }
    
    m_workspaceCombo->addItems(files);
    m_workspaceCombo->blockSignals(false);
}
