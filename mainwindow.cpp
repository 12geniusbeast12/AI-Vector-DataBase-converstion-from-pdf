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
      m_embedCombo(new QComboBox(this)), m_reasonCombo(new QComboBox(this)), m_rerankCombo(new QComboBox(this)),
      m_embedHealth(new QLabel("🔴 Embedding", this)), m_reasonHealth(new QLabel("🔴 Reasoning", this)),
      m_rerankHealth(new QLabel("🔴 Reranking", this)) {
    
    m_showRankDiffCheck = new QCheckBox("Show Rank Diff (⬆️)", this);
    m_mmrCheck = new QCheckBox("Adaptive MMR (Exp)", this);
    m_explorationCheck = new QCheckBox("Exploration (Exp)", this);
    
    // ... UI Setup ...
    QWidget *centralWidget = new QWidget(this);
    setCentralWidget(centralWidget);
    QVBoxLayout *layout = new QVBoxLayout(centralWidget);

    QLabel *titleLabel = new QLabel("PDF Vector DB Converter (v4.4.0 - Autonomous Knowledge Engine)", this);
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

    // Triple-Engine Control Rows
    QVBoxLayout *engineBlock = new QVBoxLayout();
    
    QHBoxLayout *engineRow1 = new QHBoxLayout();
    m_embedCombo->setMinimumWidth(250);
    m_reasonCombo->setMinimumWidth(250);
    m_embedHealth->setStyleSheet("color: #e74c3c; font-weight: bold; margin-right: 10px;");
    m_reasonHealth->setStyleSheet("color: #e74c3c; font-weight: bold;");

    engineRow1->addWidget(new QLabel("⚡ Embedding Engine:", this));
    engineRow1->addWidget(m_embedCombo);
    engineRow1->addWidget(m_embedHealth);
    engineRow1->addSpacing(20);
    engineRow1->addWidget(new QLabel("🧠 Reasoning Engine:", this));
    engineRow1->addWidget(m_reasonCombo);
    engineRow1->addWidget(m_reasonHealth);
    engineRow1->addStretch();
    
    QHBoxLayout *engineRow2 = new QHBoxLayout();
    m_rerankCombo->setMinimumWidth(250);
    m_rerankHealth->setStyleSheet("color: #e74c3c; font-weight: bold; margin-right: 10px;");
    m_showRankDiffCheck->setChecked(true);
    
    engineRow2->addWidget(new QLabel("🎯 Reranking Engine:", this));
    engineRow2->addWidget(m_rerankCombo);
    engineRow2->addWidget(m_rerankHealth);
    engineRow2->addSpacing(20);
    engineRow2->addWidget(m_showRankDiffCheck);
    engineRow2->addStretch();

    engineBlock->addLayout(engineRow1);
    engineBlock->addLayout(engineRow2);
    layout->addLayout(engineBlock);

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
            QString savedRerank = m_store->getMetadata("rerank_engine");
            
            if (!savedEmbed.isEmpty()) m_embedCombo->setCurrentText(savedEmbed);
            if (!savedReason.isEmpty()) m_reasonCombo->setCurrentText(savedReason);
            if (!savedRerank.isEmpty()) m_rerankCombo->setCurrentText(savedRerank);
            
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
    m_mmrCheck->setToolTip("Phase 4.1 diversity scheduling (Experimental)");
    m_mmrCheck->setStyleSheet("color: #27ae60; font-weight: bold;");

    m_explorationCheck->setToolTip("Phase 4.3 Active signal acquisition (Experimental)");
    m_explorationCheck->setStyleSheet("color: #9b59b6; font-weight: bold;");

    searchLayout->addWidget(m_searchEdit);
    searchLayout->addWidget(m_hybridCheck);
    searchLayout->addWidget(m_rerankCheck);
    searchLayout->addWidget(m_mmrCheck);
    searchLayout->addWidget(m_explorationCheck);
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
        
        // Phase 4.3: Feedback Loop - Interaction Boost (Quarantine Aware)
        QString docId = item->data(Qt::UserRole).toString();
        int entryId = item->data(Qt::UserRole + 1).toInt();
        bool isExp = item->data(Qt::UserRole + 2).toBool();
        
        m_store->addInteraction(entryId, m_searchEdit->text(), isExp);
        
        QSqlQuery q(m_store->database()); 
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
        m_rerankCombo->clear();
        
        m_embedCombo->addItem("Gemini-Embedding-001 (Cloud)");
        m_reasonCombo->addItem("Gemini-1.5-Flash (Cloud)");
        m_rerankCombo->addItem("(No Reranker - Direct Retrieval)");
        
        m_lastDiscoveredModels = models;
        int embedCount = 0;
        int reasonCount = 0;
        int rerankCount = 0;

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
            if (m.capabilities.contains(ModelCapability::Rerank)) {
                m_rerankCombo->addItem(displayName);
                rerankCount++;
            }
        }
        
        // Update Health Indicators
        m_embedHealth->setText(embedCount > 0 ? "🟢 Embedding Ready" : "🟡 Embedding (Cloud Only)");
        m_embedHealth->setStyleSheet(embedCount > 0 ? "color: #27ae60; font-weight: bold;" : "color: #f39c12; font-weight: bold;");
        
        m_reasonHealth->setText(reasonCount > 0 ? "🟢 Reasoning Ready" : "🟡 Reasoning (Cloud Only)");
        m_reasonHealth->setStyleSheet(reasonCount > 0 ? "color: #27ae60; font-weight: bold;" : "color: #f39c12; font-weight: bold;");

        m_rerankHealth->setText(rerankCount > 0 ? "🟢 Reranking Ready" : "⚪ Bypassed");
        m_rerankHealth->setStyleSheet(rerankCount > 0 ? "color: #27ae60; font-weight: bold;" : "color: #7f8c8d; font-weight: bold;");

        m_statusLabel->setText(QString("Discovery: %1 embed, %2 reason, %3 rerank models found.").arg(embedCount).arg(reasonCount).arg(rerankCount));
    });

    // Triple-Engine Selection Logic
    auto updateEngines = [this]() {
        // 1. Embedding Engine
        if (m_embedCombo->currentIndex() == 0) {
            m_api->setEmbeddingModel({"gemini-embedding-001", "Gemini", "", "", {ModelCapability::Embedding}});
        } else {
            int targetIdx = m_embedCombo->currentIndex() - 1;
            int count = 0;
            for (const auto& m : m_lastDiscoveredModels) {
                if (m.capabilities.contains(ModelCapability::Embedding)) {
                    if (count == targetIdx) { m_api->setEmbeddingModel(m); break; }
                    count++;
                }
            }
        }
        m_store->setMetadata("embed_engine", m_embedCombo->currentText());

        // 2. Reasoning Engine
        if (m_reasonCombo->currentIndex() == 0) {
            m_api->setReasoningModel({"gemini-1.5-flash", "Gemini", "", "", {ModelCapability::Chat, ModelCapability::Summary, ModelCapability::Rerank}});
        } else {
            int targetIdx = m_reasonCombo->currentIndex() - 1;
            int count = 0;
            for (const auto& m : m_lastDiscoveredModels) {
                if (m.capabilities.contains(ModelCapability::Chat)) {
                    if (count == targetIdx) { m_api->setReasoningModel(m); break; }
                    count++;
                }
            }
        }
        m_store->setMetadata("reason_engine", m_reasonCombo->currentText());

        // 3. Reranking Engine
        QString rerankModelName = m_rerankCombo->currentText();
        if (m_rerankCombo->currentIndex() == 0) {
            m_api->setRerankModel({"none", "None", "", "", {}});
        } else {
            int targetIdx = m_rerankCombo->currentIndex() - 1;
            int count = 0;
            for (const auto& m : m_lastDiscoveredModels) {
                if (m.capabilities.contains(ModelCapability::Rerank)) {
                    if (count == targetIdx) { 
                        m_api->setRerankModel(m); 
                        // Phase 3B: Load persistent stats
                        float savedMean = m_store->getMetadata(rerankModelName + "_mean").toFloat();
                        float savedStd = m_store->getMetadata(rerankModelName + "_std").toFloat();
                        if (savedStd > 0) m_api->updateRerankerStats(savedMean, savedStd);
                        break; 
                    }
                    count++;
                }
            }
        }
        m_store->setMetadata("rerank_engine", rerankModelName);
    };

    connect(m_embedCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), updateEngines);
    connect(m_reasonCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), updateEngines);
    connect(m_rerankCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), updateEngines);

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
    
    // Phase 3B: Handle Reranker Stat Persistence
    connect(m_api, &GeminiApi::rerankerStatsUpdated, this, [this](float m, float s) {
        QString modelName = m_rerankCombo->currentText();
        m_store->setMetadata(modelName + "_mean", QString::number(m));
        m_store->setMetadata(modelName + "_std", QString::number(s));
    });

    connect(m_api, &GeminiApi::anomalyDetected, this, [this](const QString& title, const QString& msg) {
        m_statusLabel->setText("⚠️ " + title + ": " + msg);
        m_statusLabel->setStyleSheet("color: #e74c3c; font-weight: bold;");
    });

    connect(searchBtn, &QPushButton::clicked, [this]() {
        QString query = m_searchEdit->text();
        if (!query.isEmpty()) {
            m_isIndexing = false;
            m_searchTimer.start();
            
            // Phase 3A: Result Streaming - Step 1: Immediate FTS
            m_statusLabel->setText("Searching keywords...");
            QVector<VectorEntry> ftsResults = m_store->ftsSearch(query, 5);
            if (!ftsResults.isEmpty()) {
                m_tSearch = m_searchTimer.elapsed();
                updateResultsTable(ftsResults, "Keyword Search");
            }
            
            m_statusLabel->setText("Generating query embedding...");
            m_api->getEmbeddings(query);
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
            SearchOptions options;
            options.limit = 5;
            options.useRerank = m_rerankCheck->isChecked();
            options.deterministic = true; // Phase 4.0 Foundation
            options.experimentalMmr = m_mmrCheck->isChecked(); // Phase 4.1 Hypothesis
            options.enableExploration = m_explorationCheck->isChecked(); // Phase 4.3 Probe
            
            if (m_hybridCheck->isChecked()) {
                results = m_store->hybridSearch(m_searchEdit->text(), embedding, options);
            } else {
                results = m_store->search(embedding, 5);
            }
            
            m_tSearch = m_searchTimer.elapsed() - m_tEmbed;
            
            if (m_rerankCheck->isChecked() && !results.isEmpty() && !m_rerankCombo->currentText().contains("No Reranker")) {
                m_statusLabel->setText(QString("Hybrid found %1 potential hits. Reranking top 10 using %2...")
                                      .arg(results.size()).arg(m_rerankCombo->currentText()));
                updateResultsTable(results, "Vector-Hybrid"); // Partial Result
                m_api->rerank(m_searchEdit->text(), results.mid(0, 10));
            } else {
                // Skip reranking
                m_tRerank = 0;
                updateResultsTable(results, "Complete");
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
        updateResultsTable(results, "Refined Rerank");
    });

    resize(900, 650);
}

MainWindow::~MainWindow() {
    delete m_store;
}

void MainWindow::updateResultsTable(const QVector<VectorEntry>& results, const QString& stage) {
    QTableWidget *resultsTable = findChild<QTableWidget*>();
    if (!resultsTable) return;
    
    int totalLatency = m_searchTimer.elapsed();
    m_latencyLabel->setText(QString("[%1] Total: %2ms (Embed: %3ms, Search: %4ms, Rerank: %5ms)")
                            .arg(stage).arg(totalLatency).arg(m_tEmbed).arg(m_tSearch).arg(m_tRerank));
    
    m_statusLabel->clear();
    float stability = results.isEmpty() ? 1.0f : results[0].stabilityIndex;
    QString stabilityTag = stability > 0.8f ? "<b style='color: #27ae60;'>🛡️ Stable</b>" : "<b style='color: #e67e22;'>⚠️ Volatile</b>";
    
    m_statusLabel->setText(QString("%1 | Found %2 results (%3).").arg(stabilityTag).arg(results.size()).arg(stage));
    m_statusLabel->setToolTip(QString("Rank Stability Index: %1\n(Regulation Active)").arg(stability, 0, 'f', 2));
    
    m_lastResults = results; // Set the cached search results for Deep Dive contextualizer
    
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
        textItem->setData(Qt::UserRole + 1, entry.id); 
        textItem->setData(Qt::UserRole + 2, entry.isExploration); // Phase 4.3 Quarantine Data
        
        resultsTable->setItem(row, 0, textItem);
        resultsTable->setItem(row, 1, new QTableWidgetItem(entry.sourceFile));
        resultsTable->setItem(row, 2, new QTableWidgetItem(QString::number(entry.pageNum)));
        
        QString rankShift;
        if (m_rerankCheck->isChecked() && m_showRankDiffCheck->isChecked() && m_tRerank > 0) {
            int original = entry.rerankRank + 1; // 1-indexed for comparison
            int current = row + 1;
            if (current < original) {
                rankShift = QString(" ⬆️ (+%1)").arg(original - current);
            } else if (current > original) {
                rankShift = QString(" ⬇️ (-%1)").arg(current - original);
            } else {
                rankShift = " ➖";
            }
        }
        
        QString label;
        if (entry.keywordRank > 0 && entry.semanticRank > 0) label = " (Hybrid)";
        else if (entry.keywordRank > 0) label = " (Keyword)";
        else if (entry.semanticRank > 0) label = " (Vector)";
        
        QString rerankTag = (m_tRerank > 0) ? " [R]" : "";
        resultsTable->setItem(row, 3, new QTableWidgetItem(QString::number(entry.score, 'f', 4) + label + rerankTag + rankShift));
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
    if (query.isEmpty() || m_lastResults.isEmpty()) return;

    m_statusLabel->setText("Deep Dive: Synthesizing grounded reasoning...");
    m_deepDiveBtn->setEnabled(false);

    QVector<SourceContext> contextIslands;
    int limit = qMin(m_lastResults.size(), 5);
    
    // Hardening: Extract full metadata contexts
    for (int i = 0; i < limit; ++i) {
        VectorEntry entry = m_lastResults[i];
        QString stage = m_rerankCheck->isChecked() ? "rerank" : (m_hybridCheck->isChecked() ? "hybrid" : "semantic");
        SourceContext ctx = m_store->getSourceContext(entry, 2, stage);
        ctx.promptIndex = i + 1; // 1-indexed for the prompt
        contextIslands.append(ctx);
    }

    m_api->synthesizeResponse(query, contextIslands);
}

#include <QSplitter>
#include <QScrollArea>

void MainWindow::handleSynthesisReady(const QVector<ClaimNode>& claims, const QVector<SourceContext>& contexts, const QMap<QString, QVariant>& metadata) {
    m_statusLabel->setText("Deep Dive: Synthesis Complete.");
    m_deepDiveBtn->setEnabled(true);

    QDialog *reportDlg = new QDialog(this);
    reportDlg->setWindowTitle("🧠 Synthesis Report: Verifiable Reasoning");
    reportDlg->resize(1000, 650);
    
    QVBoxLayout *mainLayout = new QVBoxLayout(reportDlg);
    QSplitter *splitter = new QSplitter(Qt::Horizontal);
    
    // --- LEFT PANE: Reasoning Engine output ---
    QTextEdit *reasoningEdit = new QTextEdit();
    reasoningEdit->setReadOnly(true);
    reasoningEdit->setStyleSheet("background-color: #ffffff; border: 1px solid #ddd; padding: 20px; font-size: 15px;");
    
    QString html = "<div style='line-height: 1.8; font-family: Inter, Segoe UI, sans-serif;'>";
    if (claims.isEmpty()) {
        html += "<h3 style='color: #666;'>No grounded answer found.</h3>";
        html += "<p>The retrieval engine could not find sufficient evidence within your workspace documents to answer this query with high confidence.</p>";
    } else {
        html += "<h2 style='margin-top: 0;'>Synthesized Answer</h2>";
        int claimIdx = 0;
        for (const ClaimNode& claim : claims) {
            // Confidence border coloring
            QString borderColor = "#e0e0e0"; 
            if (claim.confidence >= 0.7f) borderColor = "#4caf50";       // High
            else if (claim.confidence >= 0.4f) borderColor = "#ff9800";  // Medium
            
            html += QString("<div class='claim' style='border-left: 4px solid %1; padding-left: 12px; margin-bottom: 16px;'>")
                        .arg(borderColor);
            html += QString("<span style='color: #2c3e50;'>%1</span> ").arg(claim.statement.toHtmlEscaped());
            
            for (int srcIdx : claim.sourceIndices) {
                html += QString("<a href='source_%1' style='text-decoration: none;'><span style='background-color: #e3f2fd; color: #1976d2; padding: 2px 6px; border-radius: 4px; font-size: 12px; font-weight: bold; margin-left: 4px;'>[%1]</span></a>")
                            .arg(srcIdx);
            }
            html += "</div>";
            claimIdx++;
        }
    }
    html += "</div>";
    reasoningEdit->setHtml(html);
    
    // --- RIGHT PANE: Source Cards ---
    QWidget *sourceWidget = new QWidget();
    QVBoxLayout *sourceLayout = new QVBoxLayout(sourceWidget);
    sourceLayout->setContentsMargins(0, 0, 0, 0);
    sourceLayout->setSpacing(10);
    
    QLabel *sourceHeader = new QLabel("<b>Cited Sources</b>");
    sourceHeader->setStyleSheet("font-size: 14px; color: #555; padding: 10px;");
    sourceLayout->addWidget(sourceHeader);
    
    QScrollArea *scrollArea = new QScrollArea();
    QWidget *scrollContent = new QWidget();
    QVBoxLayout *cardsLayout = new QVBoxLayout(scrollContent);
    
    for (const SourceContext& ctx : contexts) {
        QTextEdit *card = new QTextEdit();
        card->setReadOnly(true);
        card->setFixedHeight(160);
        
        QString cardHtml = QString("<div style='font-family: sans-serif;'>");
        cardHtml += QString("<div style='font-size: 11px; color: #666; margin-bottom: 4px;'><b>[%1]</b> %2</div>")
                        .arg(ctx.promptIndex).arg(ctx.docName.toHtmlEscaped());
        cardHtml += QString("<div style='font-size: 12px; font-weight: bold; color: #4b0082; margin-bottom: 4px;'>%1</div>")
                        .arg(ctx.headingPath.toHtmlEscaped());
        
        // Phase 4.2: Trust Indicators
        QString trustColor = ctx.trustScore > 0.8 ? "#27ae60" : (ctx.trustScore > 0.5 ? "#f39c12" : "#e74c3c");
        cardHtml += QString("<div style='font-size: 10px; color: %1; margin-bottom: 8px;'>🛡️ <b>Trust: %2</b> (%3)</div>")
                        .arg(trustColor).arg(ctx.trustScore, 0, 'f', 2).arg(ctx.trustReason.toHtmlEscaped());

        cardHtml += QString("<div style='font-size: 13px; color: #333; line-height: 1.4;'>%1...</div>")
                        .arg(ctx.chunkText.left(200).toHtmlEscaped());
        cardHtml += "</div>";
        
        card->setHtml(cardHtml);
        card->setStyleSheet("background-color: #f9f9f9; border: 1px solid #ddd; border-top: 3px solid #6200ee; border-radius: 4px; padding: 8px;");
        cardsLayout->addWidget(card);
    }
    cardsLayout->addStretch();
    
    scrollArea->setWidget(scrollContent);
    scrollArea->setWidgetResizable(true);
    sourceLayout->addWidget(scrollArea);
    
    splitter->addWidget(reasoningEdit);
    splitter->addWidget(sourceWidget);
    splitter->setStretchFactor(0, 3);
    splitter->setStretchFactor(1, 2);
    
    mainLayout->addWidget(splitter);
    
    QPushButton *closeBtn = new QPushButton("Close Synthesis");
    closeBtn->setStyleSheet("padding: 10px 20px; background-color: #6200ee; color: white; border-radius: 6px; font-weight: bold;");
    connect(closeBtn, &QPushButton::clicked, reportDlg, &QDialog::accept);
    
    QHBoxLayout *btnLayout = new QHBoxLayout();
    btnLayout->addStretch();
    btnLayout->addWidget(closeBtn);
    mainLayout->addLayout(btnLayout);
    
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
