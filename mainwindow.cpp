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

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent), m_store(new VectorStore()), m_pdfProcessor(new PdfProcessor(this)), m_statusLabel(nullptr) {
    
    // ... UI Setup ...
    QWidget *centralWidget = new QWidget(this);
    setCentralWidget(centralWidget);
    QVBoxLayout *layout = new QVBoxLayout(centralWidget);

    QLabel *titleLabel = new QLabel("PDF Vector DB Converter (v3.6.0 - Advanced RAG)", this);
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
            m_statusLabel->setText(QString("Switched to workspace: %1").arg(dbName));
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
    layout->addLayout(searchLayout);

    // Results
    QTableWidget *resultsTable = new QTableWidget(0, 4, this);
    resultsTable->setHorizontalHeaderLabels({"Text Chunk", "Source File", "Page", "Relevance"});
    resultsTable->horizontalHeader()->setSectionResizeMode(0, QHeaderView::Stretch);
    resultsTable->horizontalHeader()->setSectionResizeMode(1, QHeaderView::ResizeToContents);
    resultsTable->horizontalHeader()->setSectionResizeMode(2, QHeaderView::ResizeToContents);
    layout->addWidget(resultsTable);

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
    connect(m_api, &GeminiApi::discoveredModelsReady, this, [this, providerCombo](const QVector<ModelInfo>& models) {
        QString current = providerCombo->currentText();
        providerCombo->clear();
        providerCombo->addItem("Google Gemini (Cloud)");
        m_lastDiscoveredModels = models;
        for (const auto& m : models) {
            providerCombo->addItem(QString("[%1] %2").arg(m.engine).arg(m.name));
        }
        
        if (models.isEmpty()) {
            m_statusLabel->setText("No Local AI detected. Ensure LM Studio Server is STARTED on port 1234.");
            providerCombo->addItem("--- Manual Entry (See Troubleshoot) ---");
        } else {
            m_statusLabel->setText(QString("Detected %1 local models.").arg(models.size()));
        }

        int idx = providerCombo->findText(current);
        if (idx >= 0) providerCombo->setCurrentIndex(idx);
    });

    // Connections
    connect(providerCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), [this, providerCombo](int index) {
        if (index <= 0) {
            m_api->setLocalMode(0); // Gemini
        } else {
            // Re-fetch from the current model info stored in m_lastDiscoveredModels
            if (index - 1 < m_lastDiscoveredModels.size()) {
                const auto& m = m_lastDiscoveredModels[index - 1];
                int mode = (m.engine == "Ollama") ? 1 : 2;
                m_api->setLocalMode(mode);
                m_api->setSelectedModel(m);
            }
        }
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
            progressBar->setFormat("Extracting chunks...");
            m_isIndexing = true;
            
            QVector<Chunk> chunks = m_pdfProcessor->extractChunks(fileName);
            if (!chunks.isEmpty()) {
                chunkAndProcess(chunks, progressBar);
            } else {
                m_isIndexing = false;
                QMessageBox::critical(this, "Error", "Failed to extract chunks from PDF locally.");
            }
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
    
    connect(searchBtn, &QPushButton::clicked, [this]() {
        if (!m_searchEdit->text().isEmpty()) {
            m_isIndexing = false;
            m_searchTimer.start();
            m_statusLabel->setText("Generating query embedding...");
            m_api->getEmbeddings(m_searchEdit->text());
        }
    });

    connect(m_api, &GeminiApi::embeddingsReady, this, [this, resultsTable, progressBar](const QString& text, const QVector<float>& embedding, const QMap<QString, QVariant>& metadata) {
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
            
            m_store->addEntry(text, embedding, m_currentFileName, m_currentDocId, pageNum, chunkIdx, modelSig);
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
        // Log the absolute winner with all ranks/latencies
        m_store->logRetrieval(m_searchEdit->text(), results[0].semanticRank, results[0].keywordRank, 1, 
                              m_tEmbed, m_tSearch, 0, m_tRerank, results[0].score);
    }

    resultsTable->setRowCount(0);
    for (const auto& entry : results) {
        int row = resultsTable->rowCount();
        resultsTable->insertRow(row);
        resultsTable->setItem(row, 0, new QTableWidgetItem(entry.text));
        resultsTable->setItem(row, 1, new QTableWidgetItem(entry.sourceFile));
        resultsTable->setItem(row, 2, new QTableWidgetItem(QString::number(entry.pageNum)));
        
        QString label = m_rerankCheck->isChecked() ? " (Refined)" : (m_hybridCheck->isChecked() ? " (Hybrid)" : "");
        resultsTable->setItem(row, 3, new QTableWidgetItem(QString::number(entry.score, 'f', 4) + label));
    }
}

void MainWindow::handlePdfProcessed(const QString& text) { }

void MainWindow::handleError(const QString& error) {
    QMessageBox::critical(this, "Error", error);
}

void MainWindow::processNextChunk(QProgressBar* progressBar) {
    if (m_chunkQueue.isEmpty()) {
        m_isIndexing = false;
        progressBar->setFormat("Indexing Complete!");
        m_statusLabel->setText(QString("Database Ready: %1 chunks indexed.").arg(m_store->count()));
        return;
    }

    Chunk next = m_chunkQueue.takeFirst();
    QMap<QString, QVariant> metadata;
    metadata["page"] = next.pageNum;
    metadata["index"] = m_processedChunks;
    
    m_api->getEmbeddings(next.text, metadata);
}

void MainWindow::chunkAndProcess(const QVector<Chunk>& chunks, QProgressBar* progressBar) {
    m_chunkQueue = chunks;
    m_totalChunks = m_chunkQueue.size();
    m_processedChunks = 0;
    
    if (m_totalChunks == 0) {
        QMessageBox::warning(this, "No Text", "Detailed extraction found no valid text blocks.");
        return;
    }

    progressBar->setRange(0, m_totalChunks);
    progressBar->setValue(0);
    progressBar->setFormat("Indexing Chunks: %v / %m");

    processNextChunk(progressBar);
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
