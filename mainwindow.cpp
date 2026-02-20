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

    QLabel *titleLabel = new QLabel("PDF Vector DB Converter (v3.3.3 - PRO)", this);
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
    QPushButton *dbNameBtn = new QPushButton("Step 0: Set/Switch Database", this);
    dbNameBtn->setStyleSheet("background-color: #3498db; color: white; font-weight: bold;");
    QLabel *currentDbLabel = new QLabel("Current DB: vector_db.sqlite", this);
    currentDbLabel->setStyleSheet("color: #2980b9; font-weight: bold;");
    dbRow->addWidget(dbNameBtn);
    dbRow->addWidget(currentDbLabel);
    dbRow->addStretch();
    layout->addLayout(dbRow);

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
    QLineEdit *searchEdit = new QLineEdit(this);
    searchEdit->setPlaceholderText("Search your local vector database...");
    QPushButton *searchBtn = new QPushButton("Search", this);
    searchLayout->addWidget(searchEdit);
    searchLayout->addWidget(searchBtn);
    layout->addLayout(searchLayout);

    // Results
    QTableWidget *resultsTable = new QTableWidget(0, 3, this);
    resultsTable->setHorizontalHeaderLabels({"Text Chunk", "Source File", "Relevance"});
    resultsTable->horizontalHeader()->setSectionResizeMode(0, QHeaderView::Stretch);
    resultsTable->horizontalHeader()->setSectionResizeMode(1, QHeaderView::ResizeToContents);
    layout->addWidget(resultsTable);

    m_statusLabel = new QLabel("Database: Initializing...", this);
    m_statusLabel->setStyleSheet("color: #7f8c8d; font-style: italic;");
    layout->addWidget(m_statusLabel);

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

    connect(dbNameBtn, &QPushButton::clicked, [this, currentDbLabel, resultsTable]() {
        bool ok;
        QString text = QInputDialog::getText(this, "Switch Database",
                                             "Enter new database name (e.g. my_project):",
                                             QLineEdit::Normal,
                                             "", &ok);
        if (ok && !text.trimmed().isEmpty()) {
            QString name = text.trimmed();
            if (!name.endsWith(".sqlite") && !name.endsWith(".db")) {
                name += ".sqlite";
            }
            
            m_store->close();
            m_store->setPath(name);
            if (m_store->init()) {
                currentDbLabel->setText("Current DB: " + name);
                resultsTable->setRowCount(0);
                m_statusLabel->setText(QString("Switched to %1. %2 chunks available.").arg(name).arg(m_store->count()));
            }
        }
    });

    connect(selectBtn, &QPushButton::clicked, [this, fileLabel, progressBar]() {
        QString fileName = QFileDialog::getOpenFileName(this, "Open PDF", "", "PDF Files (*.pdf)");
        if (!fileName.isEmpty()) {
            m_currentFileName = QFileInfo(fileName).fileName();
            fileLabel->setText(fileName);
            progressBar->setValue(0);
            progressBar->setFormat("Extracting text...");
            m_isIndexing = true;
            QString extractedText = m_pdfProcessor->extractText(fileName);
            if (!extractedText.isEmpty()) {
                chunkAndProcess(extractedText, progressBar);
            } else {
                m_isIndexing = false;
                QMessageBox::critical(this, "Error", "Failed to extract text from PDF locally.");
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
    
    connect(searchBtn, &QPushButton::clicked, [this, searchEdit]() {
        if (!searchEdit->text().isEmpty()) {
            m_isIndexing = false;
            m_api->getEmbeddings(searchEdit->text());
        }
    });

    connect(m_api, &GeminiApi::embeddingsReady, this, [this, resultsTable, progressBar](const QString& text, const QVector<float>& embedding) {
        qDebug() << "Embeddings ready for:" << (text.length() > 20 ? text.left(20) + "..." : text) 
                 << "Indexing Mode:" << m_isIndexing;
        if (!m_isIndexing) { 
            // SEARCH MODE
            auto results = m_store->search(embedding);
            qDebug() << "Search returned" << results.size() << "results";
            
            QMessageBox::information(this, "Search Results", 
                QString("Found %1 relevant matches across %2 total database entries.")
                .arg(results.size()).arg(m_store->count()));

            resultsTable->setRowCount(0);
            for (const auto& entry : results) {
                int row = resultsTable->rowCount();
                resultsTable->insertRow(row);
                resultsTable->setItem(row, 0, new QTableWidgetItem(entry.text));
                resultsTable->setItem(row, 1, new QTableWidgetItem(entry.sourceFile));
                resultsTable->setItem(row, 2, new QTableWidgetItem(QString::number(entry.score, 'f', 4)));
            }
        } else {
            // INDEXING MODE
            m_store->addEntry(text, embedding, m_currentFileName);
            m_processedChunks++;
            progressBar->setValue(m_processedChunks);
            
            m_statusLabel->setText(QString("Database: Indexing (%1/%2)...").arg(m_processedChunks).arg(m_totalChunks));
            processNextChunk(progressBar);
        }
    });

    resize(900, 650);
}

MainWindow::~MainWindow() {
    delete m_store;
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
        QMessageBox::information(this, "Indexing Success", 
            QString("Successfully indexed %1 chunks.\nVector Database now has %2 total entries.")
            .arg(m_totalChunks).arg(m_store->count()));
        return;
    }

    QString next = m_chunkQueue.takeFirst();
    m_api->getEmbeddings(next);
}

void MainWindow::chunkAndProcess(const QString& fullText, QProgressBar* progressBar) {
    qDebug() << "Refined Chunking logic starting...";
    
    QStringList paragraphs = fullText.split("\n\n", Qt::SkipEmptyParts);
    m_chunkQueue.clear();
    
    const int MAX_CHUNK_SIZE = 800;
    
    for (const QString& para : paragraphs) {
        QString trimmed = para.trimmed();
        if (trimmed.isEmpty()) continue;
        
        if (trimmed.length() <= MAX_CHUNK_SIZE) {
            if (trimmed.length() > 20) m_chunkQueue.append(trimmed);
        } else {
            for (int i = 0; i < trimmed.length(); i += MAX_CHUNK_SIZE) {
                QString sub = trimmed.mid(i, MAX_CHUNK_SIZE);
                if (sub.length() > 20) m_chunkQueue.append(sub);
            }
        }
    }

    m_totalChunks = m_chunkQueue.size();
    m_processedChunks = 0;
    
    if (m_totalChunks == 0) {
        QMessageBox::warning(this, "No Text", "Detailed extraction found no valid text blocks.");
        return;
    }

    progressBar->setRange(0, m_totalChunks);
    progressBar->setValue(0);
    progressBar->setFormat("Indexing Chunks: %v / %m");

    // Start the sequential processing chain
    processNextChunk(progressBar);
}
