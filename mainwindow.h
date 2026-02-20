#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QString>
#include <QVector>
#include <QStringList>
#include "gemini_api.h"
#include "pdf_processor.h"

class VectorStore;
class QProgressBar;
class QLabel;
class QLineEdit;
class QComboBox;

#include <QElapsedTimer>
#include <QCheckBox>

class MainWindow : public QMainWindow {
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private slots:
    void handlePdfProcessed(const QString& text);
    void handleError(const QString& error);
    void processNextChunk(QProgressBar* progressBar);

private:
    GeminiApi *m_api;
    VectorStore *m_store;
    PdfProcessor *m_pdfProcessor;
    QLabel *m_statusLabel;
    QLabel *m_latencyLabel;
    QCheckBox *m_hybridCheck;
    QCheckBox *m_rerankCheck;
    QLineEdit *m_searchEdit;
    QComboBox *m_workspaceCombo;
    QElapsedTimer m_searchTimer;
    
    void refreshWorkspaces();
    
    // Latency breakdown trackers
    qint64 m_tEmbed = 0;
    qint64 m_tSearch = 0;
    qint64 m_tFusion = 0;
    qint64 m_tRerank = 0;
    bool m_isIndexing = false;
    QVector<Chunk> m_chunkQueue;
    int m_totalChunks = 0;
    int m_processedChunks = 0;
    QString m_currentFileName;
    QString m_currentDocId;
    QVector<ModelInfo> m_lastDiscoveredModels;

    void chunkAndProcess(const QVector<Chunk>& chunks, QProgressBar* progressBar);
    void updateResultsTable(const QVector<VectorEntry>& results);
};

#endif // MAINWINDOW_H
