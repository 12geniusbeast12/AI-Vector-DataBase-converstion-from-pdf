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
#include <QPushButton>
#include <QTextEdit>
#include <QDialog>
#include <QVBoxLayout>

class MainWindow : public QMainWindow {
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private slots:
    void handlePdfProcessed(const QString& text);
    void handleSummaryReady(const QString& summary, const QMap<QString, QVariant>& metadata);
    void handleSynthesisReady(const QVector<ClaimNode>& claims, const QVector<SourceContext>& contexts, const QMap<QString, QVariant>& metadata);
    void handleError(const QString& error);
    void processNextChunk(QProgressBar* progressBar);
    void processNextSummary(QProgressBar* progressBar);
    void onDeepDiveRequested();

private:
    GeminiApi *m_api;
    VectorStore *m_store;
    PdfProcessor *m_pdfProcessor;
    QLabel *m_statusLabel;
    QLabel *m_latencyLabel;
    QCheckBox *m_hybridCheck;
    QCheckBox *m_rerankCheck;
    QCheckBox *m_mmrCheck;
    QCheckBox *m_explorationCheck;
    QLineEdit *m_searchEdit;
    QPushButton *m_deepDiveBtn;
    QComboBox *m_embedCombo;
    QComboBox *m_reasonCombo;
    QComboBox *m_rerankCombo;
    QLabel *m_embedHealth;
    QLabel *m_reasonHealth;
    QLabel *m_rerankHealth;
    QCheckBox *m_showRankDiffCheck;
    QComboBox *m_workspaceCombo;
    QVector<VectorEntry> m_lastResults;
    QElapsedTimer m_searchTimer;
    
    void refreshWorkspaces();
    
    // Latency breakdown trackers
    qint64 m_tEmbed = 0;
    qint64 m_tSearch = 0;
    qint64 m_tFusion = 0;
    qint64 m_tRerank = 0;
    bool m_isIndexing = false;
    bool m_extractionComplete = false;
    QVector<Chunk> m_chunkQueue;
    int m_totalChunks = 0;
    int m_processedChunks = 0;
    
    // Phase 4 Summary State
    QMap<QString, QString> m_sectionBuffer;
    QStringList m_summaryQueue;
    int m_totalSummaries = 0;
    int m_processedSummaries = 0;

    QString m_currentFileName;
    QString m_currentDocId;
    QVector<ModelInfo> m_lastDiscoveredModels;

    void chunkAndProcess(const QVector<Chunk>& chunks, QProgressBar* progressBar);
    void updateResultsTable(const QVector<VectorEntry>& results, const QString& stage = "search");
};

#endif // MAINWINDOW_H
