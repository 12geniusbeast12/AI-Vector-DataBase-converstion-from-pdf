#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QString>
#include <QVector>
#include <QStringList>
#include "gemini_api.h"

class VectorStore;
class PdfProcessor;
class QProgressBar;
class QLabel;

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
    bool m_isIndexing = false;
    QStringList m_chunkQueue;
    int m_totalChunks = 0;
    int m_processedChunks = 0;
    QString m_currentFileName;
    QVector<ModelInfo> m_lastDiscoveredModels;

    void chunkAndProcess(const QString& fullText, QProgressBar* progressBar);
};

#endif // MAINWINDOW_H
