#include <QApplication>
#include <QStandardPaths>
#include <QDir>
#include "mainwindow.h"
#include "pdf_processor.h"

int main(int argc, char *argv[]) {
    QApplication a(argc, argv);
    
    QCoreApplication::setOrganizationName("Dev");
    QCoreApplication::setApplicationName("PDFVectorDB");
    
    PdfProcessor::initLibrary();
    
    MainWindow w;
    w.show();
    
    int result = a.exec();
    
    PdfProcessor::destroyLibrary();
    return result;
}
