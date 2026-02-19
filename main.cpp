#include <QApplication>
#include "mainwindow.h"
#include "pdf_processor.h"

int main(int argc, char *argv[]) {
    QApplication a(argc, argv);
    
    PdfProcessor::initLibrary();
    
    MainWindow w;
    w.show();
    
    int result = a.exec();
    
    PdfProcessor::destroyLibrary();
    return result;
}
