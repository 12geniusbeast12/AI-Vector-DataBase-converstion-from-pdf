cmake_minimum_required(VERSION 3.16)
project(PDFVectorDB VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

find_package(Qt6 REQUIRED COMPONENTS Widgets Network Sql Core)

qt_standard_project_setup()

# PDFium Configuration
set(PDFIUM_ROOT "${CMAKE_SOURCE_DIR}/external/pdfium")
include_directories("${PDFIUM_ROOT}/include")

add_executable(PDFVectorDB
    main.cpp
    mainwindow.cpp
    mainwindow.h
    mainwindow.ui
    gemini_api.cpp
    gemini_api.h
    vector_store.cpp
    vector_store.h
    pdf_processor.cpp
    pdf_processor.h
)

target_link_libraries(PDFVectorDB PRIVATE 
    Qt6::Widgets 
    Qt6::Network 
    Qt6::Sql 
    Qt6::Core
    "${PDFIUM_ROOT}/lib/pdfium.dll.lib"
)

# Copy PDFium DLL to build directory
add_custom_command(TARGET PDFVectorDB POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${PDFIUM_ROOT}/bin/pdfium.dll"
    $<TARGET_FILE_DIR:PDFVectorDB>
)

# Installation rules
install(TARGETS PDFVectorDB RUNTIME DESTINATION bin)
install(FILES "${PDFIUM_ROOT}/bin/pdfium.dll" DESTINATION bin)

# CPack configuration
set(CPACK_PACKAGE_NAME "PDFVectorDB")
set(CPACK_PACKAGE_VENDOR "Dev")
set(CPACK_PACKAGE_VERSION "3.2.0")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "PDF to Vector Database Converter")
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_SOURCE_DIR}/LICENSE") # Optional, but good practice
set(CPACK_GENERATOR "WIX;NSIS;ZIP")
set(CPACK_WIX_PRODUCT_GUID "D2F0E7A1-8B2C-4E3D-9F1A-2C3B4A5D6E7F") # Random GUID

# Include CPack
include(CPack)
