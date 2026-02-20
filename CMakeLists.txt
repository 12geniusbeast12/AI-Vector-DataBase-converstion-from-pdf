cmake_minimum_required(VERSION 3.16)
project(PDFVectorDB VERSION 3.3.3 LANGUAGES CXX)

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

# This glob will pick up all DLLs deployed by windeployqt in the build directory
# Note: This requires the files to exist at configure time or re-run cmake.
# Better: Install the entire directory contents if they exist.
install(DIRECTORY "${CMAKE_BINARY_DIR}/Release/" 
        DESTINATION bin
        FILES_MATCHING PATTERN "*.dll")
install(DIRECTORY "${CMAKE_BINARY_DIR}/Release/" 
        DESTINATION bin
        FILES_MATCHING PATTERN "*.json") # For Qt plugins/configs if any
install(DIRECTORY "${CMAKE_BINARY_DIR}/Release/platforms" DESTINATION bin)
install(DIRECTORY "${CMAKE_BINARY_DIR}/Release/sqldrivers" DESTINATION bin)
install(DIRECTORY "${CMAKE_BINARY_DIR}/Release/styles" DESTINATION bin)
install(DIRECTORY "${CMAKE_BINARY_DIR}/Release/iconengines" DESTINATION bin)
install(DIRECTORY "${CMAKE_BINARY_DIR}/Release/imageformats" DESTINATION bin)

# CPack configuration
set(CPACK_PACKAGE_NAME "PDFVectorDB")
set(CPACK_PACKAGE_VENDOR "Dev")
set(CPACK_PACKAGE_VERSION "3.3.3")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "PDF to Vector Database Converter")
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_SOURCE_DIR}/LICENSE.txt")
set(CPACK_GENERATOR "WIX;ZIP")

# WIX Specific (Desktop Shortcut)
set(CPACK_WIX_PRODUCT_GUID "D2F0E7A1-8B2C-4E3D-9F1A-2C3B4A5D6E7F")
set(CPACK_WIX_UPGRADE_GUID "1F6C1B37-2DD2-4886-BC1D-85262A6DD3DB")

# Create shortcuts
set(CPACK_PACKAGE_EXECUTABLES "PDFVectorDB" "PDF Vector DB")
set(CPACK_WIX_PROGRAM_MENU_FOLDER "PDF Vector DB")
set(CPACK_CREATE_DESKTOP_LINKS "PDFVectorDB")

# Include CPack
include(CPack)
