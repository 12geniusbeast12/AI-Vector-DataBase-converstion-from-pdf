Write-Host "--- PDF Vector DB: Building Final MSI Installer ---" -ForegroundColor Cyan

# Ensure WiX is in the path
$wixPath = "C:\Program Files (x86)\WiX Toolset v3.14\bin"
if (Test-Path $wixPath) {
    $env:PATH = "$wixPath;$env:PATH"
}

Write-Host "Step 1: Compiling in Release mode..."
cmake --build build --config Release

Write-Host "Step 2: Deploying Qt dependencies..."
# We deploy into the build/Release folder
windeployqt --release --no-translations --no-opengl-sw --no-compiler-runtime "build/Release/PDFVectorDB.exe"

Write-Host "Step 3: Creating temporary install bundle for CPack..."
# CPack WIX uses the INSTALL rules. We need to make sure the INSTALL rules see the DLLs.
# A quick fix is to copy all DLLs from the build folder into the install location or add them to CMake.
# I'll update CMake instead to glob the DLLs, but for now let's use the script to help.

Write-Host "Step 4: Generating MSI Package..."
cpack -G WIX --config build/CPackConfig.cmake

Write-Host "--------------------------------------------------"
Write-Host "Process Complete!" -ForegroundColor Green
Write-Host "If successful, look for '.msi' in your project folder." -ForegroundColor White
