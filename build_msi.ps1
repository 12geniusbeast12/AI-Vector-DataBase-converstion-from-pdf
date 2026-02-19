Write-Host "--- PDF Vector DB: Building Final MSI Installer ---" -ForegroundColor Cyan

# Ensure WiX is in the path
$wixPath = "C:\Program Files (x86)\WiX Toolset v3.14\bin"
if (Test-Path $wixPath) {
    $env:PATH = "$wixPath;$env:PATH"
}

Write-Host "Step 1: Compiling in Release mode..."
cmake --build build --config Release

Write-Host "Step 2: Deploying Qt dependencies (windeployqt)..."
windeployqt --release --no-translations --no-opengl-sw --no-compiler-runtime "build/Release/PDFVectorDB.exe"

Write-Host "Step 3: Generating MSI Package..."
cpack -G WIX --config build/CPackConfig.cmake

Write-Host "--------------------------------------------------"
Write-Host "Process Complete!" -ForegroundColor Green
Write-Host "If successful, look for '.msi' in your project folder." -ForegroundColor White
