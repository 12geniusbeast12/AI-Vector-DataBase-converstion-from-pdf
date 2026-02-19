# Check for Administrator privileges
if (!([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    Write-Host "CRITICAL: This script MUST be run as Administrator!" -ForegroundColor Red
    Write-Host "Right-click your terminal and select 'Run as Administrator', then run this script again."
    exit
}

Write-Host "--- WiX Toolset Installation Helper ---" -ForegroundColor Cyan

Write-Host "Step 1: Enabling .NET Framework 3.5 (Required by WiX v3)..."
dism /online /enable-feature /featurename:NetFx3 /all /norestart

Write-Host "Step 2: Installing WiX Toolset via Winget..."
winget install --id WiXToolset.WiXToolset --accept-source-agreements --accept-package-agreements

Write-Host "----------------------------------------"
Write-Host "Installation attempt finished." -ForegroundColor Green
Write-Host "IMPORTANT: Please CLOSE and RE-OPEN your terminal for changes to take effect." -ForegroundColor Yellow
