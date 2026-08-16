@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"

if "%~1"=="" (
    powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup_biohub_kaggle.ps1" -TopCount 10 -SkipCompetitionData -SkipSupportPack
) else (
    powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup_biohub_kaggle.ps1" %*
)

if errorlevel 1 (
    echo.
    echo Setup failed. See the error and manifests\setup.log above.
    pause
    exit /b 1
)

echo.
echo Setup completed.
pause
