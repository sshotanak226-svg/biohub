@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"

uv run --offline python "%~dp0download_biohub_data_kagglehub.py" %*

if errorlevel 1 (
    echo.
    echo Data download failed.
    exit /b 1
)

echo.
echo Data download completed.
exit /b 0
