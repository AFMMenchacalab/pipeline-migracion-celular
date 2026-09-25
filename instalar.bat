@echo off
REM Instalador para Windows: ejecuta instalar.ps1 sin cambiar la politica de PowerShell
cd /d "%~dp0"
set PYTHONUTF8=1
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0instalar.ps1" %*
pause
