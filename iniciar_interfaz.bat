@echo off
REM Abre la interfaz grafica (http://127.0.0.1:8080)
cd /d "%~dp0"
venv\Scripts\python.exe scripts\33_interfaz.py %*
pause
