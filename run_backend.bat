@echo off
title OnionAI - FastAPI AI Backend
echo ========================================================
echo   OnionAI - PyTorch / YOLO Quality AI Inference Server
echo ========================================================
cd /d "%~dp0"
echo Starting FastAPI inference server on port 8000...
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
pause
