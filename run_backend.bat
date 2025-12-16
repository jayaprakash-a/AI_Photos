@echo off
cd backend
call venv\Scripts\activate
uvicorn app.main:app --reload --port 8001
pause
