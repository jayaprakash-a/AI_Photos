@echo off
cd backend
call venv\Scripts\activate
celery -A app.tasks worker --loglevel=info -P solo
pause
