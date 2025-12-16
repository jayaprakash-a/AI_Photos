from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.db.session import engine, Base
import app.models # Register models

# Create tables (Simplistic for MVP, usually use Alembic)
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json"
)

# Set all CORS enabled origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Root endpoint removed to allow UI serving

from app.api.routes import router as api_router
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os

app.include_router(api_router, prefix=settings.API_V1_STR)

# Mount Static Files
static_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "frontend", "ui")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")
    app.mount("/font", StaticFiles(directory=static_dir), name="font") # fallback if needed
    
    @app.get("/")
    async def read_index():
        return FileResponse(os.path.join(static_dir, "index.html"))
    
    # Also serve style.css/app.js at root if requested relatively without /static
    @app.get("/{filename}.css")
    async def read_css(filename: str):
        return FileResponse(os.path.join(static_dir, f"{filename}.css"))

    @app.get("/{filename}.js")
    async def read_js(filename: str):
        return FileResponse(os.path.join(static_dir, f"{filename}.js"))
