import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from app.core.config import settings
from app.db.session import engine, Base
import app.models
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def init_db():
    # 1. Create Database if strictly using Postgres and it doesn't exist
    # (Skip if using SQLite)
    if "postgresql" in settings.SQLALCHEMY_DATABASE_URI:
        try:
            # Connect to default 'postgres' db to create new db
            conn = psycopg2.connect(
                user=settings.POSTGRES_USER,
                password=settings.POSTGRES_PASSWORD,
                host=settings.POSTGRES_SERVER,
                port=settings.POSTGRES_PORT,
                dbname='postgres'
            )
            conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
            cur = conn.cursor()
            
            # Check if exists
            cur.execute(f"SELECT 1 FROM pg_catalog.pg_database WHERE datname = '{settings.POSTGRES_DB}'")
            exists = cur.fetchone()
            if not exists:
                logger.info(f"Creating database {settings.POSTGRES_DB}...")
                cur.execute(f"CREATE DATABASE {settings.POSTGRES_DB}")
            else:
                logger.info(f"Database {settings.POSTGRES_DB} already exists.")
            
            cur.close()
            conn.close()
        except Exception as e:
            logger.error(f"Database creation failed (might already exist or connection error): {e}")

    # 2. Create Tables
    logger.info("Creating tables...")
    Base.metadata.create_all(bind=engine)
    logger.info("Tables created successfully.")

if __name__ == "__main__":
    init_db()
