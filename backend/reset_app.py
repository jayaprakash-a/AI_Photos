import redis
from app.core.config import settings
from app.db.session import SessionLocal
from sqlalchemy import text
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def reset_app():
    # 1. Purge Redis (Stop Ingestion/Background Tasks)
    try:
        r = redis.Redis(host=settings.REDIS_HOST, port=settings.REDIS_PORT, db=0)
        r.flushall()
        logger.info("✅ Redis queue flushed (Pending tasks removed).")
    except Exception as e:
        logger.error(f"❌ Failed to flush Redis: {e}")

    # 2. Clear Database
    db = SessionLocal()
    try:
        # Drop tables to allow schema updates (e.g. adding columns)
        logger.info("Dropping database tables...")
        db.execute(text("DROP TABLE IF EXISTS photos, faces, clusters, events CASCADE"))
        db.commit()
        logger.info("✅ Database tables truncated.")
    except Exception as e:
        # It might fail if tables don't exist, which is fine
        logger.error(f"❌ Error clearing database (tables might not exist): {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    print("WARNING: This will delete ALL photos, faces, and clusters from the database.")
    print("It will also stop any pending background tasks.")
    confirm = input("Are you sure? (y/n): ")
    if confirm.lower() == 'y':
        reset_app()
        print("Reset complete.")
    else:
        print("Cancelled.")
