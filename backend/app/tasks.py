import os
import json
import logging
import hashlib
from datetime import datetime

import redis
from celery import Celery
from PIL import Image

from app.core.config import settings
from app.db.session import SessionLocal
from app.models import Photo, Face

# Face Analyzer Singleton
_analyzer = None
def get_analyzer():
    global _analyzer
    if _analyzer is None:
        from app.ai.face_recognition.analyze_faces import Analyzer
        model_dir = os.path.join(os.path.dirname(__file__), "ai", "face_recognition", "models")
        _analyzer = Analyzer(model_dir)
    return _analyzer

# Logging Setup
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Celery Setup
celery_app = Celery("worker", broker=f"redis://{settings.REDIS_HOST}:{settings.REDIS_PORT}/0")
celery_app.conf.broker_connection_retry_on_startup = True

def notify_error(message: str):
    """Pushes an error message to the Redis 'app:errors' list."""
    try:
        r = redis.Redis(host=settings.REDIS_HOST, port=settings.REDIS_PORT, db=0)
        r.rpush("app:errors", message)
        r.ltrim("app:errors", -50, -1)
    except Exception as e:
        logger.error(f"Failed to push error to Redis: {e}")

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@celery_app.task
def ingest_photos(directory: str):
    """
    Recursively scans directory and adds photos to DB.
    """
    db = SessionLocal()
    supported_formats = (".jpg", ".jpeg", ".png", ".webp")
    
    logger.info(f"Scanning directory: {directory}")
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(supported_formats):
                file_path = os.path.join(root, file)
                
                # Check if exists
                existing = db.query(Photo).filter(Photo.path == file_path).first()
                if existing:
                    continue
                
                try:
                    # Basic Metadata
                    size = os.path.getsize(file_path)
                    
                    # Extract EXIF (Roboust)
                    from app.utils.exif import get_exif_data
                    exif_data = get_exif_data(file_path)
                    
                    timestamp = exif_data.get('timestamp')
                    lat = exif_data.get('latitude')
                    lon = exif_data.get('longitude')
                    
                    logger.info(f"Metadata for {file}: Date={timestamp}, GPS=({lat}, {lon})")

                except Exception as e:
                    logger.error(f"Metadata error for {file_path}: {e}")
                    notify_error(f"Metadata error for {file}: {e}")
                    timestamp, lat, lon = None, None, None
                    pass

                try:
                    new_photo = Photo(
                        path=file_path,
                        filename=file,
                        size_bytes=size,
                        timestamp=timestamp or datetime.utcnow(),
                        latitude=lat,
                        longitude=lon
                    )
                    db.add(new_photo)
                    db.commit()
                    db.refresh(new_photo)
                    
                    # Trigger further processing
                    logger.info(f"Ingested photo: {file.lower()} (ID: {new_photo.id})")
                    # Use delay to call the task asynchronously
                    process_photo.delay(new_photo.id)
                    
                except Exception as e:
                    logger.error(f"Error processing {file_path}: {e}")
                    notify_error(f"Failed to ingest {file}: {e}")
                    db.rollback()
    
    db.close()
    return "Ingestion Complete"

@celery_app.task
def process_photo(photo_id: int):
    """
    Orchestrate AI tasks for a single photo.
    """
    db = SessionLocal()
    photo = db.query(Photo).filter(Photo.id == photo_id).first()
    if not photo:
        db.close()
        return

    # 1. Compute Hash (Duplicate Detection) - Simplified
    # In real app, do this before DB insert
    
    # 2. Face, Eye, and Glass Analysis
    try:
        analyzer = get_analyzer()
        logger.info(f"Analyzing faces/eyes/glasses for {photo.path}...")
        results = analyzer.analyze(photo.path, "")
        
        logger.info(f"Found {len(results['faces'])} faces in {photo.filename}.")
        
        for face_data in results['faces']:
            bbox = face_data['bbox']
            # bbox is [x1, y1, x2, y2]
            x1, y1, x2, y2 = bbox
            new_face = Face(
                photo_id=photo.id,
                x=int(x1), y=int(y1), 
                w=int(x2 - x1), h=int(y2 - y1),
                identity=face_data.get('identity'),
                has_glasses=1 if face_data.get('has_glasses') else 0,
                eyes_open=1 if face_data.get('eyes_open') else 0,
                detection_confidence=face_data.get('detection_confidence'),
                recognition_confidence=face_data.get('recognition_confidence')
            )
            db.add(new_face)
        db.commit()
    except Exception as e:
        logger.error(f"Error in face analysis for {photo.id}: {e}")
        notify_error(f"Face analysis failed for photo {photo.id}: {e}")
    
    # 3. Aesthetic Score
    try:
        from app.ai.scoring import scorer
        blur_score = scorer.calculate_blur(photo.path)
        logger.info(f"Blur score for {photo.filename}: {blur_score}")
        
        photo.blur_score = blur_score
        photo.aesthetic_score = scorer.predict_nima(photo.path)
        db.commit()
    except Exception as e:
        logger.error(f"Error in scoring for {photo.id}: {e}")
        notify_error(f"Scoring failed for photo {photo.id}: {e}")

    db.close()
    logger.info(f"Completed processing for Photo {photo_id}")
    return f"Processed Photo {photo_id}"
