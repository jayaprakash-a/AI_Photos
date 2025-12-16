from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.db.session import SessionLocal
from app.tasks import ingest_photos
from app.models import Photo
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.post("/ingest")
@router.get("/ingest")
def start_ingest(directory: str):
    """
    Trigger the ingestion background task.
    """
    if not directory:
        raise HTTPException(status_code=400, detail="Directory path required")
    
    logger.info(f"Ingestion requested for directory: {directory}")
    task = ingest_photos.delay(directory)
    return {"message": "Ingestion started", "task_id": task.id}

@router.get("/health")
def health_check():
    return {"status": "ok"}

@router.get("/photos/best")
def get_best_photos(
    limit: int = 10, 
    group_by_location: bool = False, 
    db: Session = Depends(get_db)
):
    """
    Get top photos.
    - If group_by_location is True: returns top N photos PER LOCATION cluster.
    - Otherwise: returns top N sharpest photos overall.
    """
    if group_by_location:
        from app.utils.clustering import cluster_photos_by_location
        
        # Get all photos with score
        all_photos = db.query(Photo).filter(
            Photo.blur_score.isnot(None), 
            Photo.latitude.isnot(None)
        ).all()
        
        clusters = cluster_photos_by_location(all_photos, eps_meters=2000) # 2km radius
        
        result_photos = []
        
        # For each cluster, get top N best
        for label, cluster_photos in clusters.items():
            # Sort by blur score descending
            cluster_photos.sort(key=lambda x: x.blur_score or 0, reverse=True)
            # Take top 'limit' from this cluster
            result_photos.extend(cluster_photos[:limit])
            
        # Optional: Sort the final result by timestamp or score? 
        # Let's sort by timestamp to keep them ordered by event
        result_photos.sort(key=lambda x: x.timestamp or datetime.min)
        
        return result_photos

    else:
        # Original Logic
        photos = db.query(Photo).order_by(Photo.blur_score.desc()).limit(limit).all()
        return photos

from fastapi.responses import FileResponse
import os

@router.get("/photos/{photo_id}/image")
def get_photo_image(photo_id: int, db: Session = Depends(get_db)):
    photo = db.query(Photo).filter(Photo.id == photo_id).first()
    if not photo or not os.path.exists(photo.path):
        raise HTTPException(status_code=404, detail="Photo not found")
    return FileResponse(photo.path)

# --- Event / Generic Organizer API ---
from app.models import Event
from app.ai.organizer import organize_library

@router.post("/organize")
def trigger_optimization(db: Session = Depends(get_db)):
    """
    Triggers the Smart Event Organizer.
    Groups photos into events based on time and location.
    """
    count = organize_library(db)
    return {"message": f"Organization complete. Created/Updated {count} events."}

@router.get("/events")
def get_events(db: Session = Depends(get_db)):
    """
    List all events with their cover photo.
    """
    events = db.query(Event).order_by(Event.start_time.desc()).all()
    # Pydantic models would be better here for serialization control (prevent recursion)
    # For MVP, rely on FastAPI to serialize (beware circular refs if not Pydantic)
    # We'll return a simplified list manually to avoid recursion or huge payloads
    
    result = []
    for e in events:
        result.append({
            "id": e.id,
            "name": e.name,
            "location_name": e.location_name,
            "description": e.description,
            "start_time": e.start_time,
            "cover_photo_id": e.cover_photo_id,
            "photo_count": len(e.photos)
        })
    return result

@router.get("/events/{event_id}")
def get_event_details(event_id: int, db: Session = Depends(get_db)):
    event = db.query(Event).filter(Event.id == event_id).first()
    if not event: raise HTTPException(404)
    
    # Get top 50 photos for this event, sorted by score or time
    photos = db.query(Photo).filter(Photo.event_id == event_id)\
        .order_by(Photo.blur_score.desc()).limit(50).all()
        
    return {
        "event": {
            "id": event.id,
            "name": event.name,
            "description": event.description,
            "location_name": event.location_name
        },
        "photos": photos
    }

from pydantic import BaseModel
class EventUpdate(BaseModel):
    name: str = None
    description: str = None
    location_name: str = None

@router.patch("/events/{event_id}")
def update_event(event_id: int, update: EventUpdate, db: Session = Depends(get_db)):
    event = db.query(Event).filter(Event.id == event_id).first()
    if not event: raise HTTPException(404)
    
    if update.name: event.name = update.name
    if update.description: event.description = update.description
    if update.location_name: event.location_name = update.location_name
    
    db.commit()
    
    # Notify if geocoding found
    geocoding_msg = ""
    if update.location_name:
        try:
            from app.utils.geocoding import search_place
            results = search_place(update.location_name)
            if results:
                best = results[0]
                geocoding_msg = f" (Mapped to {best['name']} at {best['lat']:.2f}, {best['lon']:.2f})"
        except:
            pass
            
    return {"status": "updated", "event": event.name, "message": f"Event updated{geocoding_msg}"}

@router.get("/places/search")
def search_places_api(q: str):
    """
    Proxy to Nominatim to find places.
    """
    from app.utils.geocoding import search_place
    return search_place(q)

# --- Notifications API ---
import redis
from app.core.config import settings

@router.get("/notifications")
def get_notifications():
    """
    Retrieve and clear recent errors from the notification queue.
    """
    try:
        r = redis.Redis(host=settings.REDIS_HOST, port=settings.REDIS_PORT, db=0)
        # Pop all items
        errors = []
        while True:
            item = r.lpop("app:errors")
            if not item:
                break
            errors.append(item.decode('utf-8'))
        return errors
        return errors
    except Exception as e:
        return []

# --- People / Face Cluster API ---
from app.models import Cluster, Face

@router.get("/people")
def get_people(db: Session = Depends(get_db)):
    """
    List all detected people (clusters).
    """
    people = db.query(Cluster).all()
    # Simple serialization
    results = []
    for p in people:
        face_count = len(p.faces)
        cover_face = p.faces[0] if p.faces else None
        
        # We need a way to serve a face crop or just the full photo
        # For MVP, just return the photo ID of the first face
        results.append({
            "id": p.id,
            "name": p.name,
            "face_count": face_count,
            "cover_photo_id": cover_face.photo_id if cover_face else None
        })
    return results

class PersonUpdate(BaseModel):
    name: str

@router.patch("/people/{cluster_id}")
def update_person(cluster_id: int, update: PersonUpdate, db: Session = Depends(get_db)):
    cluster = db.query(Cluster).filter(Cluster.id == cluster_id).first()
    if not cluster: raise HTTPException(404)
    
    cluster.name = update.name
    db.commit()
    return {"status": "updated", "person": cluster.name}
