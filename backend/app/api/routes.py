from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func
from sqlalchemy.orm import Session
from typing import List
from app.db.session import SessionLocal
from app.tasks import ingest_photos
from app.models import Photo
from datetime import datetime
import logging
from fastapi.responses import FileResponse, StreamingResponse
from app.api.export_pdf import generate_person_pdf

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

@router.get("/stats")
def get_stats(db: Session = Depends(get_db)):
    total = db.query(Photo).count()
    processed = db.query(Photo).filter(Photo.blur_score.isnot(None)).count()
    return {"total_photos": total, "processed_photos": processed}

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
def get_event_details(
    event_id: int, 
    persons: List[str] = Query(None), 
    has_glasses: bool = None,
    db: Session = Depends(get_db)
):
    event = db.query(Event).filter(Event.id == event_id).first()
    if not event: raise HTTPException(404)
    
    # Base query
    query = db.query(Photo).filter(Photo.event_id == event_id)
    
    # Apply filters if present
    if (persons and len(persons) > 0) or has_glasses is not None:
        query = query.join(Face)
        if persons and len(persons) > 0:
            query = query.filter(Face.identity.in_(persons))
        if has_glasses is not None:
            query = query.filter(Face.has_glasses == (1 if has_glasses else 0))
    
    # Get top 50 photos for this event, sorted by score
    photos = query.order_by(Photo.blur_score.desc()).limit(50).all()
        
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
    List all detected people.
    Prioritizes solo photos for reference.
    """
    # 1. Get explicit clusters
    clusters = db.query(Cluster).all()
    results = []
    seen_identities = set()

    # Subquery for photos with exactly one face detection in the entire DB
    solo_photo_subq = db.query(Face.photo_id).group_by(Face.photo_id).having(func.count(Face.id) == 1).subquery()

    for p in clusters:
        # Find a solo photo where this person is the only detected face AND eyes are open
        solo_face = db.query(Face).filter(
            Face.cluster_id == p.id, 
            Face.photo_id.in_(solo_photo_subq),
            Face.eyes_open == 1
        ).order_by((Face.w * Face.h).desc(), Face.recognition_confidence.desc()).first()
        
        cover_photo_id = solo_face.photo_id if solo_face else (p.faces[0].photo_id if p.faces else None)
        
        results.append({
            "id": p.id,
            "name": p.name,
            "face_count": len(p.faces),
            "cover_photo_id": cover_photo_id,
            "is_solo_ref": solo_face is not None,
            "is_cluster": True
        })
        seen_identities.add(p.name)

    # 2. Virtual clusters from unique identities
    unclustered_rows = db.query(Face.identity).filter(
        Face.identity.isnot(None), 
        Face.identity != "",
        Face.identity.notin_(seen_identities)
    ).distinct().all()

    for row in unclustered_rows:
        identity = row.identity
        
        # Find solo photo with eyes open for this identity
        solo_face = db.query(Face).filter(
            Face.identity == identity, 
            Face.photo_id.in_(solo_photo_subq),
            Face.eyes_open == 1
        ).order_by((Face.w * Face.h).desc(), Face.recognition_confidence.desc()).first()
        
        count = db.query(func.count(Face.id)).filter(Face.identity == identity).scalar()
        
        if solo_face:
            cover_id = solo_face.photo_id
            is_solo = True
        else:
            first_face = db.query(Face).filter(Face.identity == identity).first()
            cover_id = first_face.photo_id if first_face else None
            is_solo = False

        results.append({
            "id": f"id_{identity}",
            "name": identity,
            "face_count": count,
            "cover_photo_id": cover_id,
            "is_solo_ref": is_solo,
            "is_cluster": False
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

@router.get("/people/identities")
def get_identities(db: Session = Depends(get_db)):
    """
    Get all unique identities found across all photos.
    """
    identities = db.query(Face.identity).filter(Face.identity.isnot(None), Face.identity != "").distinct().all()
    # Sort them for the UI
    name_list = sorted([i[0] for i in identities if i[0]])
    return name_list

@router.get("/people/{person_name}/photos")
def get_person_photos(person_name: str, db: Session = Depends(get_db)):
    """
    Get all photos where a specific person is present.
    """
    photos = db.query(Photo).join(Face).filter(
        (Face.identity == person_name) | (Face.cluster_id == db.query(Cluster.id).filter(Cluster.name == person_name).scalar_subquery())
    ).distinct().order_by(Photo.timestamp.asc()).all()
    
    # Simple serialization including blur_score for UI ranking indicator
    return [
        {
            "id": p.id,
            "filename": p.filename,
            "timestamp": p.timestamp,
            "blur_score": p.blur_score,
            "aesthetic_score": p.aesthetic_score
        } for p in photos
    ]

@router.get("/export/pdf")
def export_people_pdf(person: str, db: Session = Depends(get_db)):
    """
    Export a curated PDF of photos for a specific person.
    """
    if not person:
        raise HTTPException(status_code=400, detail="Person name required")
    
    pdf_buffer = generate_person_pdf(db, person)
    if not pdf_buffer:
        raise HTTPException(status_code=404, detail=f"No photos found for {person}")
    
    filename = f"Photos_{person.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.pdf"
    
    return StreamingResponse(
        pdf_buffer, 
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )

@router.get("/photos/{photo_id}")
def get_photo_detail(photo_id: int, db: Session = Depends(get_db)):
    photo = db.query(Photo).filter(Photo.id == photo_id).first()
    if not photo:
        raise HTTPException(status_code=404, detail="Photo not found")
    
    faces = db.query(Face).filter(Face.photo_id == photo_id).all()
    event = db.query(Event).filter(Event.id == photo.event_id).first()
    
    return {
        "id": photo.id,
        "filename": photo.filename,
        "path": photo.path,
        "timestamp": photo.timestamp,
        "blur_score": photo.blur_score,
        "aesthetic_score": photo.aesthetic_score,
        "event": {
            "id": event.id,
            "name": event.name,
            "location_name": event.location_name,
            "start_time": event.start_time
        } if event else None,
        "faces": [
            {
                "identity": f.identity,
                "has_glasses": bool(f.has_glasses),
                "eyes_open": bool(f.eyes_open),
                "recognition_confidence": f.recognition_confidence,
                "bbox": [f.x, f.y, f.w, f.h]
            } for f in faces
        ]
    }
