from sqlalchemy.orm import Session
from app.models import Photo, Event
from datetime import timedelta
import math
import logging
import numpy as np
from sklearn.cluster import DBSCAN

logger = logging.getLogger(__name__)

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees) in Kilometers.
    """
    if lat1 is None or lon1 is None or lat2 is None or lon2 is None:
        return 0.0

    # Convert decimal degrees to radians 
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

    # Haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a)) 
    r = 6371 # Radius of earth in kilometers.
    return c * r

def organize_library(db: Session, eps: float = 1.0, min_samples: int = 1):
    """
    Groups unassigned photos into Events using DBSCAN clustering on time and space.
    eps=1.0 roughly correlates to a neighborhood of 4 hours OR ~20km.
    """
    photos = db.query(Photo).filter(
        Photo.event_id.is_(None),
        Photo.timestamp.isnot(None)
    ).order_by(Photo.timestamp.asc()).all()
    
    if not photos:
        logger.info("No unassigned photos to organize.")
        return 0

    # 1. Prepare data for clustering
    # Scale: 1 unit = 4 hours
    # Scale: 1 degree ~ 111km. If we want 1 unit ~ 20km, factor is ~5.5
    data = []
    for p in photos:
        t = p.timestamp.timestamp() / (3600 * 4) 
        lat = (p.latitude if p.latitude is not None else 0) * 5.5
        lon = (p.longitude if p.longitude is not None else 0) * 5.5
        data.append([t, lat, lon])
    
    X = np.array(data)
    
    # 2. Run DBSCAN
    # min_samples=1 ensures every photo belongs to a cluster
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    labels = clustering.labels_
    
    # 3. Create Events from clusters
    from collections import defaultdict
    clusters = defaultdict(list)
    for i, label in enumerate(labels):
        clusters[label].append(photos[i])
        
    created_count = 0
    # Process each cluster as an event
    for label, cluster_photos in clusters.items():
        if label == -1: # Noise (shouldn't happen with min_samples=1)
            # Treat each noise point as its own event or ignore. 
            # With min_samples=1, everything is at least in its own cluster.
            continue
            
        # Create or find event for this cluster
        # Sort by time just in case
        cluster_photos.sort(key=lambda p: p.timestamp)
        first_photo = cluster_photos[0]
        
        event = create_new_event(db, first_photo)
        created_count += 1
        
        for photo in cluster_photos:
            add_photo_to_event(db, event, photo)
            
        finalize_event(db, event)
    
    db.commit()
    return created_count

def create_new_event(db: Session, first_photo: Photo) -> Event:
    event = Event(
        name=f"Event on {first_photo.timestamp.strftime('%Y-%m-%d')}",
        start_time=first_photo.timestamp,
        end_time=first_photo.timestamp,
        cover_photo_id=first_photo.id # Provisional cover
    )
    db.add(event)
    db.flush() # get ID
    
    first_photo.event_id = event.id
    return event

def add_photo_to_event(db: Session, event: Event, photo: Photo):
    photo.event_id = event.id
    # Update end time
    if photo.timestamp > event.end_time:
        event.end_time = photo.timestamp
    # Update start time (shouldn't happen if sorted, but safety)
    if photo.timestamp < event.start_time:
        event.start_time = photo.timestamp

def finalize_event(db: Session, event: Event):
    """
    Run ranking logic to pick best cover photo and set descriptive name/location.
    """
    photos = event.photos
    if not photos:
        return

    # 1. Pick Cover Photo (Best Rank)
    best_photo = max(photos, key=lambda p: calculate_rank_score(p))
    event.cover_photo_id = best_photo.id
    
    # 2. Infer Location Name (Reverse Geocoding)
    has_loc = any(p.latitude is not None for p in photos)
    if has_loc and not event.location_name:
        # Use centroid or just the best photo's location
        # Centroid is better but best_photo is simpler for MVP
        try:
            from app.utils.geocoding import reverse_geocode
            loc_name = reverse_geocode(best_photo.latitude, best_photo.longitude)
            if loc_name:
                event.location_name = loc_name
            else:
                 # Fallback
                 event.location_name = f"Lat: {best_photo.latitude:.2f}, Lon: {best_photo.longitude:.2f}"
        except Exception as e:
            logger.error(f"Failed to geocode event {event.id}: {e}")
            event.location_name = f"Lat: {best_photo.latitude:.2f}, Lon: {best_photo.longitude:.2f}"
    
    # 3. Rename based on Time (Morning/Afternoon/Eve)
    hour = event.start_time.hour
    time_of_day = "Day"
    if 5 <= hour < 12: time_of_day = "Morning"
    elif 12 <= hour < 17: time_of_day = "Afternoon"
    elif 17 <= hour < 21: time_of_day = "Evening"
    else: time_of_day = "Night"
    
    # Count faces
    total_faces = sum(len(p.faces) for p in photos)
    
    if total_faces > 0:
        event.description = f"{len(photos)} photos, {total_faces} people detected."
    else:
        event.description = f"{len(photos)} photos."

    # event.name = f"{time_of_day} on {event.start_time.strftime('%b %d')}"
    # Keep default name for now or enhance

def calculate_rank_score(photo: Photo) -> float:
    """
    Calculate a comprehensive 'Importance' score for a photo.
    """
    score = 0
    
    # 1. Sharpness (0-1000 usually, normalize?)
    if photo.blur_score:
        score += min(photo.blur_score, 500) / 10 # Cap at 50 pts
        
    # 2. People (Faces)
    # We prefer photos with people for covers
    if photo.faces:
        score += len(photo.faces) * 20
        
    # 3. Aesthetics (NIMA)
    if photo.aesthetic_score:
         score += photo.aesthetic_score * 5
         
    return score
