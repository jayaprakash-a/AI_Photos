import numpy as np
from sklearn.cluster import DBSCAN
from typing import List
from app.models import Photo

def cluster_photos_by_location(photos: List[Photo], eps_meters: float = 500.0, min_samples: int = 1):
    """
    Clusters photos based on Latitude and Longitude using DBSCAN.
    
    Args:
        photos: List of Photo objects.
        eps_meters: The maximum distance between two samples for one to be considered as in the neighborhood of the other.
        min_samples: The number of samples (or total weight) in a neighborhood for a point to be considered as a core point.
        
    Returns:
        Dict[int, List[Photo]]: Dictionary mapping cluster_id to list of photos.
                               Cluster -1 represents noise (no location or outlier).
    """
    # Filter photos with valid location
    valid_photos = [p for p in photos if p.latitude is not None and p.longitude is not None]
    
    if not valid_photos:
        return {}

    # Convert coordinates to radians for Haversine metric
    coords = np.radians([[p.latitude, p.longitude] for p in valid_photos])
    
    # Earth radius in meters
    kms_per_radian = 6371.0088
    eps_radians = (eps_meters / 1000.0) / kms_per_radian
    
    # DBSCAN
    db = DBSCAN(eps=eps_radians, min_samples=min_samples, algorithm='ball_tree', metric='haversine')
    db.fit(coords)
    
    clusters = {}
    for photo, label in zip(valid_photos, db.labels_):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(photo)
        
    return clusters
