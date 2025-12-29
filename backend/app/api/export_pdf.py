import os
import io
import logging
from typing import List
from sqlalchemy.orm import Session
from sqlalchemy import func
from app.models import Photo, Event, Face
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak
from reportlab.lib.units import inch
from datetime import datetime
import logging
import cv2
import numpy as np
from sklearn.cluster import DBSCAN

def are_backgrounds_similar(path1: str, path2: str, threshold: int = 50) -> bool:
    """
    Check if two images have nearly identical backgrounds using ORB feature matching.
    threshold: number of 'good' matches to consider them similar.
    """
    try:
        img1 = cv2.imread(path1, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(path2, cv2.IMREAD_GRAYSCALE)
        if img1 is None or img2 is None:
            return False

        # Resize for speed - ORB is scale-invariant-ish but we just need a quick check
        img1 = cv2.resize(img1, (640, 480))
        img2 = cv2.resize(img2, (640, 480))

        orb = cv2.ORB_create(nfeatures=500)
        kp1, des1 = orb.detectAndCompute(img1, None)
        kp2, des2 = orb.detectAndCompute(img2, None)

        if des1 is None or des2 is None:
            return False

        # Use BFMatcher with Hamming distance
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        
        # Sort by distance
        matches = sorted(matches, key=lambda x: x.distance)
        
        # Count high-quality matches
        good_matches = [m for m in matches if m.distance < 40]
        
        return len(good_matches) > threshold
    except Exception as e:
        logger.error(f"Error in ORB match: {e}")
        return False

def select_spatially_diverse_photos(photos: List[Photo], count: int = 6) -> List[Photo]:
    """
    Selects photos to maximize spatial diversity and background variety.
    Uses ORB to skip duplicate backgrounds (bursts/duplicates).
    """
    if not photos:
        return []
    
    if len(photos) <= count:
        # Still check for duplicates even in small sets? 
        # For now, let's keep it simple: if the user only has 3 photos, they probably want all 3.
        # But if they are 3 identical burst shots, maybe not.
        pass

    # 1. Clustering by location (if coordinates exist)
    coords = []
    photos_with_coords = []
    photos_without_coords = []
    
    for p in photos:
        if p.latitude is not None and p.longitude is not None:
            coords.append([p.latitude, p.longitude])
            photos_with_coords.append(p)
        else:
            photos_without_coords.append(p)
            
    selected_photos = []
    
    def try_add_photo(candidate):
        # Check against already selected photos for background similarity
        for s in selected_photos:
            if are_backgrounds_similar(candidate.path, s.path):
                return False
        selected_photos.append(candidate)
        return True

    if coords:
        coords_np = np.array(coords)
        clustering = DBSCAN(eps=0.002, min_samples=1).fit(coords_np)
        labels = clustering.labels_
        
        clusters = {}
        for idx, label in enumerate(labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(photos_with_coords[idx])
            
        # 2. Pick from each cluster iteratively
        cluster_ids = list(clusters.keys())
        rounds = 0
        while len(selected_photos) < count and rounds < 10:
            added_any = False
            for cid in cluster_ids:
                if len(selected_photos) >= count:
                    break
                if clusters[cid]:
                    # Pick best in cluster
                    clusters[cid].sort(key=lambda p: (len(p.faces) * 10) + (p.aesthetic_score or 0), reverse=True)
                    # Try to add the best one
                    for i in range(len(clusters[cid])):
                        if try_add_photo(clusters[cid][i]):
                            clusters[cid].pop(i)
                            added_any = True
                            break
                    else:
                        # All in this cluster are too similar? 
                        # Pop the best one anyway to avoid infinite loop but don't add
                        clusters[cid].pop(0)
            if not added_any:
                break
            rounds += 1

    # 3. Fill remaining with photos without coords
    if len(selected_photos) < count and photos_without_coords:
        photos_without_coords.sort(key=lambda p: (len(p.faces) * 10) + (p.aesthetic_score or 0), reverse=True)
        for p in photos_without_coords:
            if len(selected_photos) >= count:
                break
            try_add_photo(p)
        
    return sorted(selected_photos, key=lambda p: p.timestamp or datetime.min)

def create_pdf_document(person_name: str, export_data: List[dict]) -> io.BytesIO:
    """
    Generates the PDF document.
    export_data: List of { "event": Event, "photos": [Photo] }
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=0.5*inch, bottomMargin=0.5*inch)
    styles = getSampleStyleSheet()
    
    # Custom styles
    header_style = ParagraphStyle(
        'HeaderStyle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor("#2C3E50"),
        alignment=1, # Center
        spaceAfter=30
    )
    
    event_title_style = ParagraphStyle(
        'EventTitle',
        parent=styles['Heading2'],
        fontSize=18,
        textColor=colors.HexColor("#34495E"),
        spaceBefore=20,
        spaceAfter=5
    )
    
    event_meta_style = ParagraphStyle(
        'EventMeta',
        parent=styles['Normal'],
        fontSize=12,
        textColor=colors.grey,
        spaceAfter=20
    )
    
    elements = []
    
    # Title Page
    elements.append(Spacer(1, 2*inch))
    elements.append(Paragraph(f"Photo Highlights: {person_name}", header_style))
    elements.append(Paragraph(f"Generated on {datetime.now().strftime('%B %d, %Y')}", styles['Normal']))
    elements.append(PageBreak())
    
    for item in export_data:
        event = item['event']
        photos = item['photos']
        
        if not photos:
            continue
            
        date_str = event.start_time.strftime('%Y-%m-%d') if event.start_time else "Unknown Date"
        loc_str = event.location_name if event.location_name else "Unknown Location"
        event_name = event.name if event.name else "Unnamed Event"
        
        for i, photo in enumerate(photos):
            if i > 0 or elements: # Always page break except maybe the very first item if title page handled it
                elements.append(PageBreak())
            
            # Header on EVERY page
            elements.append(Paragraph(f"<b>{event_name}</b> | {date_str} | {loc_str}", event_meta_style))
            elements.append(Spacer(1, 0.1*inch))
            
            # Photo
            if os.path.exists(photo.path):
                from PIL import Image as PILImage, ImageOps
                
                # 1. Load and normalize orientation
                with PILImage.open(photo.path) as pil_img:
                    pil_img = ImageOps.exif_transpose(pil_img)
                    orig_w, orig_h = pil_img.size
                    
                    # 2. Convert to a format ReportLab can use directly without re-reading from disk
                    # This ensures ReportLab doesn't apply its own (possibly conflicting) EXIF logic.
                    temp_img_io = io.BytesIO()
                    pil_img.save(temp_img_io, format="JPEG", quality=90)
                    temp_img_io.seek(0)
                
                # 3. Calculate scaling
                img = Image(temp_img_io)
                aspect = orig_h / orig_w
                
                # Fitting within 7.5x7.5 inch box
                max_w = 7.5 * inch
                max_h = 7.5 * inch
                
                # The crucial part: ONLY scale down, NEVER stretch up
                # and always maintain aspect ratio
                draw_w, draw_h = orig_w, orig_h # Start with pixels
                
                # Convert pixel dims to points (72 dpi) for initial comparison
                # But ReportLab 'Image' handles points natively if we set drawWidth/drawHeight.
                
                # Scale to fit width first
                if orig_w > max_w:
                    draw_w = max_w
                    draw_h = draw_w * aspect
                
                # Then scale to fit height if still too big
                if draw_h > max_h:
                    draw_h = max_h
                    draw_w = draw_h / aspect
                
                img.drawWidth = draw_w
                img.drawHeight = draw_h
                    
                elements.append(img)
                elements.append(Spacer(1, 0.2*inch))
                elements.append(Paragraph(f"Photo: {photo.filename}", styles['Italic']))
            else:
                elements.append(Paragraph(f"[Photo missing at {photo.path}]", styles['Normal']))

    doc.build(elements)
    buffer.seek(0)
    return buffer

def generate_person_pdf(db: Session, person_name: str) -> io.BytesIO:
    """
    Main entry point for generating the PDF for a specific person.
    """
    # 1. Find all photos where this person is present
    # Using identity string as per the model
    photos = db.query(Photo).join(Face).filter(Face.identity == person_name).all()
    
    if not photos:
        return None
        
    # 2. Group by Event
    events_map = {}
    for p in photos:
        eid = p.event_id or 0 # 0 for unassigned
        if eid not in events_map:
            events_map[eid] = []
        events_map[eid].append(p)
        
    export_data = []
    
    # Sort events by date descending
    sorted_event_ids = sorted(events_map.keys(), key=lambda eid: (db.query(Event).get(eid).start_time if eid != 0 and db.query(Event).get(eid) else datetime.min), reverse=True)
    
    for eid in sorted_event_ids:
        event = db.query(Event).get(eid) if eid != 0 else Event(name="Unassigned Photos")
        event_photos = events_map[eid]
        
        # 3. Select diverse photos for this event
        selected = select_spatially_diverse_photos(event_photos, count=6)
        
        export_data.append({
            "event": event,
            "photos": selected
        })
        
    # 4. Create PDF
    return create_pdf_document(person_name, export_data)
