from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Float, Text, LargeBinary
from sqlalchemy.orm import relationship
from datetime import datetime
from app.db.session import Base

class Photo(Base):
    __tablename__ = "photos"

    id = Column(Integer, primary_key=True, index=True)
    path = Column(String, unique=True, index=True, nullable=False)
    filename = Column(String, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    size_bytes = Column(Integer)
    image_hash = Column(String, index=True) # For duplicate detection
    
    # Metadata
    latitude = Column(Float, nullable=True)
    longitude = Column(Float, nullable=True)
    
    # Aesthetic Scores
    blur_score = Column(Float, nullable=True)
    aesthetic_score = Column(Float, nullable=True) # NIMA score
    
    faces = relationship("Face", back_populates="photo")
    
    event_id = Column(Integer, ForeignKey("events.id"), nullable=True)
    event = relationship("Event", back_populates="photos", foreign_keys=[event_id])

class Event(Base):
    __tablename__ = "events"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, default="New Event")
    location_name = Column(String, nullable=True)  # e.g. "Paris"
    description = Column(Text, nullable=True)
    
    start_time = Column(DateTime, index=True)
    end_time = Column(DateTime)
    
    cover_photo_id = Column(Integer, ForeignKey("photos.id"), nullable=True)
    
    photos = relationship("Photo", back_populates="event", foreign_keys="Photo.event_id")
    cover_photo = relationship("Photo", foreign_keys=[cover_photo_id])

class Face(Base):
    __tablename__ = "faces"

    id = Column(Integer, primary_key=True, index=True)
    photo_id = Column(Integer, ForeignKey("photos.id"))
    
    # Bounding Box
    x = Column(Integer)
    y = Column(Integer)
    w = Column(Integer)
    h = Column(Integer)
    
    # Face Details
    identity = Column(String, index=True, nullable=True)
    has_glasses = Column(Integer, default=0) # 0: No, 1: Yes
    eyes_open = Column(Integer, default=1) # 0: No, 1: Yes
    detection_confidence = Column(Float, nullable=True)
    recognition_confidence = Column(Float, nullable=True)
    
    # Embedding (Store as binary or JSON for now if vector extension not available, but Plan said pgvector)
    # If pgvector is available, use Vector type. For MVP/SQLite fallback, we might use JSON/Blob
    embedding = Column(LargeBinary, nullable=True) 
    
    cluster_id = Column(Integer, ForeignKey("clusters.id"), nullable=True)
    
    photo = relationship("Photo", back_populates="faces")
    cluster = relationship("Cluster", back_populates="faces")

class Cluster(Base):
    __tablename__ = "clusters"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, default="Unknown Model") # e.g. "Person 1", or user renamed "Alice"
    
    faces = relationship("Face", back_populates="cluster")
