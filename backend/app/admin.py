from sqladmin import ModelView
from app.models import Photo, Face, Event, Cluster

class PhotoAdmin(ModelView, model=Photo):
    column_list = [Photo.id, Photo.filename, Photo.timestamp, Photo.blur_score, Photo.aesthetic_score]
    column_searchable_list = [Photo.filename, Photo.path]
    column_sortable_list = [Photo.id, Photo.timestamp, Photo.blur_score, Photo.aesthetic_score]
    column_default_sort = ("timestamp", True)
    name_plural = "Photos"
    icon = "fa-solid fa-image"

class FaceAdmin(ModelView, model=Face):
    column_list = [Face.id, "photo.filename", Face.identity, Face.has_glasses, Face.eyes_open, Face.recognition_confidence]
    column_searchable_list = [Face.identity]
    column_sortable_list = [Face.id, Face.recognition_confidence]
    name_plural = "Faces"
    icon = "fa-solid fa-user-circle"

class EventAdmin(ModelView, model=Event):
    column_list = [Event.id, Event.name, Event.location_name, Event.start_time, Event.end_time]
    column_searchable_list = [Event.name, Event.location_name]
    column_sortable_list = [Event.id, Event.start_time]
    name_plural = "Events"
    icon = "fa-solid fa-calendar-days"

class ClusterAdmin(ModelView, model=Cluster):
    column_list = [Cluster.id, Cluster.name]
    column_searchable_list = [Cluster.name]
    name_plural = "Clusters"
    icon = "fa-solid fa-people-group"
