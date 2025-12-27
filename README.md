# AI Photos App 📸

A local-first, AI-powered photo organization and management tool with intelligent event detection, face recognition, and aesthetic scoring.

## ✨ Features

### 🔍 Smart Photo Analysis
- **EXIF Metadata Extraction**: Automatically extracts timestamps, GPS coordinates, and camera settings
- **Face Recognition & Analysis**:
  - Detection and identity recognition across your library
  - **Eyeglasses Detection**: AI identifies if subjects are wearing glasses
  - **Eye State Analysis**: Detects if eyes are open or closed for every subject
  - High-confidence scoring for detection and recognition
- **Aesthetic Scoring**: 
  - Blur/Sharpness detection to find your best shots
  - NIMA (Neural Image Assessment) for aesthetic quality scoring
- **Database Dashboard**: Integrated web interface for direct database management at `/admin`

### 📅 Intelligent Event Organization
- **Automatic Event Clustering**: Groups photos into events based on:
  - Time proximity (configurable gap threshold)
  - Location clustering (GPS-based grouping)
  - Smart event naming with time-of-day detection
- **Event Annotations**: 
  - Custom event names and descriptions
  - **Location Geotagging**: Automatic reverse geocoding to resolve GPS coordinates to place names
  - **Place Autocomplete**: Live suggestions for landmarks, cities, tourist spots, and buildings
  - Geocoding notifications showing mapped coordinates

### 🗺️ Location Intelligence
- **Reverse Geocoding**: Converts GPS coordinates to readable location names (e.g., "Paris, France")
- **Forward Geocoding**: Search and autocomplete for places worldwide
- **Location-based Clustering**: Groups photos by geographic proximity
- **POI Support**: Recognizes landmarks, buildings, tourist attractions, and more

### 📊 Best Shots Gallery
- **Sharpness Ranking**: Automatically identifies your sharpest photos
- **Location Clustering**: Option to view best shots grouped by location
- **Multi-factor Scoring**: Combines sharpness, face count, and aesthetic scores

### 🎨 Modern UI
- **Dark Theme**: Beautiful, modern interface with smooth animations
- **Real-time Updates**: Live status indicators and background processing notifications
- **Interactive Modals**: Polished annotation interface with autocomplete
- **Responsive Design**: Optimized for various screen sizes

### 🔔 Notifications & Error Handling
- **Toast Notifications**: Real-time feedback for all operations
- **Error Queue**: Redis-backed error notification system
- **Background Processing**: Non-blocking AI operations with Celery

### 🖼️ Interactive Gallery
- **Multi-select Filtering**: Filter events by multiple people and eyeglasses status simultaneously
- **Photo Detail Viewer**: Full-screen lightbox with comprehensive metadata sidebar
- **Technicals View**: Real-time display of blur and aesthetic scores
- **Keyboard Navigation**: Quick exit with `Escape` key

## 🏗️ Architecture

- **Backend**: FastAPI (Python) with SQLAlchemy ORM
- **Database**: PostgreSQL for relational data
- **Queue**: Redis + Celery for background task processing
- **AI/ML**: PyTorch, FaceNet, OpenCV, scikit-learn
- **Geocoding**: OpenStreetMap Nominatim API
- **Frontend**: Vanilla JavaScript with modern CSS

## 🚀 Getting Started

### 1. Prerequisites
- **Python 3.10+**
- **PostgreSQL** (Database) and **Redis** (Queue)
  - *Option A (Docker):* Run `docker-compose up -d` in this directory.
  - *Option B (Manual):* Install them locally and update `backend/app/core/config.py` if defaults (localhost:5432/6379) don't match.

### 2. Installation
1.  **Backend Setup**:
    ```bash
    cd backend
    python -m venv venv
    venv\Scripts\activate
    pip install -r requirements.txt
    ```

2.  **AI Model Setup**:
    To enable face recognition and analysis features, you must download the pre-trained model files.
    - **Download**: [Download AI Models](https://drive.google.com/drive/folders/1aJu__xIydSxNKTI8jxdNqBxSzAGvGAVe?usp=sharing)
    - **Installation**: Extract the downloaded `.pth` files and paste them into the following directory:
      `backend/app/ai/face_recognition/models/`

### 3. Running the App
You need two terminal windows running simultaneously.

**Terminal 1: The API Server & UI**
Double-click `run_backend.bat`  
*OR run:*
```bash
cd backend
venv\Scripts\activate
uvicorn app.main:app --reload --port 8001
```

**Terminal 2: The AI Worker** (Background processing)
Double-click `run_worker.bat`  
*OR run:*
```bash
cd backend
venv\Scripts\activate
celery -A app.tasks worker --loglevel=info -P solo
```

### 4. Using the App
1.  Open your browser to: **[http://localhost:8001](http://localhost:8001)**
2.  **Ingest Photos**:
    - Enter the full path to your photo directory (e.g., `C:\Users\You\Pictures\Vacation`) in the input box on the Dashboard.
    - Click "Start Ingestion".
3.  **Monitor Progress**:
    - Watch the "Worker" terminal window to see files being scanned and processed.
    - Check the Dashboard for real-time statistics
4.  **Organize into Events**:
    - Click "Run AI Organizer" to automatically group photos into smart events
    - Events are created based on time and location clustering
5.  **View & Annotate Events**:
    - Navigate to "Smart Library" to see all detected events
    - Click any event to view details
    - Use the "✏️ Annotate Event" button to:
      - Edit event name and description
      - Set or modify location (with live autocomplete suggestions)
      - Get geocoding feedback with coordinates
6.  **Browse Best Shots**:
    - Click "Best Shots" in the sidebar to see the sharpest images
    - Toggle "Group by Location" to group by geographic areas
7.  **Manage Database**:
    - Visit **[http://localhost:8001/admin](http://localhost:8001/admin)** to manage raw database entries (Photos, Faces, Events) directly.

## 📡 API Reference

The backend provides a RESTful API under the `/api/v1` prefix.

### Photos
- `GET /photos/best`: List top-rated photos with optional location clustering
- `GET /photos/{id}/image`: Serve the actual image file
- `GET /photos/{id}`: Detailed metadata including detected faces and event info

### Library & Events
- `POST /organize`: Trigger the AI grouping algorithm
- `GET /events`: List all discovered events
- `GET /events/{id}`: Get photos for an event with multi-person filtering
- `PATCH /events/{id}`: Annotate event name, description, and location

### AI & People
- `GET /people/identities`: Get list of uniquely recognized people
- `GET /people`: Get clusters/people with cover photos
- `PATCH /people/{id}`: Update person/cluster names

### System
- `GET /stats`: Current library statistics (total vs processed)
- `GET /health`: Backend connectivity check
- `GET /notifications`: Fetch recent background task errors
- `GET /places/search?q=...`: Geographic name search/autocomplete

## 🎯 Key Workflows

### Photo Ingestion
1. Point to a directory containing photos
2. Background worker extracts EXIF data (timestamp, GPS)
3. AI processes each photo:
   - Face detection and embedding
   - Blur/sharpness scoring
   - Aesthetic quality assessment

### Event Organization
1. Click "Run AI Organizer"
2. Algorithm clusters photos by:
   - Time gaps (default: 4 hours)
   - Location distance (default: 20km)
3. Events are auto-named and geotagged
4. Best photo selected as cover based on multi-factor scoring

### Event Annotation
1. Open any event from Smart Library
2. Click "✏️ Annotate Event"
3. Type location name to see live suggestions:
   - Cities, countries
   - Landmarks (e.g., "Eiffel Tower")
   - Tourist attractions
   - Buildings and POIs
4. Save to get geocoding confirmation with coordinates

## 🛠 Troubleshooting

### Database Errors
Run the initialization script manually if tables are missing:
```bash
backend\venv\Scripts\python backend/init_db.py
```

### Resetting the App
To clear all data and stop all tasks:
```bash
backend\venv\Scripts\python backend/reset_app.py
```

### Geocoding Issues
- Nominatim API has rate limits (1 request/second)
- Requires internet connection for place search
- Results are in English by default

## 📁 Project Structure

```
Photos/
├── backend/
│   ├── app/
│   │   ├── ai/              # AI modules (face detection, scoring, organizer)
│   │   ├── api/             # FastAPI routes
│   │   ├── core/            # Configuration
│   │   ├── db/              # Database session management
│   │   ├── utils/           # Utilities (EXIF, clustering, geocoding)
│   │   ├── models.py        # SQLAlchemy models
│   │   ├── tasks.py         # Celery background tasks
│   │   └── main.py          # FastAPI application
│   ├── requirements.txt
│   └── init_db.py
├── frontend/
│   └── ui/
│       ├── index.html       # Main UI
│       ├── app.js           # Frontend logic
│       └── style.css        # Styling
├── docker-compose.yml       # PostgreSQL + Redis
├── run_backend.bat
├── run_worker.bat
└── README.md
```

## 🔮 Future Enhancements

- Face clustering and person identification
- Advanced search and filtering
- Photo editing capabilities
- Mobile app support
- Cloud backup integration
- Sharing and collaboration features

## 📝 License

This is a personal project for photo management and organization.
