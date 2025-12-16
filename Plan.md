# Photos App - Architecture & Requirements (Revised)
## 1. Project Overview
A local-first, AI-powered application to organize large event photo collections (vacations, weddings), curate best shots based on aesthetic and emotional criteria, and generate video memories.
*Key Constraints:*
-   *Local Only:* All AI processing must happen on-device (Privacy).
-   *No Cost:* Use open-source/free models only.
-   *Long-term Analysis:* Persist comprehensive analysis data (People ID, Emotions, Scene Tags, Aesthetic Scores) for every image in the DB for future querying/analytics.
## 2. User Workflow
1.  *Ingest:* User points app to a folder (e.g., "Paris Vacation"). App starts background processing.
2.  *Organize (AI):* App clusters faces and locations.
3.  *Label:* User names identifying faces (e.g., "Cluster 1" -> "Alice") and corrects key landmarks.
4.  *Curate:* App selects "Best N" photos per sub-event based on aesthetic score (Smiling, Open Eyes, Lighting).
5.  *Review:* User reviews selection. Options:
    -   Approve: Use as is.
    -   Enhance: Run AI restoration on good content but poor quality photos.
    -   Replace: Choose from other candidates.
6.  *Create:* Generate slideshow/video.
## 3. Technical Architecture (MVP)
### A. Tech Stack
-   *Backend:* Python (FastAPI/Django) - Essential for ML ecosystem.
-   *Frontend:* Next.js / React - Responsive UI for gallery and selection.
-   *Database:* PostgreSQL (Metadata) + pgvector (Embeddings).
-   *Queue/Background:* Redis + Celery/BullMQ (Essential for long-running AI tasks).
### B. AI Implementation Strategy ("The How")
#### 1. Face Recognition
-   *Detection:* RetinaFace or MTCNN (Open Source/MIT License. Free).
-   *Embedding:* FaceNet or InsightFace (Open Source/MIT License. Free).
-   *Clustering (Unsupervised):* DBSCAN or Chinese Whispers (Standard Algorithms via Scikit-Learn. Free).
-   *Recognition (Supervised):* KNN classifier (Standard Algorithms. Free).
#### 2. Landmark/Scene Recognition
-   *Primary (Metadata):* GPS coordinates -> Reverse Geocoding (OpenStreetMap/Nominatim local docker).
*Fallback (No GPS):* 
    -   Time Clustering: Infer location from other photos in the same time cluster that do have GPS.
    -   Visual Scene: Use CLIP to detect "Eiffel Tower" or "Mountain" to assign generic location tags.
-   *Secondary (Visual):* *CLIP* (OpenAI/LAION).
    -   Method: Compare image embedding with text prompts ["Beach", "Mountain", "Eiffel Tower", "Wedding Cake"].
    -   Why: Zero-shot classification. No need to train specific landmark models.
#### 3. Aesthetic Scoring & Selection
-   *Model:* Open Source implementation of NIMA (e.g., kentsyu/Neural-IMage-Assessment or LAION Aesthetic Predictor). *Free to use.*
-   *Criteria Checks:*
    -   Blur: Laplacian Variance method (Opencv).
    -   Eyes/Smile: Facial landmark detection (dlib/mediapipe).
-   *Selection Logic:*
    -   Group by Time (Sub-event).
    -   Filter (Blurry < Threshold).
    -   Rank by (Smile Score + NIMA Score).
    -   Take Top N.
#### 4. Image Enhancement (Phase 2 feature)
-   *Face Restoration:* GFPGAN or CodeFormer (State-of-the-art face fixers).
-   *Upscaling:* Real-ESRGAN.
## 4. Development Phases
### Phase 1: The Core (MVP)
-   [ ] Ingestion Pipeline (Recursive folder scan).
-   [ ] Face Clustering (Unlabeled).
-   [ ] Basic Aesthetic Score (Blur + Lighting only).
-   [ ] Simple Slideshow Generation (FFmpeg).
### Phase 2: Refinement & User Control
-   [ ] User UI for naming faces ("Who is this?").
-   [ ] "Smart Selection" (Eyes open/Smile detection).
-   [ ] *Digital Album/Collage:* Generate static or video-based "photo album" layouts (e.g., text + multiple photos per page).
-   [ ] Advanced Video (Beat sync logic).
### Phase 3: Enhancement & Scale
-   [ ] *Generative Cleanup:* "Magic Eraser" to remove crowds/objects (LaMa or similar).
-   [ ] *Smart Retouch:* User-selectable fixes (Remove face shadows, clear skin).
-   [ ] "Fix this photo" button (GFPGAN integration).
-   [ ] Handling >100GB libraries (Optimization).
## 5. Hardware & Performance Estimates (Low Spec)
### Requirements
-   *Minimum:* 4GB RAM, 2 Cores. (Runs slow but works).
-   *Recommended:* 8GB RAM, 4 Cores.
-   *Optional:* NVIDIA GPU (Speeds up AI by 10-20x), but CPU-only is fully supported.
### Estimated Processing Time (CPU Only)
Assumption: 2,000 photos (~3GB) on a modern laptop CPU (e.g., i5/Ryzen 5) or Mini PC.
| Step | Time per Photo | Total for 2,000 Photos | Notes |
| :--- | :--- | :--- | :--- |
| *Ingest (Hash/Exif)* | ~0.05s | ~2 mins | Purely I/O bound. |
| *Face Detection* | ~0.3s | ~10 mins | Depends on image resolution. |
| *Face Embedding* | ~0.3s | ~10 mins | Run only on detected faces. |
| *Aesthetic Score* | ~0.2s | ~7 mins | Low resolution is fine. |
| *TOTAL* | *~1s* | *~30-45 mins* | *Background Job.* |
Note: The app runs in the background. You can start the import and check back in an hour.