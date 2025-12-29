# AI Photos: Design & Algorithms Guide

This document details the core logic and architectural decisions behind the AI Photos application.

---

## 1. Smart Event Clustering (DBSCAN)
**Endpoint**: `/organize`

### The Problem
Traditional time-gap clustering (e.g., "new event every 4 hours") fails for multifaceted trips. A morning museum visit and an afternoon cafe stop in the same city might be 5 hours apart but belong to the same "Day 1" event. Conversely, a flight taken 2 hours after a meeting clearly belongs to a different context.

### The Algorithm
We use **Density-Based Spatial Clustering of Applications with Noise (DBSCAN)**.
- **Inputs**: (Time, Latitude, Longitude)
- **Feature Scaling**: 
  - Time is scaled such that 1 unit = 4 hours.
  - Location is scaled such that 1 unit = ~20km.
- **Logic**: DBSCAN identifies "dense" regions in this 3D space. It can find events of any shape and automatically segregates isolated "noise" photos if they don't fit a pattern.
- **Benefit**: Captures the "density" of your activity. If you take 50 photos in a 20-minute span, that's a dense event regardless of the time gap to the next photo.

---

## 2. Highlight Diversity (ORB Feature Matching)
**Endpoint**: `/export/pdf`

### The Problem
Burst shots (taking 10 photos of the same thing) look repetitive in a highlight PDF. Spatial clustering alone isn't enough if you stand in one spot and take many similar shots.

### The Algorithm
We use **ORB (Oriented FAST and Rotated BRIEF)** for background comparison.
- **Process**:
  1. For every candidate photo in an event, we compare it against already selected photos.
  2. Detect ~500 keypoints and compute descriptors using ORB.
  3. Uses **BFMatcher (Brute-Force Matcher)** with Hamming distance.
  4. Counts "Good Matches" (Hamming distance < 40).
- **Threshold**: if more than 50 robust feature matches are found, the backgrounds are considered "duplicates" or "bursts," and the candidate is skipped.
- **Benefit**: Ensures the PDF contains unique, distinct moments even within the same location.

---

## 3. Solo-Shot Reference Selection
**Endpoint**: `/people`

### The Problem
A "People" view is only useful if the thumbnail clearly shows the person's face. Using a random photo often results in group shots where it's hard to tell who is who.

### The Logic
We prioritize photos using a multi-stage filter:
1. **Solo Constraint**: Only photos where `COUNT(faces) == 1` are considered high-priority.
2. **Quality Check**: Must have `eyes_open == True`.
3. **Selection**: Among qualified solo shots, we pick the one with the highest **recognition confidence** and largest **face area**.
4. **Fallback**: If no solo shot exists, we fall back to a group shot where they are present.

---

## 4. Robust PDF Imaging
**Function**: `create_pdf_document` in `export_pdf.py`

### The Problem
ReportLab and various browser viewers often handle EXIF orientation metadata inconsistently, leading to "sideways" photos. Direct scaling of raw image files also often leads to "stretching" if the aspect ratio calculation doesn't match the internal rotation.

### The Solution
- **Normalization**: Every image is first opened via PIL and passed through `ImageOps.exif_transpose()`. This physically rotates the pixel data to the correct orientation and strips the confusing metadata.
- **Buffer-Based Feeding**: The normalized image is saved to an in-memory `BytesIO` buffer.
- **Strict Scaling**: We calculate scaling based on the *actual* dimensions of the normalized pixel data, ensuring a perfect fit within PDF boundaries with zero stretching.

---

## 5. Async Progress Feedback
**Endpoint**: Long-running fetch in `app.js`

### The Problem
PDF generation (especially with ORB matching and high-quality rescaling) can take several seconds. A frozen UI leads to multiple clicks and user frustration.

### The Implementation
- **Overlay**: A global CSS overlay blocks further interaction while the PDF builds.
- **Dynamic Simulation**: While the `fetch()` is pending, a JavaScript interval increments a progress bar smoothly toward 90%, providing visual "aliveness."
- **Auto-Completion**: Upon receiving the Blob, the progress bar snaps to 100% and the file triggers a browser download.
