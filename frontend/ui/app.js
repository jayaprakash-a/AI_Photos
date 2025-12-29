const API_BASE = '/api/v1';

// State
let currentSection = 'dashboard';

// Init
document.addEventListener('DOMContentLoaded', () => {
    checkBackendHealth();
    loadDashboardStats(); // Start periodic updates
});

// Navigation
function showSection(sectionId) {
    document.querySelectorAll('.content-section').forEach(el => el.classList.remove('active'));
    document.getElementById(sectionId).classList.add('active');

    document.querySelectorAll('.nav-item').forEach(el => el.classList.remove('active'));
    // Find button that calls specific showSection
    if (event && event.target) {
        event.target.classList.add('active');
    }

    if (sectionId === 'bestshots') {
        loadBestShots();
    } else if (sectionId === 'library') {
        loadEvents();
    } else if (sectionId === 'people') {
        loadPeople();
    } else if (sectionId === 'dashboard') {
        loadDashboardStats();
    }

    checkBackendHealth();
}

// Backend Health
async function checkBackendHealth() {
    const el = document.getElementById('backend-status');
    if (!el) return;
    try {
        await fetch(`${API_BASE}/health`);
        el.innerText = 'Backend Online';
        el.parentElement.classList.add('online');
    } catch (e) {
        el.innerText = 'Backend Offline';
        el.parentElement.classList.remove('online');
    }
}

// Ingestion
async function startIngestion() {
    const dir = document.getElementById('folderPath').value;
    const statusEl = document.getElementById('ingestionStatus');

    if (!dir) return;

    statusEl.textContent = "Starting ingestion... Check command prompt for logs.";

    try {
        const res = await fetch(`${API_BASE}/ingest?directory=${encodeURIComponent(dir)}`);
        if (!res.ok) throw new Error("Request failed");
        const data = await res.json();
        statusEl.textContent = `Success! Task ID: ${data.task_id}. Processing in background...`;
        statusEl.style.color = 'var(--success)';

        // Refresh stats after a bit
        setTimeout(loadDashboardStats, 2000);
    } catch (e) {
        statusEl.textContent = "Error triggering ingestion. Is backend running?";
        statusEl.style.color = 'var(--danger)';
    }
}

// Best Photos
async function loadBestShots() {
    const grid = document.getElementById('bestShotsGrid');
    grid.innerHTML = '<div class="loading-spinner">Loading...</div>';

    const toggle = document.getElementById('clusterToggle');
    const useClustering = toggle ? toggle.checked : false;

    try {
        const res = await fetch(`${API_BASE}/photos/best?limit=20&group_by_location=${useClustering}`);
        const photos = await res.json();

        grid.innerHTML = '';
        if (photos.length === 0) {
            grid.innerHTML = '<p>No photos found. Run ingestion first.</p>';
            return;
        }

        photos.forEach(photo => {
            const div = document.createElement('div');
            div.className = 'photo-item';
            div.onclick = () => openPhotoDetail(photo.id);
            // Placeholder: We need a way to serve images.
            // Assuming for MVP we can't serve local files easily due to browser security
            // UNLESS we mount the photo directory in backend. 
            // For MVP: We show filename and score.

            div.innerHTML = `
                <img src="${API_BASE}/photos/${photo.id}/image" alt="${photo.filename}" loading="lazy">
                <div class="photo-info">
                    <div class="photo-name">${photo.filename}</div>
                    <div class="photo-score">${photo.blur_score ? photo.blur_score.toFixed(1) : 'N/A'}</div>
                </div>
            `;
            // If we want to verify image display, we'd need a /static/photos endpoint
            // mirroring the local path, which is complex for arbitrary paths.

            grid.appendChild(div);
        });

    } catch (e) {
        grid.innerHTML = '<p style="color:var(--danger)">Error loading photos.</p>';
    }
}

// Dashboard Stats
async function loadDashboardStats() {
    try {
        const res = await fetch(`${API_BASE}/stats`);
        const data = await res.json();

        document.getElementById('statTotalPhotos').textContent = data.total_photos;
        document.getElementById('statProcessed').textContent = data.processed_photos;
    } catch (e) {
        console.error("Failed to load stats", e);
    }
}

// --- Smart Events ---

async function triggerOrganize() {
    const statusEl = document.getElementById('organizeStatus');
    statusEl.textContent = "Running AI Organizer... This may take a while.";
    statusEl.style.color = 'var(--text-secondary)';

    try {
        const res = await fetch(`${API_BASE}/organize`, { method: 'POST' });
        const data = await res.json();
        statusEl.textContent = data.message;
        statusEl.style.color = 'var(--success)';
        loadEvents(); // Refresh list
    } catch (e) {
        statusEl.textContent = "Error running organizer.";
        statusEl.style.color = 'var(--danger)';
    }
}

async function loadEvents() {
    const grid = document.getElementById('eventsGrid');
    grid.innerHTML = '<div class="loading-spinner">Loading Events...</div>';

    try {
        const res = await fetch(`${API_BASE}/events`);
        const events = await res.json();

        grid.innerHTML = '';
        if (events.length === 0) {
            grid.innerHTML = '<p>No events found. Run "Organize Library" first.</p>';
            return;
        }

        events.forEach(event => {
            const card = document.createElement('div');
            card.className = 'event-card';
            card.onclick = () => openEvent(event.id);

            // Format Date
            const date = new Date(event.start_time).toLocaleDateString(undefined, {
                year: 'numeric', month: 'long', day: 'numeric'
            });

            const coverUrl = event.cover_photo_id ? `${API_BASE}/photos/${event.cover_photo_id}/image` : '';

            card.innerHTML = `
                <div class="event-cover">
                    ${coverUrl ? `<img src="${coverUrl}" loading="lazy">` : ''}
                </div>
                <div class="event-info">
                    <div class="event-title">${event.name}</div>
                    <div class="event-meta">
                        <div>📅 ${date}</div>
                        <div>📍 ${event.location_name || 'Unknown Location'}</div>
                        <div>📸 ${event.photo_count} photos</div>
                    </div>
                </div>
            `;
            grid.appendChild(card);
        });
    } catch (e) {
        grid.innerHTML = '<p style="color:var(--danger)">Error loading events.</p>';
    }
}

function editEventDetails() {
    if (!currentEventId) return;

    const data = window.currentEventData || {};
    document.getElementById('editEventName').value = data.name || document.getElementById('detailTitle').textContent;
    document.getElementById('editEventLocation').value = data.location_name || "";
    document.getElementById('editEventDesc').value = data.description || "";

    document.getElementById('editEventModal').classList.add('active');
}

function closeEditModal() {
    document.getElementById('editEventModal').classList.remove('active');
}

async function saveEventDetails() {
    const newName = document.getElementById('editEventName').value;
    const newLocation = document.getElementById('editEventLocation').value;
    const newDesc = document.getElementById('editEventDesc').value;

    if (!currentEventId) return;

    const payload = {};
    if (newName) payload.name = newName;
    if (newLocation !== undefined) payload.location_name = newLocation;
    if (newDesc) payload.description = newDesc;

    if (Object.keys(payload).length === 0) {
        closeEditModal();
        return;
    }

    try {
        const res = await fetch(`${API_BASE}/events/${currentEventId}`, {
            method: 'PATCH',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        if (!res.ok) throw new Error("Update failed");

        const data = await res.json();
        showToast(data.message || "Event updated successfully", "success");

        openEvent(currentEventId); // Refresh details view
        loadEvents(); // Refresh grid in background
        closeEditModal();
    } catch (e) {
        showToast("Failed to update event", "error");
    }
}

async function loadPeople() {
    const grid = document.getElementById('peopleGrid');
    if (!grid) return;
    grid.innerHTML = '<div class="loading-spinner">Loading People...</div>';

    try {
        const res = await fetch(`${API_BASE}/people`);
        const people = await res.json();

        grid.innerHTML = '';
        if (people.length === 0) {
            grid.innerHTML = '<p>No people detected yet. Ensure AI processing is complete.</p>';
            return;
        }

        people.forEach(p => {
            const card = document.createElement('div');
            card.className = 'person-card';

            const coverUrl = p.cover_photo_id ? `${API_BASE}/photos/${p.cover_photo_id}/image` : '';

            card.innerHTML = `
                <div class="person-cover" onclick="openPersonPhotos('${p.name}')" style="cursor:pointer">
                    ${p.is_solo_ref ? '<div class="solo-tag">Reference Shot</div>' : ''}
                    ${coverUrl ? `<img src="${coverUrl}" loading="lazy">` : `<div class="p-placeholder">${p.name[0]}</div>`}
                </div>
                <div class="person-info">
                    <div class="person-name" title="${p.name}" onclick="openPersonPhotos('${p.name}')" style="cursor:pointer">${p.name}</div>
                    <div class="person-meta">${p.face_count} photos</div>
                    <button class="btn-primary export-btn" onclick="exportPersonPDF('${p.name}')">
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="margin-right:4px">
                            <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v4a2 2 0 012-2h14a2 2 0 012 2zM7 10l5 5 5-5M12 15V3"/>
                        </svg>
                        Export PDF
                    </button>
                </div>
            `;
            grid.appendChild(card);
        });
    } catch (e) {
        console.error(e);
        grid.innerHTML = '<p style="color:var(--danger)">Error loading people.</p>';
    }
}

async function openPersonPhotos(name) {
    document.getElementById('peopleGridView').style.display = 'none';
    const detailView = document.getElementById('personDetailView');
    detailView.style.display = 'block';

    document.getElementById('personDetailName').textContent = name;
    const grid = document.getElementById('personPhotosGrid');
    grid.innerHTML = '<div class="loading-spinner">Loading photos...</div>';

    try {
        const res = await fetch(`${API_BASE}/people/${encodeURIComponent(name)}/photos`);
        const photos = await res.json();

        grid.innerHTML = '';
        if (photos.length === 0) {
            grid.innerHTML = '<p style="grid-column:1/-1; text-align:center">No photos found.</p>';
            return;
        }

        photos.forEach(photo => {
            const container = document.createElement('div');
            container.className = 'person-photo-item';

            // Format time
            const dateStr = photo.timestamp ? new Date(photo.timestamp).toLocaleDateString() : 'Unknown date';

            container.innerHTML = `
                <div class="photo-item" onclick="openPhotoDetail(${photo.id})">
                    <img src="${API_BASE}/photos/${photo.id}/image" loading="lazy">
                </div>
                <div class="photo-item-title">${photo.filename} • ${dateStr}</div>
            `;
            grid.appendChild(container);
        });
    } catch (e) {
        console.error(e);
        grid.innerHTML = '<p style="color:var(--danger)">Error loading photos.</p>';
    }
}

function closePersonDetail() {
    document.getElementById('personDetailView').style.display = 'none';
    document.getElementById('peopleGridView').style.display = 'block';
}

async function exportPersonPDF(name) {
    const overlay = document.getElementById('exportOverlay');
    const bar = document.querySelector('.progress-bar-fill');

    // Show overlay
    overlay.classList.add('active');
    bar.style.width = '0%';

    // Fake progress animation (reaches 90% in 5 seconds)
    let progress = 0;
    const interval = setInterval(() => {
        if (progress < 90) {
            progress += 1;
            bar.style.width = `${progress}%`;
        }
    }, 50);

    try {
        const res = await fetch(`${API_BASE}/export/pdf?person=${encodeURIComponent(name)}`);

        if (!res.ok) throw new Error('Failed to generate PDF');

        const blob = await res.blob();

        // Complete progress
        clearInterval(interval);
        bar.style.width = '100%';

        setTimeout(() => {
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `Highlights_${name.replace(/\s+/g, '_')}.pdf`;
            document.body.appendChild(a);
            a.click();
            a.remove();

            // Cleanup
            overlay.classList.remove('active');
            showToast(`PDF exported for ${name}`, 'success');
        }, 300);

    } catch (e) {
        clearInterval(interval);
        overlay.classList.remove('active');
        console.error(e);
        showToast(`Failed to export PDF`, 'error');
    }
}

async function openEvent(eventId) {
    currentEventId = eventId; // Set current event ID
    document.getElementById('eventsGrid').style.display = 'none';
    const detailView = document.getElementById('eventDetailView');
    detailView.style.display = 'block';

    // Get filter values
    const personCheckboxes = document.querySelectorAll('.person-checkbox:checked');
    const persons = Array.from(personCheckboxes).map(cb => cb.value);
    const glassesToggle = document.getElementById('glassesFilter');
    const hasGlasses = glassesToggle ? glassesToggle.checked : false;

    // Update count display
    const countEl = document.getElementById('selectedPeopleCount');
    if (countEl) {
        countEl.textContent = persons.length > 0 ? `${persons.length} selected` : 'Select People';
    }

    // Reset contents
    document.getElementById('detailTitle').textContent = "Loading...";
    document.getElementById('detailDesc').textContent = "";
    document.getElementById('detailGallery').innerHTML = '<div class="loading-spinner">Loading...</div>';

    try {
        let url = `${API_BASE}/events/${eventId}?`;
        if (persons && persons.length > 0) {
            persons.forEach(p => url += `persons=${encodeURIComponent(p)}&`);
        }
        if (hasGlasses) url += `has_glasses=true&`;

        const res = await fetch(url);
        const data = await res.json();
        const event = data.event;
        const photos = data.photos;

        window.currentEventData = event; // Store for editing

        document.getElementById('detailTitle').textContent = event.name;
        document.getElementById('detailDesc').textContent = `${event.description || ''} • ${event.location_name || ''}`;

        const gallery = document.getElementById('detailGallery');
        gallery.innerHTML = '';

        if (photos.length === 0) {
            gallery.innerHTML = '<p style="grid-column: 1/-1; text-align: center; padding: 2rem;">No photos match these filters.</p>';
            return;
        }

        photos.forEach(photo => {
            const div = document.createElement('div');
            div.className = 'photo-item';
            div.onclick = () => openPhotoDetail(photo.id);
            div.innerHTML = `
                <img src="${API_BASE}/photos/${photo.id}/image" loading="lazy">
                <div class="photo-info">
                    <div class="photo-score">${photo.blur_score ? photo.blur_score.toFixed(0) : ''}</div>
                </div>
            `;
            gallery.appendChild(div);
        });

    } catch (e) {
        console.error(e);
        document.getElementById('detailGallery').innerHTML = '<p style="color:var(--danger)">Error loading photos.</p>';
    }
}

async function applyFilters() {
    if (currentEventId) {
        openEvent(currentEventId);
    }
}

async function loadIdentities() {
    try {
        const res = await fetch(`${API_BASE}/people/identities`);
        const identities = await res.json();
        const container = document.getElementById('personFilterOptions');
        if (!container) return;

        container.innerHTML = '';

        identities.forEach(name => {
            const div = document.createElement('div');
            div.className = 'multi-select-option';
            div.innerHTML = `
                <input type="checkbox" class="person-checkbox" value="${name}" id="p_${name.replace(/\s+/g, '_')}" onchange="applyFilters()">
                <label for="p_${name.replace(/\s+/g, '_')}">${name}</label>
            `;
            container.appendChild(div);
        });
    } catch (e) {
        console.error("Failed to load identities", e);
    }
}

function toggleMultiSelect(e) {
    e.stopPropagation();
    const options = document.getElementById('personFilterOptions');
    options.classList.toggle('active');
}

// Close multi-select on click outside
document.addEventListener('click', (e) => {
    const options = document.getElementById('personFilterOptions');
    if (options && options.classList.contains('active') && !e.target.closest('.multi-select-container')) {
        options.classList.remove('active');
    }
});

function closeEventDetail() {
    document.getElementById('eventDetailView').style.display = 'none';
    document.getElementById('eventsGrid').style.display = 'grid';
    currentEventId = null; // Track current
}

let currentEventId = null; // Global tracker

// --- Notifications ---

function pollNotifications() {
    setInterval(async () => {
        try {
            const res = await fetch(`${API_BASE}/notifications`);
            if (res.ok) {
                const errors = await res.json();
                errors.forEach(msg => showToast(msg, 'error'));
            }
        } catch (e) {
            // Using console error only to avoid infinite toast loops if poll fails
            console.error("Poll failed", e);
        }
    }, 5000); // Poll every 5 seconds
}

function showToast(message, type = 'info') {
    const container = document.getElementById('toast-container');
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.innerHTML = `<span>${message}</span> <button onclick="this.parentElement.remove()" style="background:none;border:none;color:inherit;cursor:pointer;">&times;</button>`;

    container.appendChild(toast);

    // Auto remove after 5s
    setTimeout(() => {
        toast.style.animation = 'fadeOut 0.5s forwards';
        setTimeout(() => toast.remove(), 500);
    }, 5000);
}

// Start Polling
document.addEventListener('DOMContentLoaded', () => {
    // Existing Init
    // checkBackendHealth called in original init
    pollNotifications();

    // Setup autocomplete for location input
    setupLocationAutocomplete();

    // Load initial data
    loadIdentities();
});

// --- Place Autocomplete ---
let autocompleteHandler = null; // Store the handler to prevent duplicates

function setupLocationAutocomplete() {
    const input = document.getElementById('editEventLocation');
    if (!input) return;

    // Remove existing listener if any
    if (autocompleteHandler) {
        input.removeEventListener('input', autocompleteHandler);
    }

    // Create new handler
    autocompleteHandler = debounce(async (e) => {
        const query = e.target.value;
        const listId = 'location-suggestions';

        // Remove ALL existing suggestion lists (in case of duplicates)
        document.querySelectorAll('.autocomplete-list').forEach(list => list.remove());

        if (query.length < 3) return;

        try {
            const res = await fetch(`${API_BASE}/places/search?q=${encodeURIComponent(query)}`);
            const results = await res.json();

            if (results.length > 0) {
                const list = document.createElement('ul');
                list.id = listId;
                list.className = 'autocomplete-list';

                results.forEach(place => {
                    const item = document.createElement('li');
                    item.textContent = place.name;
                    item.onclick = () => {
                        document.getElementById('editEventLocation').value = place.name;
                        list.remove();
                    };
                    list.appendChild(item);
                });

                // Position relative to input
                const parent = e.target.parentNode;
                parent.style.position = 'relative';
                parent.appendChild(list);
            }
        } catch (err) {
            console.error("Autocomplete failed", err);
        }
    }, 300);

    // Add the new listener
    input.addEventListener('input', autocompleteHandler);
}

// Close autocomplete on click outside
document.addEventListener('click', (e) => {
    if (e.target.id !== 'editEventLocation') {
        const list = document.getElementById('location-suggestions');
        if (list) list.remove();
    }
});

// --- Photo Detail Viewer ---

async function openPhotoDetail(photoId) {
    const modal = document.getElementById('photoDetailModal');
    const largeImg = document.getElementById('detailLargeImage');

    // Show modal immediately with loading state
    largeImg.src = `${API_BASE}/photos/${photoId}/image`;
    modal.classList.add('active');

    try {
        const res = await fetch(`${API_BASE}/photos/${photoId}`);
        const data = await res.json();

        // Populate Sidebar
        document.getElementById('metaFilename').textContent = data.filename;

        // Event Info
        if (data.event) {
            document.getElementById('metaEventName').textContent = data.event.name || "Unnamed Event";
            document.getElementById('metaEventLocation').textContent = data.event.location_name || "Location unknown";
            document.getElementById('metaEventTime').textContent = new Date(data.event.start_time).toLocaleDateString();
        }

        // Faces
        const facesList = document.getElementById('metaFacesList');
        facesList.innerHTML = '';
        if (data.faces && data.faces.length > 0) {
            data.faces.forEach(face => {
                const faceDiv = document.createElement('div');
                faceDiv.className = 'face-card';
                faceDiv.innerHTML = `
                    <span class="face-name">${face.identity || 'Unknown Person'}</span>
                    ${face.has_glasses ? '<span class="face-tag">Wearing Glasses</span>' : ''}
                    ${!face.eyes_open ? '<span class="face-tag">Eyes Closed</span>' : '<span class="face-tag">Eyes Open</span>'}
                    <div class="meta-sub" style="font-size:0.7rem">Confidence: ${(face.recognition_confidence * 100).toFixed(0)}%</div>
                `;
                facesList.appendChild(faceDiv);
            });
        } else {
            facesList.innerHTML = '<p class="meta-sub">No faces detected in this shot.</p>';
        }

        // Technicals
        document.getElementById('metaBlurScore').textContent = data.blur_score.toFixed(0);
        document.getElementById('metaAestheticScore').textContent = data.aesthetic_score.toFixed(1);

    } catch (e) {
        console.error("Failed to load photo details", e);
    }
}

function closePhotoDetail() {
    const modal = document.getElementById('photoDetailModal');
    modal.classList.remove('active');
}

// Global listeners
document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
        closePhotoDetail();
        closeEditModal();
        // and close lists
        const list = document.getElementById('location-suggestions');
        if (list) list.remove();
        const options = document.getElementById('personFilterOptions');
        if (options) options.classList.remove('active');
    }
});

function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}
