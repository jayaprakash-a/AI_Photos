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
    // Simple heuristic for now
    event.target.classList.add('active');

    if (sectionId === 'best') {
        loadBestPhotos();
    }

    checkBackendHealth();
}

// Backend Health
async function checkBackendHealth() {
    const el = document.getElementById('serverStatus');
    try {
        await fetch(`${API_BASE}/health`);
        el.innerHTML = '<span class="dot"></span> Online';
        el.classList.add('online');
    } catch (e) {
        el.innerHTML = '<span class="dot"></span> Backend Offline';
        el.classList.remove('online');
    }
}

// Ingestion
async function triggerIngest() {
    const dir = document.getElementById('dirInput').value;
    const statusEl = document.getElementById('ingestStatus');

    if (!dir) return;

    statusEl.textContent = "Starting ingestion... Check command prompt for logs.";

    try {
        const res = await fetch(`${API_BASE}/ingest?directory=${encodeURIComponent(dir)}`);
        if (!res.ok) throw new Error("Request failed");
        const data = await res.json();
        statusEl.textContent = `Success! Task ID: ${data.task_id}. Processing in background...`;
        statusEl.style.color = 'var(--success)';
    } catch (e) {
        statusEl.textContent = "Error triggering ingestion. Is backend running? Use run_backend.bat";
        statusEl.style.color = 'var(--danger)';
    }
}

// Best Photos
async function loadBestPhotos() {
    const grid = document.getElementById('bestGallery');
    grid.innerHTML = '<div class="loading-spinner">Loading...</div>';

    const useClustering = document.getElementById('clusterToggle').checked;

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

// Mock Stats (Real stats would need a new endpoint)
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

async function openEvent(eventId) {
    currentEventId = eventId; // Set current event ID
    document.getElementById('eventsGrid').style.display = 'none';
    const detailView = document.getElementById('eventDetailView');
    detailView.style.display = 'block';

    // Reset contents
    document.getElementById('detailTitle').textContent = "Loading...";
    document.getElementById('detailDesc').textContent = "";
    document.getElementById('detailGallery').innerHTML = '<div class="loading-spinner">Loading...</div>';

    try {
        const res = await fetch(`${API_BASE}/events/${eventId}`);
        const data = await res.json();
        const event = data.event;
        const photos = data.photos;

        window.currentEventData = event; // Store for editing

        document.getElementById('detailTitle').textContent = event.name;
        document.getElementById('detailDesc').textContent = `${event.description || ''} • ${event.location_name || ''}`;

        const gallery = document.getElementById('detailGallery');
        gallery.innerHTML = '';

        photos.forEach(photo => {
            const div = document.createElement('div');
            div.className = 'photo-item';
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
    }
}

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
