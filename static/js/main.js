// main.js - Common functionality for Traffic Monitoring System

// API endpoints
const API_BASE_URL = '';
const API_ENDPOINTS = {
    SOURCES: `${API_BASE_URL}/api/sources`,
    VIOLATIONS: `${API_BASE_URL}/api/violations`,
    ACCIDENTS: `${API_BASE_URL}/api/accidents`,
    SYSTEM_STATUS: `${API_BASE_URL}/api/system/status`
};

// WebSocket endpoints
const WS_BASE_URL = window.location.protocol === 'https:' ? 'wss://' : 'ws://';
const WS_HOST = window.location.host;
const WS_ENDPOINTS = {
    VIOLATIONS: `${WS_BASE_URL}${WS_HOST}/ws/violations`,
    ACCIDENTS: `${WS_BASE_URL}${WS_HOST}/ws/accidents`,
    SYSTEM: `${WS_BASE_URL}${WS_HOST}/ws/system`,
    VIDEO: (sourceId) => `${WS_BASE_URL}${WS_HOST}/ws/video/${sourceId}`
};

// Global variables
let videoSources = [];
let systemSocket = null;

// Fetch all video sources
async function fetchVideoSources() {
    try {
        const response = await fetch(API_ENDPOINTS.SOURCES);
        const sources = await response.json();
        videoSources = sources;
        
        // Update UI elements that display sources
        updateVideoSourcesUI();
        
        return sources;
    } catch (error) {
        console.error('Error fetching video sources:', error);
        return [];
    }
}

// Update video sources UI
function updateVideoSourcesUI() {
    // Update active sources list on dashboard
    const activeSourcesList = document.getElementById('activeSourcesList');
    if (activeSourcesList) {
        if (videoSources.length === 0) {
            activeSourcesList.innerHTML = '<p class="text-center text-muted">No active sources</p>';
        } else {
            activeSourcesList.innerHTML = '';
            videoSources.forEach(source => {
                const statusClass = getStatusClass(source.status);
                activeSourcesList.innerHTML += `
                    <a href="config.html?source=${source.id}" class="list-group-item list-group-item-action">
                        <div class="d-flex justify-content-between align-items-center">
                            <div>
                                <strong>${source.name}</strong>
                                <br>
                                <small class="text-muted">${source.location}</small>
                            </div>
                            <span class="badge bg-${statusClass}">${source.status}</span>
                        </div>
                    </a>
                `;
            });
        }
    }
    
    // Update video sources list on sources page
    const videoSourcesList = document.getElementById('videoSourcesList');
    if (videoSourcesList) {
        if (videoSources.length === 0) {
            videoSourcesList.innerHTML = '<p class="loading-text">No video sources added</p>';
        } else {
            let tableHTML = `
                <table>
                    <thead>
                        <tr>
                            <th>Name</th>
                            <th>Location</th>
                            <th>Source</th>
                            <th>Type</th>
                            <th>Speed Limit</th>
                            <th>Status</th>
                            <th>Actions</th>
                        </tr>
                    </thead>
                    <tbody>
            `;

            videoSources.forEach(source => {
                const statusClass = source.status || 'stopped';
                const statusText = statusClass.charAt(0).toUpperCase() + statusClass.slice(1);

                tableHTML += `
                    <tr>
                        <td><strong>${source.name}</strong></td>
                        <td>${source.location}</td>
                        <td>${source.source}</td>
                        <td>${source.use_stream ? 'Stream' : 'File'}</td>
                        <td>${source.speed_limit} km/h</td>
                        <td>
                            <div class="source-status status-${statusClass}"></div>
                            <span class="status-text ${statusClass}">${statusText}</span>
                        </td>
                        <td>
                            <div class="action-buttons">
                                <button class="btn btn-primary" onclick="configureSource('${source.id}')">
                                    <i class="fas fa-cog"></i> Configure
                                </button>
                                ${source.status === 'running' ?
                                `<button class="btn btn-danger" onclick="stopSource('${source.id}')">
                                    <i class="fas fa-stop"></i> Stop
                                </button>` :
                                `<button class="btn btn-success" onclick="startSource('${source.id}')">
                                    <i class="fas fa-play"></i> Start
                                </button>`}
                                <button class="btn btn-danger" onclick="removeSource('${source.id}')">
                                    <i class="fas fa-trash"></i>
                                </button>
                            </div>
                        </td>
                    </tr>
                `;
            });

            tableHTML += `
                    </tbody>
                </table>
            `;

            videoSourcesList.innerHTML = tableHTML;
        }
    }
    
    // Update source selector on configuration page
    const sourceSelector = document.getElementById('sourceSelector');
    if (sourceSelector) {
        // Keep the first option (-- Select Video Source --)
        const firstOption = sourceSelector.options[0];
        sourceSelector.innerHTML = '';
        sourceSelector.appendChild(firstOption);
        
        // Add all sources as options
        videoSources.forEach(source => {
            const option = document.createElement('option');
            option.value = source.id;
            option.textContent = `${source.name} (${source.location})`;
            sourceSelector.appendChild(option);
        });
    }
}

// Add a new video source
async function addVideoSource(sourceData) {
    try {
        const response = await fetch(API_ENDPOINTS.SOURCES, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(sourceData)
        });
        
        if (response.ok) {
            // Close modal
            const modal = document.getElementById('addSourceModal');
            modal.classList.remove('active');

            // Reset form
            document.getElementById('addSourceForm').reset();

            // Refresh sources list
            await fetchVideoSources();

            // Show success alert
            alert('Video source added successfully!');
        } else {
            const error = await response.json();
            alert(`Failed to add source: ${error.detail}`);
        }
    } catch (error) {
        console.error('Error adding video source:', error);
        alert('Failed to add source. Check console for errors.');
    }
}

// Start a video source
async function startSource(sourceId) {
    try {
        const response = await fetch(`${API_ENDPOINTS.SOURCES}/${sourceId}/start`, {
            method: 'POST'
        });
        
        if (response.ok) {
            await fetchVideoSources();
        } else {
            const error = await response.json();
            alert(`Failed to start source: ${error.detail}`);
        }
    } catch (error) {
        console.error('Error starting video source:', error);
        alert('Failed to start source. Check console for errors.');
    }
}

// Stop a video source
async function stopSource(sourceId) {
    try {
        const response = await fetch(`${API_ENDPOINTS.SOURCES}/${sourceId}/stop`, {
            method: 'POST'
        });
        
        if (response.ok) {
            await fetchVideoSources();
        } else {
            const error = await response.json();
            alert(`Failed to stop source: ${error.detail}`);
        }
    } catch (error) {
        console.error('Error stopping video source:', error);
        alert('Failed to stop source. Check console for errors.');
    }
}

// Remove a video source
async function removeSource(sourceId) {
    if (confirm('Are you sure you want to remove this video source?')) {
        try {
            const response = await fetch(`${API_ENDPOINTS.SOURCES}/${sourceId}`, {
                method: 'DELETE'
            });
            
            if (response.ok) {
                await fetchVideoSources();
            } else {
                const error = await response.json();
                alert(`Failed to remove source: ${error.detail}`);
            }
        } catch (error) {
            console.error('Error removing video source:', error);
            alert('Failed to remove source. Check console for errors.');
        }
    }
}

// Navigate to configuration page for a source
function configureSource(sourceId) {
    window.location.href = `config.html?source=${sourceId}`;
}

// Connect to system WebSocket
function connectToSystemWebSocket() {
    const systemStatusDiv = document.getElementById('systemStatus');
    if (!systemStatusDiv) return;
    
    systemSocket = new WebSocket(WS_ENDPOINTS.SYSTEM);
    
    systemSocket.onopen = function(event) {
        console.log('Connected to system WebSocket');
    };
    
    systemSocket.onmessage = function(event) {
        const data = JSON.parse(event.data);
        if (data.event === 'status_update') {
            updateSystemStatus(data.data);
        }
    };
    
    systemSocket.onclose = function(event) {
        console.log('Disconnected from system WebSocket');
        // Try to reconnect after delay
        setTimeout(connectToSystemWebSocket, 5000);
    };
    
    systemSocket.onerror = function(error) {
        console.error('WebSocket error:', error);
    };
}

// Update system status display
function updateSystemStatus(status) {
    const systemStatusDiv = document.getElementById('systemStatus');
    if (!systemStatusDiv) return;
    
    const uptime = formatUptime(status.system_uptime);
    
    systemStatusDiv.innerHTML = `
        <div class="row">
            <div class="col-md-6">
                <p><strong>Active Sources:</strong> ${status.active_sources}</p>
                <p><strong>System Uptime:</strong> ${uptime}</p>
            </div>
            <div class="col-md-6">
                <p><strong>WebSocket Connections:</strong></p>
                <ul>
                    <li>Violations: ${status.active_connections.violations}</li>
                    <li>Accidents: ${status.active_connections.accidents}</li>
                    <li>System: ${status.active_connections.system}</li>
                </ul>
            </div>
        </div>
        <p class="text-muted text-end small">Last update: ${new Date(status.timestamp).toLocaleTimeString()}</p>
    `;
}

// Format uptime from seconds to days, hours, minutes, seconds
function formatUptime(seconds) {
    const days = Math.floor(seconds / 86400);
    seconds %= 86400;
    const hours = Math.floor(seconds / 3600);
    seconds %= 3600;
    const minutes = Math.floor(seconds / 60);
    seconds = Math.floor(seconds % 60);
    
    let result = '';
    if (days > 0) result += `${days}d `;
    if (hours > 0 || days > 0) result += `${hours}h `;
    if (minutes > 0 || hours > 0 || days > 0) result += `${minutes}m `;
    result += `${seconds}s`;
    
    return result;
}

// Get status class for Bootstrap badges
function getStatusClass(status) {
    if (status === 'running') return 'success';
    if (status === 'stopped') return 'danger';
    if (status.includes('error')) return 'warning';
    return 'secondary';
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    // Load video sources on all pages
    fetchVideoSources();

    // Connect to system WebSocket if on dashboard
    if (document.getElementById('systemStatus')) {
        connectToSystemWebSocket();
    }

    // Initialize area configuration if on config page
    if (typeof setupAreaConfigUI === 'function') {
        setupAreaConfigUI();
    }
});
