// area-config.js - Area configuration functionality for Traffic Monitoring System

// Global variables
let canvas = null;
let fabricCanvas = null;
let selectedSource = null;
let currentAreaType = 'detection';
let areaColors = {
    'detection': { color: 'rgb(0, 255, 0)', fillColor: 'rgba(0, 255, 0, 0.2)' },
    'speed': { color: 'rgb(255, 0, 0)', fillColor: 'rgba(255, 0, 0, 0.2)' },
    'wrong_direction': { color: 'rgb(0, 0, 255)', fillColor: 'rgba(0, 0, 255, 0.2)' },
    'parking': { color: 'rgb(255, 255, 0)', fillColor: 'rgba(255, 255, 0, 0.2)' },
    'traffic_line': { color: 'rgb(255, 0, 255)', fillColor: 'rgba(255, 0, 255, 0.2)' },
    'traffic_sign': { color: 'rgb(0, 255, 255)', fillColor: 'rgba(0, 255, 255, 0.2)' },
    'custom': { color: 'rgb(128, 128, 128)', fillColor: 'rgba(128, 128, 128, 0.2)' }
};
let videoElement = null;
let videoStream = null;
let isDrawing = false;
let currentShape = null;
let linePoints = [];
let polygonPoints = [];
let existingAreas = {};
let currentFrameWidth = 0;
let currentFrameHeight = 0;
const DEFAULT_FRAME_WIDTH = 1920; // Default width, adjust as needed
const DEFAULT_FRAME_HEIGHT = 1080; // Default height, adjust as needed
let toastTimeout = null;
let localVideoSources = []; // Store loaded video sources locally

// Setup the area configuration UI
async function setupAreaConfigUI() {
    // Set up canvas
    canvas = document.getElementById('videoCanvas');

    // Initialize fabric canvas
    fabricCanvas = new fabric.Canvas('videoCanvas', {
        selection: false,
        preserveObjectStacking: true
    });

    // Force canvas to initialize with appropriate dimensions
    fabricCanvas.setWidth(1280);
    fabricCanvas.setHeight(720);

    // Add resize listener for when window is resized
    window.addEventListener('resize', resizeCanvas);

    // Try to resize canvas after a short delay to ensure DOM is ready
    setTimeout(() => {
        resizeCanvas();
    }, 100);

    // Setup event listeners
    setupEventListeners();

    // Initialize model buttons
    initializeModelButtons();

    // Load video sources first
    await loadVideoSources();

    // Get query parameters and handle source selection
    const urlParams = new URLSearchParams(window.location.search);
    const sourceId = urlParams.get('source');

    // If source ID is provided in URL, select it
    if (sourceId) {
        const sourceSelector = document.getElementById('sourceSelector');
        if (sourceSelector) {
            sourceSelector.value = sourceId;
            // Trigger the change event to load the source
            onSourceChange({ target: { value: sourceId } });
        }
    }

    // Create toast container for notifications if it doesn't exist
    if (!document.getElementById('toast-container')) {
        const toastContainer = document.createElement('div');
        toastContainer.id = 'toast-container';
        toastContainer.className = 'position-fixed bottom-0 end-0 p-3';
        toastContainer.style.zIndex = 1050;
        document.body.appendChild(toastContainer);
    }
}

// Show toast notification
function showToast(message, type = 'info') {
    // Clear any existing toast
    if (toastTimeout) {
        clearTimeout(toastTimeout);
    }
    
    // Remove existing toast
    const existingToast = document.querySelector('.toast');
    if (existingToast) {
        existingToast.remove();
    }
    
    // Create toast
    const toastContainer = document.getElementById('toast-container');
    
    // Set toast class based on type
    let bgClass = 'bg-info';
    if (type === 'success') bgClass = 'bg-success';
    if (type === 'error') bgClass = 'bg-danger';
    if (type === 'warning') bgClass = 'bg-warning';
    
    const toastHtml = `
        <div class="toast show" role="alert" aria-live="assertive" aria-atomic="true">
            <div class="toast-header ${bgClass} text-white">
                <strong class="me-auto">Notification</strong>
                <button type="button" class="btn-close btn-close-white" data-bs-dismiss="toast" aria-label="Close"></button>
            </div>
            <div class="toast-body">
                ${message}
            </div>
        </div>
    `;
    
    toastContainer.innerHTML = toastHtml;
    
    // Add close functionality
    const closeBtn = toastContainer.querySelector('.btn-close');
    if (closeBtn) {
        closeBtn.addEventListener('click', () => {
            const toast = document.querySelector('.toast');
            if (toast) toast.remove();
        });
    }
    
    // Auto-hide after 3 seconds
    toastTimeout = setTimeout(() => {
        const toast = document.querySelector('.toast');
        if (toast) toast.remove();
    }, 3000);
}

// Load video sources for selector
async function loadVideoSources() {
    try {
        const response = await fetch(API_ENDPOINTS.SOURCES);
        const sources = await response.json();
        localVideoSources = sources;
        
        // Populate source selector
        const sourceSelector = document.getElementById('sourceSelector');
        sourceSelector.innerHTML = '<option value="">-- Select Video Source --</option>';
        
        sources.forEach(source => {
            const option = document.createElement('option');
            option.value = source.id;
            option.textContent = `${source.name} (${source.location})`;
            sourceSelector.appendChild(option);
        });
        
        return sources;
    } catch (error) {
        console.error('Error loading video sources:', error);
        return [];
    }
}

// Setup event listeners
function setupEventListeners() {
    // Source selector
    document.getElementById('sourceSelector').addEventListener('change', onSourceChange);
    
    // Area type selector
    document.getElementById('areaTypeSelector').addEventListener('change', onAreaTypeChange);
    
    // Drawing controls
    document.getElementById('btnStartPolygon').addEventListener('click', startDrawingPolygon);
    document.getElementById('btnStartLine').addEventListener('click', startDrawingLine);
    document.getElementById('btnClear').addEventListener('click', clearAreas);
    document.getElementById('btnClearAll').addEventListener('click', clearAllAreas);
    document.getElementById('btnSaveConfig').addEventListener('click', saveAreaConfiguration);
    document.getElementById('btnToggleControls').addEventListener('click', toggleControls);
    
    // Model toggle controls
    document.getElementById('btnApplySettings').addEventListener('click', saveModelSettings);

    // Model status buttons
    document.getElementById('btnAccidentModel').addEventListener('click', toggleAccidentModel);
    document.getElementById('btnHelmetModel').addEventListener('click', toggleHelmetModel);
    
    // Initialize fabric.js event handlers
    initializeFabricEvents();
}

// Initialize fabric.js event handlers
function initializeFabricEvents() {
    // Mouse down event
    fabricCanvas.on('mouse:down', function(options) {
        if (!isDrawing) return;
        
        const pointer = fabricCanvas.getPointer(options.e);
        
        if (currentShape === 'polygon') {
            // First point of polygon
            if (polygonPoints.length === 0) {
                polygonPoints.push({ x: pointer.x, y: pointer.y });
                
                // Create circle for first point
                const circle = new fabric.Circle({
                    left: pointer.x - 5,
                    top: pointer.y - 5,
                    radius: 5,
                    fill: areaColors[currentAreaType].color,
                    stroke: '#fff',
                    strokeWidth: 1,
                    originX: 'center',
                    originY: 'center',
                    selectable: false,
                    hasBorders: false,
                    hasControls: false
                });
                
                fabricCanvas.add(circle);
                fabricCanvas.renderAll();
            } 
            // Subsequent points
            else {
                polygonPoints.push({ x: pointer.x, y: pointer.y });
                
                // Create circle for this point
                const circle = new fabric.Circle({
                    left: pointer.x - 5,
                    top: pointer.y - 5,
                    radius: 5,
                    fill: areaColors[currentAreaType].color,
                    stroke: '#fff',
                    strokeWidth: 1,
                    originX: 'center',
                    originY: 'center',
                    selectable: false,
                    hasBorders: false,
                    hasControls: false
                });
                
                fabricCanvas.add(circle);
                
                // Draw line from previous point
                if (polygonPoints.length > 1) {
                    const prevPoint = polygonPoints[polygonPoints.length - 2];
                    const line = new fabric.Line(
                        [prevPoint.x, prevPoint.y, pointer.x, pointer.y],
                        {
                            stroke: areaColors[currentAreaType].color,
                            strokeWidth: 2,
                            selectable: false,
                            hasBorders: false,
                            hasControls: false,
                            evented: false
                        }
                    );
                    
                    fabricCanvas.add(line);
                }
                
                fabricCanvas.renderAll();
            }
        } 
        // Line drawing
        else if (currentShape === 'line') {
            if (linePoints.length === 0) {
                linePoints.push({ x: pointer.x, y: pointer.y });
                
                // Create circle for first point
                const circle = new fabric.Circle({
                    left: pointer.x - 5,
                    top: pointer.y - 5,
                    radius: 5,
                    fill: areaColors[currentAreaType].color,
                    stroke: '#fff',
                    strokeWidth: 1,
                    originX: 'center',
                    originY: 'center',
                    selectable: false,
                    hasBorders: false,
                    hasControls: false
                });
                
                fabricCanvas.add(circle);
                fabricCanvas.renderAll();
            } 
            else if (linePoints.length === 1) {
                linePoints.push({ x: pointer.x, y: pointer.y });
                
                // Create circle for second point
                const circle = new fabric.Circle({
                    left: pointer.x - 5,
                    top: pointer.y - 5,
                    radius: 5,
                    fill: areaColors[currentAreaType].color,
                    stroke: '#fff',
                    strokeWidth: 1,
                    originX: 'center',
                    originY: 'center',
                    selectable: false,
                    hasBorders: false,
                    hasControls: false
                });
                
                fabricCanvas.add(circle);
                
                // Draw line connecting the two points
                const prevPoint = linePoints[0];
                const line = new fabric.Line(
                    [prevPoint.x, prevPoint.y, pointer.x, pointer.y],
                    {
                        stroke: areaColors[currentAreaType].color,
                        strokeWidth: 2,
                        selectable: false,
                        hasBorders: false,
                        hasControls: false,
                        evented: false
                    }
                );
                
                fabricCanvas.add(line);
                fabricCanvas.renderAll();
                
                // Save the line and stop drawing
                saveShape();
            }
        }
    });
    
    // Double click to complete polygon
    fabricCanvas.on('mouse:dblclick', function() {
        if (isDrawing && currentShape === 'polygon' && polygonPoints.length >= 3) {
            saveShape();
        }
    });
    
    // Mouse move to show preview
    fabricCanvas.on('mouse:move', function(options) {
        if (!isDrawing) return;
        
        const pointer = fabricCanvas.getPointer(options.e);
        
        if (currentShape === 'polygon' && polygonPoints.length > 0) {
            // Remove the preview line if it exists
            fabricCanvas.getObjects().forEach(obj => {
                if (obj.isPreviewLine) {
                    fabricCanvas.remove(obj);
                }
            });
            
            // Draw preview line
            const lastPoint = polygonPoints[polygonPoints.length - 1];
            const previewLine = new fabric.Line(
                [lastPoint.x, lastPoint.y, pointer.x, pointer.y],
                {
                    stroke: areaColors[currentAreaType].color,
                    strokeWidth: 2,
                    strokeDashArray: [5, 5],
                    selectable: false,
                    hasBorders: false,
                    hasControls: false,
                    evented: false,
                    isPreviewLine: true
                }
            );
            
            fabricCanvas.add(previewLine);
            fabricCanvas.renderAll();
        }
    });
}

// Resize canvas to match container while maintaining video aspect ratio
function resizeCanvas() {
    // Get the container dimensions - use class selector since HTML uses class, not ID
    const canvasContainer = document.querySelector('.canvas-container');
    if (!canvasContainer) {
        console.warn('Canvas container not found, using default dimensions');
        // Use default dimensions if container not found
        const canvasWidth = 1280;
        const canvasHeight = 720;
        fabricCanvas.setWidth(canvasWidth);
        fabricCanvas.setHeight(canvasHeight);
        return;
    }

    const canvasWidth = canvasContainer.offsetWidth;
    const canvasHeight = canvasContainer.offsetHeight;
    
    // Calculate background image dimensions while maintaining aspect ratio
    // Use current frame dimensions or defaults if window.origVideo* is not set
    const videoWidth = window.origVideoWidth || currentFrameWidth || DEFAULT_FRAME_WIDTH;
    const videoHeight = window.origVideoHeight || currentFrameHeight || DEFAULT_FRAME_HEIGHT;
    const videoAspectRatio = videoWidth / videoHeight;
    let bgWidth = canvasWidth;
    let bgHeight = canvasWidth / videoAspectRatio;
    
    // Adjust if height exceeds container
    if (bgHeight > canvasHeight) {
        bgHeight = canvasHeight;
        bgWidth = canvasHeight * videoAspectRatio;
    }
    
    // Calculate scaling and offset
    const scaleX = bgWidth / videoWidth;
    const scaleY = bgHeight / videoHeight;
    const left = (canvasWidth - bgWidth) / 2;
    const top = (canvasHeight - bgHeight) / 2;
    
    // Store the background image info for coordinate conversions
    window.backgroundImageInfo = {
        scaleX: scaleX,
        scaleY: scaleY,
        left: left,
        top: top
    };
    console.log('resizeCanvas → backgroundImageInfo:',
        window.backgroundImageInfo,
        'origVideo:', {w: videoWidth, h: videoHeight},
        'scaled:', {w: bgWidth, h: bgHeight}
    );
    
    // Update canvas dimensions
    fabricCanvas.setWidth(canvasWidth);
    fabricCanvas.setHeight(canvasHeight);
    
    // Update background image position and scale
    if (fabricCanvas.backgroundImage) {
        fabricCanvas.backgroundImage.set({
            scaleX: scaleX,
            scaleY: scaleY,
            left: left,
            top: top
        });
    }
    
    // Render the canvas
    fabricCanvas.renderAll();
    
    // Redraw areas if they exist
    if (window.loadedAreas) {
        redrawAreas();
    }
}

// Handle source selection change
async function onSourceChange(e) {
    const sourceId = e.target.value;
    
    // Close any existing WebSocket connection
    if (videoStream) {
        videoStream.close();
        videoStream = null;
    }
    
    if (!sourceId) {
        selectedSource = null;
        clearCanvas();
        // Reset frame dimensions when no source is selected
        currentFrameWidth = 0;
        currentFrameHeight = 0;
        // Only resize if canvas container is available
        if (document.querySelector('.canvas-container')) {
            resizeCanvas(); // Resize to default or empty state
        }
        return;
    }
    
    selectedSource = sourceId;
    
    // Find the source details
    const source = localVideoSources.find(s => s.id === sourceId);
    
    if (!source) {
        console.error('Source not found:', sourceId);
        return;
    }
    
    // Clear canvas and reset frame dimensions before loading new source
    clearCanvas(); // This also stops drawing and clears existingAreas
    currentFrameWidth = 0; // Reset before loading new video
    currentFrameHeight = 0;
    // resizeCanvas(); // Initial resize before video metadata is known, will use defaults

    // The order of operations will be: loadVideo (gets dimensions, then calls resizeCanvas),
    // then loadAreaConfiguration (which uses the new canvas size from resizeCanvas),
    // then loadModelSettings.
    try {
        await loadVideo(source); // loadVideo will call resizeCanvas after getting frame dimensions

        // Always attempt to load area configuration, even if it might be empty
        console.log('Loading area configuration for selected source...');
        await loadAreaConfiguration(sourceId);

        // Load model settings
        await loadModelSettings(sourceId);

        showToast(`Video source "${source.name}" loaded successfully`, 'success');
    } catch (error) {
        console.error('Error loading video source:', error);
        showToast('Failed to load video source. Please try again.', 'error');
    }
}

// Load video stream or file
async function loadVideo(source) {
    // Flag to track if this is a restart
    const isRestart = selectedSource === source.id && videoStream !== null;
    
    // Stop any existing stream
    if (videoStream) {
        videoStream.close();
        videoStream = null;
    }

    try {
        // First, try to get a frame to determine dimensions
        const response = await fetch(`${API_ENDPOINTS.SOURCES}/${source.id}/frame`);
        if (!response.ok) {
            throw new Error(`Failed to get frame: ${response.status}`);
        }

        const frameData = await response.json();
        currentFrameWidth = parseInt(frameData.width, 10) || DEFAULT_FRAME_WIDTH;
        currentFrameHeight = parseInt(frameData.height, 10) || DEFAULT_FRAME_HEIGHT;

        console.log(`Video dimensions: ${currentFrameWidth}x${currentFrameHeight}, isRestart: ${isRestart}`);

        // Update canvas size based on video aspect ratio
        // Only resize if canvas container is available
        if (document.querySelector('.canvas-container')) {
            resizeCanvas(); // Call resizeCanvas AFTER frame dimensions are known
        }

        // Create initial image for background
        const img = new Image();
        img.onload = function() {
            const imgWidth = img.width || currentFrameWidth;
            const imgHeight = img.height || currentFrameHeight;
            const imgAspectRatio = imgWidth / imgHeight;
            const canvasAspectRatio = fabricCanvas.width / fabricCanvas.height;
            
            let scaleX, scaleY, left = 0, top = 0;
            
            // Scale to completely fill the canvas while maintaining aspect ratio
            if (canvasAspectRatio > imgAspectRatio) {
                // Canvas is wider than image
                scaleX = fabricCanvas.width / imgWidth;
                scaleY = scaleX; // Keep aspect ratio
                // Center vertically
                top = (fabricCanvas.height - (imgHeight * scaleY)) / 2;
            } else {
                // Canvas is taller than image
                scaleY = fabricCanvas.height / imgHeight;
                scaleX = scaleY; // Keep aspect ratio
                // Center horizontally
                left = (fabricCanvas.width - (imgWidth * scaleX)) / 2;
            }
            
            console.log(`Initial background image: size=${imgWidth}x${imgHeight}, scale=${scaleX.toFixed(2)}, position=${left.toFixed(0)},${top.toFixed(0)}`);
            
            const fabricImage = new fabric.Image(img, {
                left: left,
                top: top,
                originX: 'left',
                originY: 'top',
                scaleX: scaleX,
                scaleY: scaleY
            });
            
            // Store original dimensions and scaling for coordinate transformations
            window.backgroundImageInfo = {
                width: imgWidth,
                height: imgHeight,
                scaleX: scaleX,
                scaleY: scaleY,
                left: left,
                top: top
            };
            
            fabricCanvas.setBackgroundImage(fabricImage, fabricCanvas.renderAll.bind(fabricCanvas));
        };
        img.src = "data:image/jpeg;base64," + frameData.frame_data;

        // Connect to WebSocket for streaming
        const wsUrl = `${window.location.protocol === 'https:' ? 'wss://' : 'ws://'}${window.location.host}/ws/video/${source.id}`;
        videoStream = new WebSocket(wsUrl);

        videoStream.onopen = function() {
            console.log(`Connected to video stream: ${source.id}`);
        };

        videoStream.onmessage = function(event) {
            const data = JSON.parse(event.data);

            if (data.event === 'frame') {
                const newFrameWidth = parseInt(data.width, 10);
                const newFrameHeight = parseInt(data.height, 10);

                let dimensionsChanged = false;
                if (newFrameWidth && newFrameHeight && 
                    (newFrameWidth !== currentFrameWidth || newFrameHeight !== currentFrameHeight)) {
                    currentFrameWidth = newFrameWidth;
                    currentFrameHeight = newFrameHeight;
                    dimensionsChanged = true;
                    console.log(`Video dimensions changed to: ${newFrameWidth}x${newFrameHeight}`);
                }

                if (dimensionsChanged) {
                    // Only resize if canvas container is available
                    if (document.querySelector('.canvas-container')) {
                        resizeCanvas(); // This will also rescale the background if it exists
                    }
                    // If dimensions changed, reload areas to ensure they're properly scaled
                    if (Object.keys(existingAreas).length > 0) {
                        redrawAreas();
                    } else {
                        // If no areas are loaded yet, try loading them from the API
                        loadAreaConfiguration(selectedSource);
                    }
                }                // Update the background image
                const img = new Image();
                img.onload = function() {
                    const imgWidth = img.width || currentFrameWidth;
                    const imgHeight = img.height || currentFrameHeight;
                    const imgAspectRatio = imgWidth / imgHeight;
                    const canvasAspectRatio = fabricCanvas.width / fabricCanvas.height;
                    
                    let scaleX, scaleY, left = 0, top = 0;
                    
                    // Scale to completely fill the canvas while maintaining aspect ratio
                    if (canvasAspectRatio > imgAspectRatio) {
                        // Canvas is wider than image
                        scaleX = fabricCanvas.width / imgWidth;

                        // Scale to completely fill the canvas while maintaining aspect ratio
                        scaleY = scaleX; // Keep aspect ratio
                        // Center vertically
                        top = (fabricCanvas.height - (imgHeight * scaleY)) / 2;
                    } else {
                        // Canvas is taller than image
                        scaleY = fabricCanvas.height / imgHeight;

                        // Scale to completely fill the canvas while maintaining aspect ratio

                        
                        scaleX = scaleY; // Keep aspect ratio
                        // Center horizontally
                        left = (fabricCanvas.width - (imgWidth * scaleX)) / 2;
                    }
                    
                    // Store original dimensions and scaling for coordinate transformations
                    window.backgroundImageInfo = {
                        width: imgWidth,
                        height: imgHeight,
                        scaleX: scaleX,
                        scaleY: scaleY,
                        left: left,
                        top: top
                    };
                    
                    const fabricImage = new fabric.Image(img, {
                        left: left,
                        top: top,
                        originX: 'left',
                        originY: 'top',
                        scaleX: scaleX,
                        scaleY: scaleY
                    });
                    
                    fabricCanvas.setBackgroundImage(fabricImage, fabricCanvas.renderAll.bind(fabricCanvas));
                };
                img.src = "data:image/jpeg;base64," + data.frame_data;
            } else if (data.event === 'error') {
                console.error('Video stream error:', data.message);
            }
        };

        videoStream.onclose = function() {
            console.log('Video stream disconnected');

            // Set a timer to attempt reconnection and reload configuration if needed
            if (selectedSource) {
                setTimeout(async () => {
                    try {
                        console.log(`Attempting to reconnect to video source: ${selectedSource}`);
                        const source = localVideoSources.find(s => s.id === selectedSource);
                        if (source) {
                            // Reconnect to the video source
                            await loadVideo(source);

                            // Always reload area configuration after reconnection
                            // This ensures areas are loaded even if the video processor was restarted
                            console.log('Reloading area configuration after reconnection...');
                            await loadAreaConfiguration(selectedSource);

                            // Also reload model settings
                            await loadModelSettings(selectedSource);
                        }
                    } catch (error) {
                        console.error('Error reconnecting to video source:', error);
                        showToast('Failed to reconnect to video source. Please refresh the page.', 'error');
                    }
                }, 2000); // Wait 2 seconds before attempting reconnection
            }
        };

        videoStream.onerror = function(error) {
            console.error('Video stream WebSocket error:', error);
        };

    } catch (error) {
        console.error('Error loading video:', error);
        // Use default dimensions if video load fails before getting dimensions
        if (!currentFrameWidth || !currentFrameHeight) {
             currentFrameWidth = DEFAULT_FRAME_WIDTH;
             currentFrameHeight = DEFAULT_FRAME_HEIGHT;
        }
        // Only resize if canvas container is available
        if (document.querySelector('.canvas-container')) {
            resizeCanvas(); // Ensure canvas is sized even on error
        }
        showToast('Failed to load video. Check console for errors.', 'error');
    }
}

// Load existing area configuration
async function loadAreaConfiguration(sourceId) {
    try {
        console.log(`Loading area configuration for source: ${sourceId}`);

        const response = await fetch(`${API_ENDPOINTS.SOURCES}/${sourceId}/areas`);
        if (!response.ok) {
            if (response.status === 404) {
                console.log('No area configuration found for this source');
                existingAreas = {};
                clearCanvas();
                return {};
            }
            throw new Error(`Failed to load area configuration: ${response.status}`);
        }

        const config = await response.json();

        if (config && config.areas && Object.keys(config.areas).length > 0) {
            console.log(`Area configuration loaded successfully. Areas found: ${Object.keys(config.areas).join(', ')}`);
            existingAreas = config.areas;

            // Always redraw areas after loading configuration
            redrawAreas();

            // Show success message
            showToast(`Loaded ${Object.keys(config.areas).length} area type(s)`, 'info');

            // Return the loaded configuration for chaining
            return config.areas;
        } else {
            console.log('No area configuration found or empty configuration');
            existingAreas = {};
            clearCanvas();
            showToast('No area configuration found. Ready to define new areas.', 'info');
            return {};
        }
    } catch (error) {
        console.error('Error loading area configuration:', error);
        showToast('Failed to load area configuration. Will use any locally cached areas if available.', 'warning');
        return null; // Return null to indicate error
    }
}

// Redraw existing areas on canvas
function redrawAreas() {
    // Clear canvas objects except background
    const backgroundImage = fabricCanvas.backgroundImage;
    fabricCanvas.clear(); // Clears all objects, including points and lines drawn during shape creation
    if (backgroundImage) {
        fabricCanvas.setBackgroundImage(backgroundImage, fabricCanvas.renderAll.bind(fabricCanvas));
    }

    // Ensure currentFrameWidth and currentFrameHeight are valid, otherwise use defaults
    const frameW = currentFrameWidth > 0 ? currentFrameWidth : DEFAULT_FRAME_WIDTH;
    const frameH = currentFrameHeight > 0 ? currentFrameHeight : DEFAULT_FRAME_HEIGHT;
    
    console.log(`Redrawing areas with frame dimensions: ${frameW}x${frameH}, canvas: ${fabricCanvas.width}x${fabricCanvas.height}`);

    // Draw each area by type
    for (const [areaType, areas] of Object.entries(existingAreas)) {
        areas.forEach(area => {
            if (area.points) {
                // Convert saved video coordinates to canvas coordinates for drawing
                const canvasPoints = area.points.map(p => {
                    // Improved and more accurate scaling calculation
                    const scaledX = (p[0] / frameW) * fabricCanvas.width;
                    const scaledY = (p[1] / frameH) * fabricCanvas.height;
                    console.log(`Converting video point (${p[0]}, ${p[1]}) to canvas point (${scaledX}, ${scaledY})`);
                    return {
                        x: scaledX,
                        y: scaledY
                    };
                });

                if (canvasPoints.length === 2) {
                    drawExistingLine(canvasPoints, areaType);
                } else if (canvasPoints.length >= 3) {
                    drawExistingPolygon(canvasPoints, areaType);
                }
            }
        });
    }
    fabricCanvas.renderAll(); // Ensure canvas is rendered after all updates
}

// Draw existing polygon on canvas (expects points in canvas coordinates)
function drawExistingPolygon(canvasPoints, areaType) {
    const areaColor = areaColors[areaType] || areaColors.custom;

    const polygon = new fabric.Polygon(canvasPoints, {
        stroke: areaColor.color,
        strokeWidth: 2,
        fill: areaColor.fillColor,
        selectable: false,
        hasBorders: false,
        hasControls: false,
        evented: false,
        objectType: 'area',
        areaType: areaType
    });
    fabricCanvas.add(polygon);

    // Add points as circles
    canvasPoints.forEach(point => {
        const circle = new fabric.Circle({
            left: point.x - 5,
            top: point.y - 5,
            radius: 5,
            fill: areaColor.color,
            stroke: '#fff',
            strokeWidth: 1,
            originX: 'center',
            originY: 'center',
            selectable: false,
            hasBorders: false,
            hasControls: false
        });
        fabricCanvas.add(circle);
    });
    // fabricCanvas.renderAll(); // Moved to end of redrawAreas for efficiency
}

// Draw existing line on canvas (expects points in canvas coordinates)
function drawExistingLine(canvasPoints, areaType) {
    const areaColor = areaColors[areaType] || areaColors.custom;

    const line = new fabric.Line(
        [canvasPoints[0].x, canvasPoints[0].y, canvasPoints[1].x, canvasPoints[1].y],
        {
            stroke: areaColor.color,
            strokeWidth: 2,
            selectable: false,
            hasBorders: false,
            hasControls: false,
            evented: false,
            objectType: 'area',
            areaType: areaType
        }
    );
    fabricCanvas.add(line);

    // Add points as circles
    canvasPoints.forEach(point => {
        const circle = new fabric.Circle({
            left: point.x - 5,
            top: point.y - 5,
            radius: 5,
            fill: areaColor.color,
            stroke: '#fff',
            strokeWidth: 1,
            originX: 'center',
            originY: 'center',
            selectable: false,
            hasBorders: false,
            hasControls: false
        });
        fabricCanvas.add(circle);
    });
    // fabricCanvas.renderAll(); // Moved to end of redrawAreas for efficiency
}

// Handle area type change
function onAreaTypeChange(e) {
    currentAreaType = e.target.value;
}

// Start drawing a polygon
function startDrawingPolygon() {
    if (!selectedSource) {
        alert('Please select a video source first.');
        return;
    }
    
    // Stop any existing drawing
    stopDrawing();
    
    // Start new drawing
    isDrawing = true;
    currentShape = 'polygon';
    polygonPoints = [];
    
    alert('Click on the video to add points to your polygon. Double-click to complete.');
}

// Start drawing a line
function startDrawingLine() {
    if (!selectedSource) {
        alert('Please select a video source first.');
        return;
    }
    
    // Check if selected area type supports lines
    const lineAreaTypes = ['speed', 'wrong_direction', 'traffic_line'];
    if (!lineAreaTypes.includes(currentAreaType)) {
        alert(`Line drawing is only supported for these area types: ${lineAreaTypes.join(', ')}`);
        return;
    }
    
    // Stop any existing drawing
    stopDrawing();
    
    // Start new drawing
    isDrawing = true;
    currentShape = 'line';
    linePoints = [];
    
    alert('Click on the video to set the start point, then click again to set the end point.');
}

// Stop drawing
function stopDrawing() {
    isDrawing = false;
    currentShape = null;
    
    // Remove any temporary lines
    fabricCanvas.getObjects().forEach(obj => {
        if (obj.isPreviewLine) {
            fabricCanvas.remove(obj);
        }
    });
    
    fabricCanvas.renderAll();
}

// Save the current shape (polygon or line)
function saveShape() {
    if (!selectedSource) return;

    let newArea = null;

    // Add the shape to the existing areas
    if (!existingAreas[currentAreaType]) {
        existingAreas[currentAreaType] = [];
    }

    // Ensure currentFrameWidth and currentFrameHeight are valid, otherwise use defaults
    const frameW = currentFrameWidth > 0 ? currentFrameWidth : DEFAULT_FRAME_WIDTH;
    const frameH = currentFrameHeight > 0 ? currentFrameHeight : DEFAULT_FRAME_HEIGHT;
    
    // Log the exact dimensions being used for scaling
    console.log(`Saving shape with frame dimensions: ${frameW}x${frameH}, canvas: ${fabricCanvas.width}x${fabricCanvas.height}`);    if (currentShape === 'polygon' && polygonPoints.length >= 3) {
        // Convert from canvas coordinates to original video coordinates
        const apiPoints = polygonPoints.map(point => {
            // Get background image scaling info
            const bgInfo = window.backgroundImageInfo || {
                width: frameW,
                height: frameH,
                scaleX: fabricCanvas.width / frameW,
                scaleY: fabricCanvas.height / frameH,
                left: 0,
                top: 0
            };
            
            // Adjust coordinates based on background image position and scale
            // First, account for any offset due to centering
            const adjustedX = point.x - bgInfo.left;
            const adjustedY = point.y - bgInfo.top;
            
            // Then convert from canvas scale to video coordinates
            const x = Math.round(adjustedX / bgInfo.scaleX);
            const y = Math.round(adjustedY / bgInfo.scaleY);
            
            console.log(`Converting canvas point (${point.x}, ${point.y}) → adjusted (${adjustedX}, ${adjustedY}) → video (${x}, ${y})`);
            return [x, y];
        });

        newArea = { points: apiPoints };
        existingAreas[currentAreaType].push(newArea);

        // Complete the polygon shape for display on canvas (already in canvas coordinates)
        const fabricPoints = polygonPoints.map(point => ({ x: point.x, y: point.y }));
        const polygon = new fabric.Polygon(fabricPoints, {
            stroke: areaColors[currentAreaType].color,
            strokeWidth: 2,
            fill: areaColors[currentAreaType].fillColor,
            selectable: false,
            hasBorders: false,
            hasControls: false,
            evented: false,
            objectType: 'area',
            areaType: currentAreaType
        });
        fabricCanvas.add(polygon);
        // Add circles for points
        fabricPoints.forEach(point => {
            const circle = new fabric.Circle({
                left: point.x - 5,
                top: point.y - 5,
                radius: 5,
                fill: areaColors[currentAreaType].color,
                stroke: '#fff',
                strokeWidth: 1,
                originX: 'center',
                originY: 'center',
                selectable: false,
                hasBorders: false,
                hasControls: false
            });
            fabricCanvas.add(circle);
        });    } else if (currentShape === 'line' && linePoints.length === 2) {
        // Convert from canvas coordinates to original video coordinates
        const apiPoints = linePoints.map(point => {
            // Get background image scaling info
            const bgInfo = window.backgroundImageInfo || {
                width: frameW,
                height: frameH,
                scaleX: fabricCanvas.width / frameW,
                scaleY: fabricCanvas.height / frameH,
                left: 0,
                top: 0
            };
            
            // Adjust coordinates based on background image position and scale
            // First, account for any offset due to centering
            const adjustedX = point.x - bgInfo.left;
            const adjustedY = point.y - bgInfo.top;
            
            // Then convert from canvas scale to video coordinates
            const x = Math.round(adjustedX / bgInfo.scaleX);
            const y = Math.round(adjustedY / bgInfo.scaleY);
            
            console.log(`Converting canvas point (${point.x}, ${point.y}) → adjusted (${adjustedX}, ${adjustedY}) → video (${x}, ${y})`);
            return [x, y];
        });

        newArea = { points: apiPoints };
        existingAreas[currentAreaType].push(newArea);

        // Draw the line for display on canvas (already in canvas coordinates)
        const fabricLinePoints = [
            linePoints[0].x, linePoints[0].y,
            linePoints[1].x, linePoints[1].y
        ];
        const line = new fabric.Line(fabricLinePoints, {
            stroke: areaColors[currentAreaType].color,
            strokeWidth: 2,
            selectable: false,
            hasBorders: false,
            hasControls: false,
            evented: false,
            objectType: 'area',
            areaType: currentAreaType
        });
        fabricCanvas.add(line);
        // Add circles for points
        linePoints.forEach(point => {
            const circle = new fabric.Circle({
                left: point.x - 5,
                top: point.y - 5,
                radius: 5,
                fill: areaColors[currentAreaType].color,
                stroke: '#fff',
                strokeWidth: 1,
                originX: 'center',
                originY: 'center',
                selectable: false,
                hasBorders: false,
                hasControls: false
            });
            fabricCanvas.add(circle);
        });
    }

    // Reset drawing state
    stopDrawing();
    linePoints = [];
    polygonPoints = [];

    // Redraw all areas for consistency if a new shape was added
    if (newArea) {
        redrawAreas();
    } else {
        fabricCanvas.renderAll(); // just render if no new area was fully formed
    }
}

// Clear all areas of current type
function clearAreas() {
    if (!selectedSource) {
        alert('Please select a video source first.');
        return;
    }

    if (confirm(`Are you sure you want to clear all ${currentAreaType} areas?`)) {
        if (existingAreas[currentAreaType]) {
            existingAreas[currentAreaType] = [];
        }

        redrawAreas();
    }
}

// Clear all area configurations and restart (equivalent to 'd' key functionality)
async function clearAllAreas() {
    if (!selectedSource) {
        alert('Please select a video source first.');
        return;
    }

    if (confirm('Are you sure you want to clear ALL area configurations? This will delete the area config file and restart area configuration from scratch.')) {
        try {
            const response = await fetch(`${API_ENDPOINTS.SOURCES}/${selectedSource}/areas/clear-all`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            });

            if (response.ok) {
                // Clear local areas
                existingAreas = {};

                // Clear canvas
                clearCanvas();

                showToast('All area configurations cleared successfully! Ready to define new areas.', 'success');
            } else {
                const error = await response.json();
                showToast(`Failed to clear area configurations: ${error.detail}`, 'error');
            }
        } catch (error) {
            console.error('Error clearing all area configurations:', error);
            showToast('Failed to clear area configurations. Check console for errors.', 'error');
        }
    }
}

// Clear the entire canvas
function clearCanvas() {
    // Stop any existing drawing
    stopDrawing();
    
    // Clear existing areas
    existingAreas = {};
    
    // Clear canvas
    fabricCanvas.clear();
}

// Save area configuration
async function saveAreaConfiguration() {
    if (!selectedSource) {
        alert('Please select a video source first.');
        return;
    }
    
    try {
        const response = await fetch(`${API_ENDPOINTS.SOURCES}/${selectedSource}/areas`, {
            method: 'PUT',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(existingAreas)
        });
        
        if (response.ok) {
            alert('Area configuration saved successfully!');
        } else {
            const error = await response.json();
            alert(`Failed to save area configuration: ${error.detail}`);
        }
    } catch (error) {
        console.error('Error saving area configuration:', error);
        alert('Failed to save area configuration. Check console for errors.');
    }
}

// Toggle player controls
function toggleControls() {
    const controlsDiv = document.getElementById('playerControls');
    const btnToggle = document.getElementById('btnToggleControls');
    
    if (controlsDiv.style.display === 'none') {
        controlsDiv.style.display = 'flex';
        btnToggle.innerHTML = '<i class="fas fa-cog"></i> Hide Controls';
        
        // Load current model settings if available
        if (selectedSource) {
            loadModelSettings(selectedSource);
        }
    } else {
        controlsDiv.style.display = 'none';
        btnToggle.innerHTML = '<i class="fas fa-cog"></i> Show Controls';
    }
}

// Load model settings for a source
async function loadModelSettings(sourceId) {
    try {
        const response = await fetch(`${API_ENDPOINTS.SOURCES}/${sourceId}/settings`);
        if (!response.ok) {
            console.warn('Failed to load model settings, using defaults');
            return;
        }
        
        const settings = await response.json();
        
        // Update toggle switches - default to false (disabled)
        if (settings.models) {
            document.getElementById('toggleAccident').checked = settings.models.accident_detection ?? false;
            document.getElementById('toggleHelmet').checked = settings.models.helmet_detection ?? false;
            document.getElementById('toggleTraffic').checked = settings.models.traffic_violation ?? false;
            document.getElementById('toggleSpeed').checked = settings.models.speed_detection ?? false;
            document.getElementById('toggleParking').checked = settings.models.parking_detection ?? false;
            document.getElementById('toggleWrongDir').checked = settings.models.wrong_direction ?? false;

            // Update model status buttons - default to false (disabled)
            updateModelButtonStatus('accident', settings.models.accident_detection ?? false);
            updateModelButtonStatus('helmet', settings.models.helmet_detection ?? false);
        } else {
            // Initialize with default values if no settings found (all disabled)
            initializeModelButtons();
        }
    } catch (error) {
        console.error('Error loading model settings:', error);
        showToast('Failed to load model settings. Using defaults (all disabled).', 'warning');
        // Initialize with all models disabled on error
        initializeModelButtons();
    }
}

// Save model settings
async function saveModelSettings() {
    if (!selectedSource) {
        alert('Please select a video source first.');
        return;
    }
    
    const settings = {
        models: {
            accident_detection: document.getElementById('toggleAccident')?.checked ?? false,
            helmet_detection: document.getElementById('toggleHelmet')?.checked ?? false,
            traffic_violation: document.getElementById('toggleTraffic')?.checked ?? false,
            speed_detection: document.getElementById('toggleSpeed')?.checked ?? false,
            parking_detection: document.getElementById('toggleParking')?.checked ?? false,
            wrong_direction: document.getElementById('toggleWrongDir')?.checked ?? false
        }
    };
    
    try {
        const response = await fetch(`${API_ENDPOINTS.SOURCES}/${selectedSource}/settings`, {
            method: 'PUT',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(settings)
        });
        
        if (response.ok) {
            showToast('Model settings applied successfully!', 'success');
        } else {
            const error = await response.json();
            showToast(`Failed to apply settings: ${error.detail}`, 'error');
        }
    } catch (error) {
        console.error('Error saving model settings:', error);
        showToast('Failed to save settings. Check console for errors.', 'error');
    }
}

// Toggle Accident Detection Model
async function toggleAccidentModel() {
    if (!selectedSource) {
        alert('Please select a video source first.');
        return;
    }

    const button = document.getElementById('btnAccidentModel');
    const badge = document.getElementById('badgeAccidentStatus');
    const isCurrentlyActive = button.classList.contains('active');

    // Set loading state
    button.classList.add('loading');
    badge.textContent = 'LOADING';

    try {
        // Get current settings
        const response = await fetch(`${API_ENDPOINTS.SOURCES}/${selectedSource}/settings`);
        let settings = { models: {} };

        if (response.ok) {
            settings = await response.json();
        }

        // Toggle accident detection
        const newState = !isCurrentlyActive;
        settings.models.accident_detection = newState;

        // Save settings
        const saveResponse = await fetch(`${API_ENDPOINTS.SOURCES}/${selectedSource}/settings`, {
            method: 'PUT',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(settings)
        });

        if (saveResponse.ok) {
            updateModelButtonStatus('accident', newState);
            showToast(`Accident Detection ${newState ? 'enabled' : 'disabled'}`, 'success');

            // Also update the toggle switch in the controls panel
            document.getElementById('toggleAccident').checked = newState;
        } else {
            const error = await saveResponse.json();
            showToast(`Failed to toggle accident detection: ${error.detail}`, 'error');
        }
    } catch (error) {
        console.error('Error toggling accident detection:', error);
        showToast('Failed to toggle accident detection. Check console for errors.', 'error');
    } finally {
        // Remove loading state
        button.classList.remove('loading');
    }
}

// Toggle Helmet Detection Model
async function toggleHelmetModel() {
    if (!selectedSource) {
        alert('Please select a video source first.');
        return;
    }

    const button = document.getElementById('btnHelmetModel');
    const badge = document.getElementById('badgeHelmetStatus');
    const isCurrentlyActive = button.classList.contains('active');

    // Set loading state
    button.classList.add('loading');
    badge.textContent = 'LOADING';

    try {
        // Get current settings
        const response = await fetch(`${API_ENDPOINTS.SOURCES}/${selectedSource}/settings`);
        let settings = { models: {} };

        if (response.ok) {
            settings = await response.json();
        }

        // Toggle helmet detection
        const newState = !isCurrentlyActive;
        settings.models.helmet_detection = newState;

        // Save settings
        const saveResponse = await fetch(`${API_ENDPOINTS.SOURCES}/${selectedSource}/settings`, {
            method: 'PUT',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(settings)
        });

        if (saveResponse.ok) {
            updateModelButtonStatus('helmet', newState);
            showToast(`Helmet Detection ${newState ? 'enabled' : 'disabled'}`, 'success');

            // Also update the toggle switch in the controls panel
            document.getElementById('toggleHelmet').checked = newState;
        } else {
            const error = await saveResponse.json();
            showToast(`Failed to toggle helmet detection: ${error.detail}`, 'error');
        }
    } catch (error) {
        console.error('Error toggling helmet detection:', error);
        showToast('Failed to toggle helmet detection. Check console for errors.', 'error');
    } finally {
        // Remove loading state
        button.classList.remove('loading');
    }
}

// Update model button status
function updateModelButtonStatus(modelType, isActive) {
    const buttonId = modelType === 'accident' ? 'btnAccidentModel' : 'btnHelmetModel';
    const badgeId = modelType === 'accident' ? 'badgeAccidentStatus' : 'badgeHelmetStatus';

    const button = document.getElementById(buttonId);
    const badge = document.getElementById(badgeId);

    // Remove existing classes
    button.classList.remove('active', 'inactive', 'btn-outline-secondary');

    // Add model-status-btn class if not present
    if (!button.classList.contains('model-status-btn')) {
        button.classList.add('model-status-btn');
    }

    if (isActive) {
        button.classList.add('active');
        badge.textContent = 'ON';
        badge.classList.remove('bg-danger');
        badge.classList.add('bg-success');
    } else {
        button.classList.add('inactive');
        badge.textContent = 'OFF';
        badge.classList.remove('bg-success');
        badge.classList.add('bg-danger');
    }
}

// Initialize model button statuses when loading settings
function initializeModelButtons() {
    // Set initial state to inactive (disabled by default)
    updateModelButtonStatus('accident', false);
    updateModelButtonStatus('helmet', false);

    // Also initialize toggle switches to unchecked
    document.getElementById('toggleAccident').checked = false;
    document.getElementById('toggleHelmet').checked = false;
    document.getElementById('toggleTraffic').checked = false;
    document.getElementById('toggleSpeed').checked = false;
    document.getElementById('toggleParking').checked = false;
    document.getElementById('toggleWrongDir').checked = false;
}

// Initialize the area configuration UI when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    console.log('Area configuration UI initializing...');

    // Initialize canvas and load video sources
    setupAreaConfigUI();

    console.log('Area configuration UI initialized');
});