import cv2
import os
import re
import requests
import time
import threading
from vidgear.gears import CamGear
from pytube import YouTube

class VideoReader:
    """
    Unified video reader class that handles both local videos (via OpenCV) and 
    streaming sources like YouTube (via CamGear or pafy).
    """
    def __init__(self, source, use_stream=False, verbose=True):
        """
        Initialize the video reader
        
        Args:
            source (str): Path to video file or URL to stream
            use_stream (bool): If True, use CamGear for streaming sources like YouTube
            verbose (bool): If True, print detailed logs
        """
        self.source = source
        self.use_stream = use_stream
        self.verbose = verbose
        self.stream = None
        self.cap = None
        self.initialized = False
        self.is_youtube = False
        
        # Add a lock for thread safety
        self.lock = threading.Lock()
        
        # Try to detect if this is a YouTube URL even if use_stream=False
        if isinstance(self.source, str) and ('youtube.com' in self.source or 'youtu.be' in self.source):
            self.is_youtube = True
            self.use_stream = True  # Force use_stream=True for YouTube URLs
            
        # Attempt to initialize the video source
        self._initialize_source()
        
    def _extract_youtube_id(self, url):
        """Extract YouTube video ID from URL"""
        youtube_id_match = re.search(r'(?:v=|\/)([0-9A-Za-z_-]{11}).*', url)
        if youtube_id_match:
            return youtube_id_match.group(1)
        return None
        
    def _verify_youtube_url(self, url):
        """
        Verify if a YouTube video exists and is accessible
        
        Returns:
            tuple: (success, video_url) where success is True if video is accessible
        """
        video_id = self._extract_youtube_id(url)
        if not video_id:
            if self.verbose:
                print(f"Could not extract YouTube video ID from: {url}")
            return False, url
            
        try:
            # Try to fetch video info page to check availability
            response = requests.head(f"https://img.youtube.com/vi/{video_id}/0.jpg")
            if response.status_code == 200:
                # Convert to standard URL format
                standard_url = f"https://www.youtube.com/watch?v={video_id}"
                if self.verbose:
                    print(f"YouTube video verified: {standard_url}")
                return True, standard_url
            else:
                if self.verbose:
                    print(f"YouTube video not available (status code: {response.status_code})")
                return False, url
        except Exception as e:
            if self.verbose:
                print(f"Error verifying YouTube URL: {e}")
            return False, url
            
    def _initialize_source(self):
        """Initialize the appropriate video source based on type"""
        if self.use_stream:
            # For YouTube/online streams
            if self.verbose:
                print(f"Connecting to stream: {self.source}")
                
            try:
                # Handle YouTube URLs
                if self.is_youtube:
                    # Verify and standardize the YouTube URL
                    valid, standard_url = self._verify_youtube_url(self.source)
                    if not valid and self.verbose:
                        print(f"Warning: YouTube URL validation failed, trying anyway")
                    source_url = standard_url
                else:
                    source_url = self.source

                # First try: Use YouTube-DL backend with CamGear
                if self.verbose:
                    print(f"Attempting to connect with CamGear YouTube-DL backend...")
                
                options = {
                    "THREADED_QUEUE_MODE": True,
                }
                
                try:
                    # Import required for YouTube handling
                    import yt_dlp
                    options["backend"] = "yt_dlp" # Use yt-dlp backend which works better than youtube-dl
                    
                    self.stream = CamGear(source=source_url, stream_mode=True, **options).start()
                    
                    # Test if stream is working
                    frame = self.stream.read()
                    if frame is None:
                        raise ValueError("CamGear stream returned empty frame")
                    
                    if self.verbose:
                        print(f"Successfully connected to stream using CamGear with yt-dlp backend")
                    self.initialized = True
                    return
                    
                except Exception as e:
                    if self.verbose:
                        print(f"yt-dlp backend failed: {e}")
                        print(f"Trying alternate methods...")
                        
                    # Clean up failed attempt
                    if hasattr(self, 'stream') and self.stream is not None:
                        self.stream.stop()
                
                # Second try: Use direct CamGear 
                try:
                    if self.verbose:
                        print("Trying direct CamGear connection...")
                    
                    self.stream = CamGear(source=source_url, **options).start()
                    frame = self.stream.read()
                    if frame is None:
                        raise ValueError("Direct CamGear returned empty frame")
                    
                    if self.verbose:
                        print("Successfully connected using direct CamGear")
                    self.initialized = True
                    return
                    
                except Exception as e:
                    if self.verbose:
                        print(f"Direct CamGear failed: {e}")
                    
                    # Clean up failed attempt
                    if hasattr(self, 'stream') and self.stream is not None:
                        self.stream.stop()
                
                # Third try: Use pytube with OpenCV
                try:
                    if self.is_youtube:
                        if self.verbose:
                            print("Trying pytube with OpenCV...")
                        
                        # Use pytube instead of pafy (which is deprecated)
                        
                        yt = YouTube(source_url)
                        stream = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first()
                        
                        if self.verbose:
                            print(f"Opening stream URL: {stream.url}")
                        
                        self.cap = cv2.VideoCapture(stream.url)
                        if not self.cap.isOpened():
                            raise ValueError("Failed to open stream with pytube/OpenCV")
                        
                        if self.verbose:
                            print("Successfully connected using pytube and OpenCV")
                        self.use_stream = False  # Switch to OpenCV mode
                        self.initialized = True
                        return
                except ImportError:
                    if self.verbose:
                        print("pytube not installed, skipping this method")
                except Exception as e:
                    if self.verbose:
                        print(f"pafy/OpenCV method failed: {e}")
                
                # Last resort: Direct OpenCV
                if self.verbose:
                    print("Trying direct OpenCV connection...")
                
                self.use_stream = False  # Switch to OpenCV mode
                self.cap = cv2.VideoCapture(self.source)
                if not self.cap.isOpened():
                    raise ValueError(f"All stream connection methods failed for: {self.source}")
                
                ret, frame = self.cap.read()
                if not ret:
                    raise ValueError("OpenCV returned empty frame")
                
                if self.verbose:
                    print("Successfully connected using direct OpenCV")
                self.initialized = True
                
            except Exception as e:
                if self.verbose:
                    print(f"Error connecting to stream: {str(e)}")
                raise ValueError(f"Failed to open stream: {self.source}")
        else:
            # For local video files and RTSP, use OpenCV's VideoCapture
            if self.verbose:
                print(f"Opening local video or RTSP: {self.source}")
                
            # Verify file exists if it's a local file (not RTSP or HTTP)
            if not self.source.startswith(("rtsp://", "rtmp://", "http://", "https://")) and not os.path.exists(self.source):
                raise FileNotFoundError(f"Video file not found: {self.source}")
            
            # Set OpenCV VideoCapture properties to address FFmpeg threading issues
            # These properties can help prevent the "Assertion fctx->async_lock failed" error
            self.cap = cv2.VideoCapture(self.source)
            
            # Set buffer size to 0 to get the latest frame
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            # Disable multi-threading in FFmpeg if causing issues
            # OpenCV 4.x:
            if hasattr(cv2, 'CAP_PROP_CODEC_PIXEL_FORMAT'):
                self.cap.set(cv2.CAP_PROP_CODEC_PIXEL_FORMAT, 0)
            
            if not self.cap.isOpened():
                raise ValueError(f"Failed to open video file: {self.source}")
            self.initialized = True
    
    def read(self):
        """
        Read a frame from the video source. Thread-safe.
        
        Returns:
            tuple: (success, frame) where success is True if frame was read successfully
        """
        if not self.initialized:
            return False, None
        
        # Use lock to ensure thread safety
        with self.lock:
            try:
                if self.use_stream and self.stream is not None:
                    frame = self.stream.read()
                    return frame is not None, frame
                elif self.cap is not None:
                    return self.cap.read()
                else:
                    return False, None
            except Exception as e:
                if self.verbose:
                    print(f"Error reading frame: {str(e)}")
                return False, None
    
    def get_frame_size(self):
        """
        Get the dimensions of video frames
        
        Returns:
            tuple: (width, height) of the video frames
        """
        if not self.initialized:
            return None
            
        try:
            if self.use_stream and self.stream is not None:
                # Try to get dimensions from a frame
                frame = self.read()[1]
                if frame is not None:
                    height, width = frame.shape[:2]
                    return width, height
                return None
            elif self.cap is not None:
                width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                return width, height
            else:
                return None
        except Exception as e:
            if self.verbose:
                print(f"Error getting frame size: {str(e)}")
            return None
    
    def get_fps(self):
        """
        Get the frames per second of the video
        
        Returns:
            float: Frames per second
        """
        if not self.initialized:
            return 0
            
        try:
            if self.use_stream and self.stream is not None:
                # Most streams run at 30 or 25 fps, return 30 as default
                return 30.0
            elif self.cap is not None:
                fps = self.cap.get(cv2.CAP_PROP_FPS)
                # If FPS is 0 or very small, default to a reasonable value
                if fps < 0.1:
                    fps = 30.0
                return fps
            else:
                return 0
        except Exception as e:
            if self.verbose:
                print(f"Error getting FPS: {str(e)}")
            return 30.0  # Safe default
    
    def reset(self):
        """
        Reset video to beginning (only works for local files). Thread-safe.
        """
        if not self.initialized:
            return
            
        # Use lock to ensure thread safety
        with self.lock:
            try:
                if not self.use_stream and self.cap is not None:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            except Exception as e:
                if self.verbose:
                    print(f"Warning: Could not reset video to start: {e}")
    
    def release(self):
        """
        Release the video resources. Thread-safe.
        """
        if not self.initialized:
            return
        
        # Use lock to ensure thread safety
        with self.lock:
            try:
                if self.use_stream and self.stream is not None:
                    if self.verbose:
                        print(f"Stopping stream...")
                    self.stream.stop()
                    self.stream = None
                elif self.cap is not None:
                    if self.verbose:
                        print(f"Releasing video capture...")
                    self.cap.release()
                    self.cap = None
                self.initialized = False
            except Exception as e:
                if self.verbose:
                    print(f"Error releasing video resource: {str(e)}")
    
    def __enter__(self):
        """Support context manager protocol"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Release resources when exiting context"""
        self.release()
