# Vision Patrol - Main multiprocessing video processor
import cv2
import time
import sys
import os
import multiprocessing

# Add the Processing Models directory to the Python path
processing_models_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Processing Models")
sys.path.append(processing_models_path)

# Import VideoProcessorMP from the Processing Models directory
from VideoProcessorMP import VideoProcessorMP

# Required for Windows to ensure proper subprocess execution
if __name__ == '__main__':
    # Configure multiprocessing for Windows
    multiprocessing.freeze_support()
    
    # Define video sources - add more or change sources as needed
    video_sources = [
        # Example configuration for local video file
        # {
        #     "id": "video2", 
        #     "source": r"Testing Vids\license vid.mp4",
        #     "use_stream": False,
        #     "location": "Test Road Intersection",
        #     "coordinates": {"lat": 0.0, "lng": 0.0},
        #     "speed_limit": 1
        # },
        # Example configuration for YouTube stream
        # {
        #     "id": "YouTube Stream", 
        #     "source": r"https://youtu.be/RGY622xx1s4",
        #     "use_stream": True,
        #     "location": "Downtown Crossing",
        #     "coordinates": {"lat": 0.0, "lng": 0.0},
        #     "speed_limit": 50
        # }
    ]
    

    # Create and start video processor processes
    processors = []

    try:
        # Create shared exit event for coordinated termination
        exit_event = multiprocessing.Event()
        
        # Create a process for each video source
        for source in video_sources:
            # Ensure the source file exists
            if not source["use_stream"] and not os.path.exists(source["source"]):
                print(f"Warning: Source file '{source['source']}' not found. Skipping.")
                continue
                
            processor = VideoProcessorMP(
                video_id=source["id"],
                source=source["source"],
                use_stream=source["use_stream"],
                camera_location=source.get("location", "Unknown"),
                coordinates=source.get("coordinates", {"lat": 0.0, "lng": 0.0}),
                speed_limit=source.get("speed_limit", 60)  # Default to 60 km/h if not specified
            )
            processors.append(processor)
            processor.start()
            # Small delay to avoid overwhelming the system when starting multiple processes
            time.sleep(1)

        print(f"Started {len(processors)} video processors")
        
        # Main monitoring loop - keeps the program running until user exits
        while not exit_event.is_set():
            # Check if all processes are still alive
            all_alive = False
            for processor in processors:
                if processor.is_alive():
                    all_alive = True
                    break
            
            if not all_alive:
                print("All processes have stopped, exiting...")
                break
            
            # Global key handling for user exit
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):  # ESC or Q key for global exit
                print("Exit key pressed globally, terminating all processes...")
                exit_event.set()
                break
                
            # Brief pause to prevent excessive CPU usage
            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
    except Exception as e:
        print(f"\nError: {str(e)}")
    finally:
        print("\nShutting down all video processors...")
        
        # Signal all processes to exit gracefully
        for processor in processors:
            if processor.is_alive():
                processor.stop()
                print(f"Stop signal sent to {processor.video_id}")
        
        # Wait for processes to terminate with timeout
        print("Waiting for processes to terminate...")
        start_time = time.time()
        timeout = 5  # Timeout in seconds
        
        while time.time() - start_time < timeout:
            if not any(p.is_alive() for p in processors):
                print("All processes successfully terminated")
                break
            time.sleep(0.5)
        
        # Force terminate any remaining processes
        for processor in processors:
            if processor.is_alive():
                print(f"Forcing termination of {processor.video_id}...")
                processor.terminate()
                processor.join(1)  # Allow time for process cleanup
        
        # Close all OpenCV windows
        cv2.destroyAllWindows()
        
        # Allow OS time to process window destroy commands
        time.sleep(0.5)
        
        print("Cleanup completed. Exiting program.")
        sys.exit(0)