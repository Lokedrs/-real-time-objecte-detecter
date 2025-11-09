"""
Real-Time Object Detection System
Main application that coordinates webcam capture, object detection, and display
"""

import cv2
import time
import argparse
import pyttsx3
import os
import sys
from object_detector import ObjectDetector

class RealTimeObjectDetection:
    def __init__(self, config_path, weights_path, names_path, use_voice=False, save_frames_for=None):
        """
        Initialize real-time object detection system
        
        Args:
            config_path: Path to YOLO config file
            weights_path: Path to YOLO weights file
            names_path: Path to COCO names file
            use_voice: Enable voice announcements
            save_frames_for: Save frames when specific object detected
        """
        print("🚀 Initializing Real-Time Object Detection System...")
        
        # Initialize object detector
        self.detector = ObjectDetector(config_path, weights_path, names_path)
        
        # Voice settings
        self.use_voice = use_voice
        self.tts_engine = None
        self._init_voice_engine()
        
        # Frame saving
        self.save_frames_for = save_frames_for
        self.saved_frame_count = 0
        
        # Performance tracking
        self.prev_frame_time = 0
        self.new_frame_time = 0
        self.frame_count = 0
        self.fps = 0
        
        # Object tracking
        self.previous_objects = set()

        # Detection display toggle
        self.show_detections = True
        
        print("✅ System initialized successfully!")
    
    def _init_voice_engine(self):
        """Initialize text-to-speech engine"""
        if not self.use_voice:
            return
            
        try:
            self.tts_engine = pyttsx3.init()
            # Configure voice settings
            self.tts_engine.setProperty('rate', 150)  # Speaking speed
            self.tts_engine.setProperty('volume', 0.8)  # Volume level
            
            # Get available voices
            voices = self.tts_engine.getProperty('voices')
            if voices:
                self.tts_engine.setProperty('voice', voices[0].id)
                
            print("✅ Voice output enabled")
        except Exception as e:
            print(f"❌ Voice output disabled: {e}")
            self.use_voice = False
    
    def calculate_fps(self):
        """Calculate frames per second"""
        self.new_frame_time = time.time()
        
        if self.prev_frame_time > 0:
            time_diff = self.new_frame_time - self.prev_frame_time
            current_fps = 1 / time_diff if time_diff > 0 else 0
            
            # Smooth FPS calculation
            self.fps = 0.8 * self.fps + 0.2 * current_fps if self.fps > 0 else current_fps
        
        self.prev_frame_time = self.new_frame_time
        self.frame_count += 1
        
        return self.fps
    
    def announce_objects(self, detected_objects):
        """Announce newly detected objects using TTS"""
        if not self.use_voice or not self.tts_engine or not detected_objects:
            return
        
        current_objects = set(detected_objects)
        new_objects = current_objects - self.previous_objects
        
        if new_objects:
            # Limit announcement length
            objects_to_announce = list(new_objects)[:3]  # Max 3 objects per announcement
            
            if len(objects_to_announce) == 1:
                announcement = f"Detected {objects_to_announce[0]}"
            else:
                announcement = f"Detected {', '.join(objects_to_announce[:-1])} and {objects_to_announce[-1]}"
            
            try:
                # Run voice announcement in a non-blocking way
                self.tts_engine.say(announcement)
                self.tts_engine.runAndWait()
            except Exception as e:
                print(f"🔊 Voice announcement failed: {e}")
        
        self.previous_objects = current_objects
    
    def save_frame(self, frame, object_name):
        """Save frame to disk"""
        save_dir = "saved_frames"
        os.makedirs(save_dir, exist_ok=True)
        
        timestamp = int(time.time())
        filename = f"{save_dir}/{object_name}_{self.saved_frame_count:04d}_{timestamp}.jpg"
        
        success = cv2.imwrite(filename, frame)
        if success:
            print(f"📸 Frame saved: {filename}")
            self.saved_frame_count += 1
        else:
            print(f"❌ Failed to save frame: {filename}")
    
    def setup_camera(self):
        """Initialize and configure webcam"""
        print("📷 Initializing webcam...")
        
        # Try different camera indices
        for camera_index in [0, 1, 2]:
            cap = cv2.VideoCapture(camera_index)
            
            if cap.isOpened():
                # Configure camera
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv2.CAP_PROP_FPS, 30)
                
                print(f"✅ Webcam found at index {camera_index}")
                return cap
        
        print("❌ No webcam found!")
        return None
    
    def draw_ui_overlay(self, frame, fps, object_count):
        """Draw UI elements on the frame"""
        # FPS display
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Object count
        cv2.putText(frame, f"Objects: {object_count}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Status information
        status_lines = []
        if self.use_voice:
            status_lines.append("Voice: ON")
        if self.save_frames_for:
            status_lines.append(f"Save: {self.save_frames_for}")
        
        for i, line in enumerate(status_lines):
            cv2.putText(frame, line, (10, 90 + i * 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Instructions
        instructions = [
            "Press 'Q' to quit",
            "Press 'V' to toggle voice",
            "Press 'S' to save frame",
            "Press 'D' to toggle detection"
        ]
        
        for i, instruction in enumerate(instructions):
            y_pos = frame.shape[0] - 80 + i * 20
            cv2.putText(frame, instruction, (10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def run_detection(self):
        """Main detection loop"""
        # Initialize camera
        cap = self.setup_camera()
        if cap is None:
            return
        
        print("\n🎬 Starting real-time object detection...")
        print("⌨️ Controls:")
        print("   - Press 'Q' to quit")
        print("   - Press 'V' to toggle voice output")
        print("   - Press 'S' to save current frame")
        print("   - Press 'C' to clear object tracking")
        print("   - Press 'D' to toggle detection display")
        print("\n⏳ Starting in 3 seconds...")
        time.sleep(3)
        
        try:
            while True:
                # Read frame
                ret, frame = cap.read()
                if not ret:
                    print("❌ Failed to capture frame")
                    break
                
                # Detect objects
                boxes, confidences, class_ids = self.detector.detect_objects(frame)
                
                # Draw detections if enabled
                if self.show_detections:
                    frame_with_detections, detected_objects = self.detector.draw_detections(
                        frame.copy(), boxes, confidences, class_ids
                    )
                else:
                    frame_with_detections = frame.copy()
                    detected_objects = []
                
                # Calculate FPS
                fps = self.calculate_fps()
                
                # Draw UI
                self.draw_ui_overlay(frame_with_detections, fps, len(detected_objects))
                
                # Voice announcements
                self.announce_objects(detected_objects)
                
                # Save frames for specific objects
                if self.save_frames_for and self.save_frames_for in detected_objects:
                    self.save_frame(frame_with_detections, self.save_frames_for)
                
                # Display frame
                cv2.imshow("Real-Time Object Detection - Press Q to quit", frame_with_detections)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == ord('Q'):
                    break
                elif key == ord('v') or key == ord('V'):
                    self.use_voice = not self.use_voice
                    status = "ON" if self.use_voice else "OFF"
                    print(f"🔊 Voice output: {status}")
                elif key == ord('s') or key == ord('S'):
                    self.save_frame(frame_with_detections, "manual_capture")
                elif key == ord('c') or key == ord('C'):
                    self.previous_objects.clear()
                    print("🧹 Object tracking cleared")
                elif key == ord('d') or key == ord('D'):
                    self.show_detections = not self.show_detections
                    status = "ON" if self.show_detections else "OFF"
                    print(f"👁️ Detection display: {status}")
                
        except KeyboardInterrupt:
            print("\n🛑 Detection interrupted by user")
        except Exception as e:
            print(f"❌ Error during detection: {e}")
        finally:
            # Cleanup
            cap.release()
            cv2.destroyAllWindows()
            if self.tts_engine:
                self.tts_engine.stop()
            
            print(f"✅ System shutdown complete.")
            print(f"📊 Stats: Processed {self.frame_count} frames, Saved {self.saved_frame_count} frames")

def check_models():
    """Check if model files exist"""
    required_files = [
        "models/yolov3.cfg",
        "models/yolov3.weights", 
        "models/coco.names"
    ]
    
    missing = [f for f in required_files if not os.path.exists(f)]
    
    if missing:
        print("❌ Missing model files detected!")
        for file in missing:
            print(f"   - {file}")
        print(f"\n💡 Please run: python download_models.py")
        return False
    
    return True

def main():
    """Main application entry point"""
    print("=" * 60)
    print("       REAL-TIME OBJECT DETECTION SYSTEM")
    print("=" * 60)
    
    # Check model files first
    if not check_models():
        return
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Real-time Object Detection using YOLOv3 and OpenCV",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                          # Basic detection
  python main.py --voice                  # With voice announcements  
  python main.py --save-for person        # Save frames with people
  python main.py --voice --save-for car   # Voice + save cars
        """
    )
    
    parser.add_argument("--config", default="models/yolov3.cfg", 
                       help="Path to YOLO config file (default: models/yolov3.cfg)")
    parser.add_argument("--weights", default="models/yolov3.weights",
                       help="Path to YOLO weights file (default: models/yolov3.weights)")
    parser.add_argument("--names", default="models/coco.names",
                       help="Path to COCO names file (default: models/coco.names)")
    parser.add_argument("--voice", action="store_true",
                       help="Enable voice output for detected objects")
    parser.add_argument("--save-for", 
                       help="Save frames when specific object is detected (e.g., 'person', 'car')")
    
    args = parser.parse_args()
    
    # Initialize and run detection system
    try:
        detector_system = RealTimeObjectDetection(
            config_path=args.config,
            weights_path=args.weights,
            names_path=args.names,
            use_voice=args.voice,
            save_frames_for=args.save_for
        )
        
        detector_system.run_detection()
        
    except Exception as e:
        print(f"❌ Failed to start object detection: {e}")
        print("💡 Troubleshooting tips:")
        print("   - Make sure webcam is connected and accessible")
        print("   - Check if model files are complete")
        print("   - Try running with basic command: python main.py")

if __name__ == "__main__":
    main()
