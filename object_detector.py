"""
Object Detector using YOLOv3
Handles model loading, object detection, and drawing bounding boxes
"""

import cv2
import numpy as np
import os

class ObjectDetector:
    def __init__(self, config_path, weights_path, names_path, confidence_threshold=0.5, nms_threshold=0.4):
        """
        Initialize YOLO object detector
        
        Args:
            config_path: Path to YOLO config file
            weights_path: Path to YOLO weights file  
            names_path: Path to COCO names file
            confidence_threshold: Minimum confidence for detection (0-1)
            nms_threshold: Non-Maximum Suppression threshold (0-1)
        """
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        
        # Validate model files exist
        self._validate_model_files(config_path, weights_path, names_path)
        
        # Load YOLO model
        self._load_yolo_model(config_path, weights_path, names_path)
        
        print("✅ YOLO Object Detector initialized successfully!")
    
    def _validate_model_files(self, config_path, weights_path, names_path):
        """Check if all model files exist"""
        missing_files = []
        
        for path, name in [(config_path, "YOLO config"), 
                          (weights_path, "YOLO weights"), 
                          (names_path, "COCO names")]:
            if not os.path.exists(path):
                missing_files.append(f"{name} ({path})")
        
        if missing_files:
            raise FileNotFoundError(
                f"Missing model files: {', '.join(missing_files)}\n"
                f"Please run 'python download_models.py' first."
            )
    
    def _load_yolo_model(self, config_path, weights_path, names_path):
        """Load YOLO model and configuration"""
        try:
            # Load neural network
            print("🔄 Loading YOLO model...")
            self.net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
            
            # Configure backend (GPU/CPU)
            self._configure_backend()
            
            # Get output layer names
            self._get_output_layers()
            
            # Load class names
            self._load_class_names(names_path)
            
            # Generate colors for different classes
            self._generate_colors()
            
        except Exception as e:
            raise Exception(f"Failed to load YOLO model: {e}")
    
    def _configure_backend(self):
        """Configure computation backend (prefer GPU if available)"""
        # Check if CUDA is available
        cuda_available = False
        try:
            if hasattr(cv2, 'cuda') and cv2.cuda.getCudaEnabledDeviceCount() > 0:
                cuda_available = True
        except:
            cuda_available = False

        if cuda_available:
            try:
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
                print("✅ Using GPU acceleration (CUDA)")
                return
            except Exception as e:
                print(f"⚠️ CUDA backend failed: {e}, falling back to CPU")

        # Fall back to CPU
        try:
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            print("ℹ️ Using CPU computation")
        except Exception as e:
            print(f"⚠️ CPU backend failed: {e}, using default")
    
    def _get_output_layers(self):
        """Get the names of the output layers"""
        layer_names = self.net.getLayerNames()
        
        # Handle different OpenCV versions
        try:
            # OpenCV 4.x
            self.output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
        except:
            # OpenCV 3.x
            self.output_layers = [layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
    
    def _load_class_names(self, names_path):
        """Load COCO class names"""
        with open(names_path, 'r') as f:
            self.classes = [line.strip() for line in f.readlines()]
        print(f"📚 Loaded {len(self.classes)} object classes")
    
    def _generate_colors(self):
        """Generate random colors for each class"""
        np.random.seed(42)  # Consistent colors
        self.colors = np.random.uniform(0, 255, size=(len(self.classes), 3))
    
    def detect_objects(self, frame):
        """
        Detect objects in the given frame
        
        Args:
            frame: Input image frame (BGR format)
            
        Returns:
            boxes: List of bounding boxes [x, y, width, height]
            confidences: List of confidence scores (0-1)
            class_ids: List of class IDs
        """
        height, width = frame.shape[:2]
        
        # Create input blob for the network
        blob = cv2.dnn.blobFromImage(
            frame, 
            1/255.0,           # Scale factor
            (416, 416),        # Input size
            swapRB=True,       # Swap Red and Blue channels
            crop=False         # Don't crop
        )
        
        # Pass blob through network
        self.net.setInput(blob)
        outputs = self.net.forward(self.output_layers)
        
        # Process detections
        boxes, confidences, class_ids = self._process_detections(outputs, width, height)
        
        # Apply Non-Maximum Suppression
        boxes, confidences, class_ids = self._apply_nms(boxes, confidences, class_ids)
        
        return boxes, confidences, class_ids
    
    def _process_detections(self, outputs, width, height):
        """Process network outputs and extract detections"""
        boxes = []
        confidences = []
        class_ids = []
        
        for output in outputs:
            for detection in output:
                # Extract scores and find class with highest confidence
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                # Filter weak detections
                if confidence > self.confidence_threshold:
                    # Convert center coordinates to bounding box
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    box_width = int(detection[2] * width)
                    box_height = int(detection[3] * height)
                    
                    # Calculate top-left corner
                    x = int(center_x - box_width / 2)
                    y = int(center_y - box_height / 2)
                    
                    boxes.append([x, y, box_width, box_height])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        
        return boxes, confidences, class_ids
    
    def _apply_nms(self, boxes, confidences, class_ids):
        """Apply Non-Maximum Suppression to remove overlapping boxes"""
        if not boxes:
            return [], [], []
        
        # Convert to format required by OpenCV NMS
        boxes_array = np.array(boxes)
        confidences_array = np.array(confidences)
        
        # Apply NMS
        indices = cv2.dnn.NMSBoxes(
            boxes, 
            confidences, 
            self.confidence_threshold,
            self.nms_threshold
        )
        
        # Filter results based on NMS indices
        if len(indices) > 0:
            if hasattr(indices, 'flatten'):
                # OpenCV 4.x
                indices = indices.flatten()
            else:
                # OpenCV 3.x
                indices = indices[:, 0]
            
            final_boxes = [boxes[i] for i in indices]
            final_confidences = [confidences[i] for i in indices]
            final_class_ids = [class_ids[i] for i in indices]
            
            return final_boxes, final_confidences, final_class_ids
        else:
            return [], [], []
    
    def draw_detections(self, frame, boxes, confidences, class_ids):
        """
        Draw bounding boxes and labels on the frame
        
        Args:
            frame: Input frame to draw on
            boxes: List of bounding boxes
            confidences: List of confidence scores  
            class_ids: List of class IDs
            
        Returns:
            frame: Frame with drawn detections
            detected_objects: List of detected object names
        """
        detected_objects = []
        
        for i, (box, confidence, class_id) in enumerate(zip(boxes, confidences, class_ids)):
            x, y, w, h = box
            
            # Get color for this class
            color = [int(c) for c in self.colors[class_id]]
            
            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            
            # Create label text
            class_name = self.classes[class_id]
            label = f"{class_name}: {confidence:.2f}"
            detected_objects.append(class_name)
            
            # Calculate text size
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
            )
            
            # Draw label background
            cv2.rectangle(
                frame, 
                (x, y - text_height - baseline - 5),
                (x + text_width, y),
                color,
                -1  # Filled rectangle
            )
            
            # Draw label text
            cv2.putText(
                frame,
                label,
                (x, y - baseline - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),  # White text
                2
            )
        
        return frame, detected_objects
