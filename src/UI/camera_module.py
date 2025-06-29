import cv2
import numpy as np
from typing import Optional, Tuple, List, Callable
from PyQt6.QtCore import QObject, pyqtSignal, QTimer, QThread
from PyQt6.QtGui import QImage, QPixmap
import time

class CameraManager(QObject):
    """Advanced camera manager with multiple camera support and enhanced features"""
    
    frame_ready = pyqtSignal(QImage)
    camera_error = pyqtSignal(str)
    camera_connected = pyqtSignal(int)
    camera_disconnected = pyqtSignal(int)
    
    def __init__(self):
        super().__init__()
        self.cameras = {}
        self.current_camera_id = 0
        self.is_capturing = False
        self.capture_thread = None
        self.frame_timer = QTimer()
        self.frame_timer.timeout.connect(self.capture_frame)
        
        # Camera settings
        self.resolution = (1920, 1080)
        self.fps = 30
        self.auto_focus = True
        self.image_stabilization = True
        
    def get_available_cameras(self) -> List[int]:
        """Get list of available camera indices"""
        available_cameras = []
        for i in range(10):  # Check first 10 camera indices
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                available_cameras.append(i)
                cap.release()
        return available_cameras
    
    def connect_camera(self, camera_id: int = 0) -> bool:
        """Connect to a specific camera"""
        if camera_id in self.cameras:
            return True
            
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            self.camera_error.emit(f"Could not open camera {camera_id}")
            return False
            
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        
        # Enable auto-focus if available
        if self.auto_focus:
            cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
            
        # Enable image stabilization if available
        if self.image_stabilization:
            cap.set(cv2.CAP_PROP_IMAGE_STABILIZATION, 1)
            
        self.cameras[camera_id] = cap
        self.current_camera_id = camera_id
        self.camera_connected.emit(camera_id)
        return True
        
    def disconnect_camera(self, camera_id: int):
        """Disconnect a specific camera"""
        if camera_id in self.cameras:
            self.cameras[camera_id].release()
            del self.cameras[camera_id]
            self.camera_disconnected.emit(camera_id)
            
    def start_capture(self, camera_id: Optional[int] = None):
        """Start capturing frames from camera"""
        if camera_id is not None:
            self.current_camera_id = camera_id
            
        if self.current_camera_id not in self.cameras:
            if not self.connect_camera(self.current_camera_id):
                return False
                
        self.is_capturing = True
        self.frame_timer.start(1000 // self.fps)  # Convert FPS to milliseconds
        return True
        
    def stop_capture(self):
        """Stop capturing frames"""
        self.is_capturing = False
        self.frame_timer.stop()
        
    def capture_frame(self):
        """Capture a single frame from the current camera"""
        if not self.is_capturing or self.current_camera_id not in self.cameras:
            return
            
        cap = self.cameras[self.current_camera_id]
        ret, frame = cap.read()
        
        if ret:
            # Apply image stabilization if enabled
            if self.image_stabilization:
                frame = self.stabilize_image(frame)
                
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Convert to QImage
            h, w, ch = rgb_frame.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            
            self.frame_ready.emit(qt_image)
        else:
            self.camera_error.emit(f"Failed to capture frame from camera {self.current_camera_id}")
            
    def capture_single_image(self, camera_id: Optional[int] = None) -> Optional[np.ndarray]:
        """Capture a single high-quality image"""
        if camera_id is not None:
            self.current_camera_id = camera_id
            
        if self.current_camera_id not in self.cameras:
            if not self.connect_camera(self.current_camera_id):
                return None
                
        cap = self.cameras[self.current_camera_id]
        
        # Capture multiple frames to get the best one
        best_frame = None
        best_sharpness = 0
        
        for _ in range(5):  # Capture 5 frames and pick the sharpest
            ret, frame = cap.read()
            if ret:
                # Calculate sharpness (Laplacian variance)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
                
                if sharpness > best_sharpness:
                    best_sharpness = sharpness
                    best_frame = frame.copy()
                    
            time.sleep(0.1)  # Small delay between captures
            
        return best_frame
        
    def stabilize_image(self, frame: np.ndarray) -> np.ndarray:
        """Apply simple image stabilization"""
        if not hasattr(self, '_prev_frame'):
            self._prev_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            return frame
            
        # Convert current frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate optical flow
        flow = cv2.calcOpticalFlowFarneback(
            self._prev_frame, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        
        # Apply stabilization
        h, w = frame.shape[:2]
        flow_map = np.column_stack((flow.reshape(-1, 2), np.zeros((h * w, 1))))
        flow_map = flow_map.reshape(h, w, 3)
        
        # Create transformation matrix
        transform_matrix = np.eye(3, dtype=np.float32)
        transform_matrix[:2, 2] = -np.mean(flow, axis=(0, 1))
        
        # Apply transformation
        stabilized = cv2.warpAffine(frame, transform_matrix[:2], (w, h))
        
        # Update previous frame
        self._prev_frame = gray
        
        return stabilized
        
    def set_camera_settings(self, resolution: Tuple[int, int] = None, 
                           fps: int = None, auto_focus: bool = None,
                           image_stabilization: bool = None):
        """Update camera settings"""
        if resolution:
            self.resolution = resolution
        if fps:
            self.fps = fps
        if auto_focus is not None:
            self.auto_focus = auto_focus
        if image_stabilization is not None:
            self.image_stabilization = image_stabilization
            
        # Apply settings to all connected cameras
        for camera_id, cap in self.cameras.items():
            if resolution:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
            if fps:
                cap.set(cv2.CAP_PROP_FPS, fps)
            if auto_focus is not None:
                cap.set(cv2.CAP_PROP_AUTOFOCUS, 1 if auto_focus else 0)
            if image_stabilization is not None:
                cap.set(cv2.CAP_PROP_IMAGE_STABILIZATION, 1 if image_stabilization else 0)
                
    def get_camera_info(self, camera_id: int) -> dict:
        """Get information about a specific camera"""
        if camera_id not in self.cameras:
            return {}
            
        cap = self.cameras[camera_id]
        info = {
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'brightness': cap.get(cv2.CAP_PROP_BRIGHTNESS),
            'contrast': cap.get(cv2.CAP_PROP_CONTRAST),
            'saturation': cap.get(cv2.CAP_PROP_SATURATION),
            'hue': cap.get(cv2.CAP_PROP_HUE),
            'gain': cap.get(cv2.CAP_PROP_GAIN),
            'exposure': cap.get(cv2.CAP_PROP_EXPOSURE),
            'auto_focus': cap.get(cv2.CAP_PROP_AUTOFOCUS),
            'focus': cap.get(cv2.CAP_PROP_FOCUS),
        }
        return info
        
    def cleanup(self):
        """Clean up all camera resources"""
        self.stop_capture()
        for camera_id in list(self.cameras.keys()):
            self.disconnect_camera(camera_id)

class CameraWidget(QObject):
    """Enhanced camera widget with advanced features"""
    
    image_captured = pyqtSignal(np.ndarray)
    capture_error = pyqtSignal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.camera_manager = CameraManager()
        self.camera_manager.frame_ready.connect(self.on_frame_ready)
        self.camera_manager.camera_error.connect(self.on_camera_error)
        
        # Camera settings
        self.auto_capture = False
        self.auto_capture_timer = QTimer()
        self.auto_capture_timer.timeout.connect(self.auto_capture_image)
        
        # Image quality settings
        self.min_sharpness = 100  # Minimum sharpness for auto-capture
        self.capture_countdown = 0
        
    def on_frame_ready(self, qimage: QImage):
        """Handle new frame from camera"""
        # Convert QImage to numpy array for processing
        width = qimage.width()
        height = qimage.height()
        ptr = qimage.bits()
        ptr.setsize(height * width * 3)
        arr = np.frombuffer(ptr, np.uint8).reshape((height, width, 3))
        
        # Check if auto-capture should trigger
        if self.auto_capture:
            self.check_auto_capture(arr)
            
    def on_camera_error(self, error_message: str):
        """Handle camera errors"""
        self.capture_error.emit(error_message)
        
    def check_auto_capture(self, frame: np.ndarray):
        """Check if frame meets auto-capture criteria"""
        # Convert to grayscale for sharpness calculation
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Check if image is sharp enough and stable
        if sharpness > self.min_sharpness:
            if self.capture_countdown <= 0:
                self.image_captured.emit(frame)
                self.auto_capture = False
                self.auto_capture_timer.stop()
            else:
                self.capture_countdown -= 1
        else:
            # Reset countdown if image is not sharp
            self.capture_countdown = 10  # Wait 10 frames for stability
            
    def start_auto_capture(self, countdown_frames: int = 30):
        """Start automatic capture when conditions are met"""
        self.auto_capture = True
        self.capture_countdown = countdown_frames
        self.auto_capture_timer.start(100)  # Check every 100ms
        
    def stop_auto_capture(self):
        """Stop automatic capture"""
        self.auto_capture = False
        self.auto_capture_timer.stop()
        
    def capture_manual(self) -> Optional[np.ndarray]:
        """Manually capture an image"""
        return self.camera_manager.capture_single_image()
        
    def start_camera(self, camera_id: int = 0) -> bool:
        """Start camera capture"""
        return self.camera_manager.start_capture(camera_id)
        
    def stop_camera(self):
        """Stop camera capture"""
        self.camera_manager.stop_capture()
        
    def set_camera_settings(self, **kwargs):
        """Set camera settings"""
        self.camera_manager.set_camera_settings(**kwargs)
        
    def get_available_cameras(self) -> List[int]:
        """Get list of available cameras"""
        return self.camera_manager.get_available_cameras()
        
    def cleanup(self):
        """Clean up camera resources"""
        self.camera_manager.cleanup() 