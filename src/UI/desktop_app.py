import sys
import os
import platform

# Add the project root to sys.path so 'src' can be imported
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFileDialog, QTextEdit, QGroupBox, QGridLayout, QSizePolicy, QScrollArea, QTabWidget, QComboBox
)
from PyQt6.QtGui import QPixmap, QImage, QFont
from PyQt6.QtCore import Qt, QTimer

import cv2
import numpy as np
import tempfile
from src.vision.preprocessing import preprocess_image
from src.vision.OCR_Processing import extract_text_with_confidence
from src.metadata.metadata_extraction import extract_metadata_with_gemini, metadata_combiner
from src.utils.isbn_detection import extract_and_validate_isbns

class CameraWidget(QWidget):
    """Widget for camera capture with live preview and camera selection"""
    def __init__(self, parent=None, camera_label_text="Camera Preview"):
        super().__init__(parent)
        self.camera = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.captured_image = None
        self.selected_camera_index = 0
        self.setup_ui(camera_label_text)

    def setup_ui(self, camera_label_text):
        layout = QVBoxLayout()

        # Camera selection dropdown
        self.camera_select = QComboBox()
        self.camera_select.setFixedWidth(180)
        self.camera_select.addItems(self.get_camera_list())
        self.camera_select.currentIndexChanged.connect(self.change_camera_index)
        layout.addWidget(self.camera_select)

        # Camera preview
        self.camera_label = QLabel(camera_label_text)
        self.camera_label.setMinimumSize(250, 350)
        self.camera_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.camera_label.setStyleSheet("border: 2px dashed #bdc3c7; background-color: #ecf0f1;")

        # Camera controls
        controls_layout = QHBoxLayout()
        self.start_btn = QPushButton("Start Camera")
        self.start_btn.clicked.connect(self.start_camera)
        self.start_btn.setStyleSheet("QPushButton { background-color: #27ae60; color: white; padding: 6px; border-radius: 4px; }")
        self.capture_btn = QPushButton("Capture")
        self.capture_btn.clicked.connect(self.capture_image)
        self.capture_btn.setEnabled(False)
        self.capture_btn.setStyleSheet("QPushButton { background-color: #3498db; color: white; padding: 6px; border-radius: 4px; }")
        self.stop_btn = QPushButton("Stop Camera")
        self.stop_btn.clicked.connect(self.stop_camera)
        self.stop_btn.setEnabled(False)
        self.stop_btn.setStyleSheet("QPushButton { background-color: #e74c3c; color: white; padding: 6px; border-radius: 4px; }")
        controls_layout.addWidget(self.start_btn)
        controls_layout.addWidget(self.capture_btn)
        controls_layout.addWidget(self.stop_btn)
        controls_layout.addStretch()

        layout.addWidget(self.camera_label)
        layout.addLayout(controls_layout)
        self.setLayout(layout)

    def get_camera_list(self):
        # Try up to 5 camera indices, only include if a frame can be read
        available = []
        for i in range(5):
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW) if platform.system() == 'Windows' else cv2.VideoCapture(i)
            if cap is not None and cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    available.append(f"Camera {i}")
                cap.release()
        if not available:
            available = ["Camera 0"]
        return available

    def change_camera_index(self, idx):
        self.selected_camera_index = idx
        self.stop_camera()

    def start_camera(self):
        # Use DirectShow backend on Windows for better compatibility
        if platform.system() == 'Windows':
            self.camera = cv2.VideoCapture(self.selected_camera_index, cv2.CAP_DSHOW)
        else:
            self.camera = cv2.VideoCapture(self.selected_camera_index)
        if self.camera.isOpened():
            self.timer.start(30)
            self.start_btn.setEnabled(False)
            self.capture_btn.setEnabled(True)
            self.stop_btn.setEnabled(True)
        else:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Camera Error", f"Could not open camera {self.selected_camera_index}")

    def stop_camera(self):
        if self.camera:
            self.timer.stop()
            self.camera.release()
            self.camera = None
            self.camera_label.clear()
            self.camera_label.setText("Camera Preview")
            self.start_btn.setEnabled(True)
            self.capture_btn.setEnabled(False)
            self.stop_btn.setEnabled(False)

    def update_frame(self):
        if self.camera and self.camera.isOpened():
            ret, frame = self.camera.read()
            if ret:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_frame.shape
                bytes_per_line = ch * w
                qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
                pixmap = QPixmap.fromImage(qt_image)
                scaled_pixmap = pixmap.scaled(self.camera_label.size(), 
                                            Qt.AspectRatioMode.KeepAspectRatio,
                                            Qt.TransformationMode.SmoothTransformation)
                self.camera_label.setPixmap(scaled_pixmap)

    def capture_image(self):
        if self.camera and self.camera.isOpened():
            ret, frame = self.camera.read()
            if ret:
                self.captured_image = frame
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_frame.shape
                bytes_per_line = ch * w
                qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
                pixmap = QPixmap.fromImage(qt_image)
                scaled_pixmap = pixmap.scaled(self.camera_label.size(), 
                                            Qt.AspectRatioMode.KeepAspectRatio,
                                            Qt.TransformationMode.SmoothTransformation)
                self.camera_label.setPixmap(scaled_pixmap)
                self.camera_label.setText("Image Captured!")
                # Notify parent to check if both covers are ready
                if self.parent() and hasattr(self.parent().parent(), 'check_ready_to_process'):
                    self.parent().parent().check_ready_to_process()
                return frame
        return None

class BookAcquisitionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Book Acquisition Tool (Desktop)")
        self.setGeometry(200, 100, 1600, 1000)

        # Central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Title
        title_label = QLabel("📚 Book Acquisition Tool")
        title_label.setFont(QFont("Arial", 20, QFont.Weight.Bold))
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setStyleSheet("color: #2c3e50; margin: 10px;")
        main_layout.addWidget(title_label)
        
        # --- Input Mode Selection ---
        mode_group = QGroupBox("📷 Input Mode")
        mode_layout = QHBoxLayout(mode_group)
        
        self.upload_mode_btn = QPushButton("File Upload")
        self.upload_mode_btn.setCheckable(True)
        self.upload_mode_btn.setChecked(True)
        self.upload_mode_btn.clicked.connect(lambda: self.switch_mode('upload'))
        self.upload_mode_btn.setStyleSheet("QPushButton { padding: 8px; border-radius: 4px; } QPushButton:checked { background-color: #3498db; color: white; }")
        
        self.camera_mode_btn = QPushButton("Camera Capture")
        self.camera_mode_btn.setCheckable(True)
        self.camera_mode_btn.clicked.connect(lambda: self.switch_mode('camera'))
        self.camera_mode_btn.setStyleSheet("QPushButton { padding: 8px; border-radius: 4px; } QPushButton:checked { background-color: #3498db; color: white; }")
        
        mode_layout.addWidget(self.upload_mode_btn)
        mode_layout.addWidget(self.camera_mode_btn)
        mode_layout.addStretch()
        main_layout.addWidget(mode_group)
        
        # --- Input Section ---
        self.input_group = QGroupBox("📷 Upload Book Covers")
        self.input_layout = QHBoxLayout(self.input_group)
        main_layout.addWidget(self.input_group)

        # Front cover group
        front_group = QGroupBox("Front Cover")
        front_layout = QVBoxLayout(front_group)
        self.front_upload_btn = QPushButton("Upload Front Cover")
        self.front_upload_btn.clicked.connect(lambda: self.upload_image('front'))
        self.front_upload_btn.setStyleSheet("QPushButton { background-color: #3498db; color: white; padding: 8px; border-radius: 4px; }")
        front_layout.addWidget(self.front_upload_btn)
        
        # Front cover images
        front_images_layout = QHBoxLayout()
        self.front_original_label = QLabel("Original")
        self.front_original_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.front_original_label.setFixedSize(250, 350)
        self.front_original_label.setStyleSheet("border: 2px dashed #bdc3c7; background-color: #ecf0f1;")
        front_images_layout.addWidget(self.front_original_label)
        
        self.front_preprocessed_label = QLabel("Preprocessed")
        self.front_preprocessed_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.front_preprocessed_label.setFixedSize(250, 350)
        self.front_preprocessed_label.setStyleSheet("border: 2px dashed #bdc3c7; background-color: #ecf0f1;")
        front_images_layout.addWidget(self.front_preprocessed_label)
        front_layout.addLayout(front_images_layout)
        self.input_layout.addWidget(front_group)

        # Back cover group
        back_group = QGroupBox("Back Cover")
        back_layout = QVBoxLayout(back_group)
        self.back_upload_btn = QPushButton("Upload Back Cover")
        self.back_upload_btn.clicked.connect(lambda: self.upload_image('back'))
        self.back_upload_btn.setStyleSheet("QPushButton { background-color: #3498db; color: white; padding: 8px; border-radius: 4px; }")
        back_layout.addWidget(self.back_upload_btn)
        
        # Back cover images
        back_images_layout = QHBoxLayout()
        self.back_original_label = QLabel("Original")
        self.back_original_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.back_original_label.setFixedSize(250, 350)
        self.back_original_label.setStyleSheet("border: 2px dashed #bdc3c7; background-color: #ecf0f1;")
        back_images_layout.addWidget(self.back_original_label)
        
        self.back_preprocessed_label = QLabel("Preprocessed")
        self.back_preprocessed_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.back_preprocessed_label.setFixedSize(250, 350)
        self.back_preprocessed_label.setStyleSheet("border: 2px dashed #bdc3c7; background-color: #ecf0f1;")
        back_images_layout.addWidget(self.back_preprocessed_label)
        back_layout.addLayout(back_images_layout)
        self.input_layout.addWidget(back_group)

        # --- Process Button ---
        self.process_btn = QPushButton("🚀 Process Both Covers")
        self.process_btn.setEnabled(False)
        self.process_btn.clicked.connect(self.process_both)
        self.process_btn.setStyleSheet("QPushButton { background-color: #27ae60; color: white; padding: 12px; border-radius: 6px; font-size: 14px; font-weight: bold; }")
        main_layout.addWidget(self.process_btn)

        # --- Results Section ---
        results_group = QGroupBox("📊 Results")
        results_layout = QVBoxLayout(results_group)
        main_layout.addWidget(results_group)

        # Create scrollable area for results
        scroll_area = QScrollArea()
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        scroll_area.setWidget(scroll_widget)
        scroll_area.setWidgetResizable(True)
        results_layout.addWidget(scroll_area)

        # OCR Results
        ocr_group = QGroupBox("📝 OCR Results")
        ocr_layout = QGridLayout(ocr_group)
        
        self.front_ocr_label = QLabel("Front Cover OCR:")
        self.front_ocr_text = QTextEdit()
        self.front_ocr_text.setReadOnly(True)
        self.front_ocr_text.setFixedHeight(120)
        self.front_ocr_text.setStyleSheet("QTextEdit { background-color: #f8f9fa; border: 1px solid #dee2e6; }")
        ocr_layout.addWidget(self.front_ocr_label, 0, 0)
        ocr_layout.addWidget(self.front_ocr_text, 0, 1)

        self.back_ocr_label = QLabel("Back Cover OCR:")
        self.back_ocr_text = QTextEdit()
        self.back_ocr_text.setReadOnly(True)
        self.back_ocr_text.setFixedHeight(120)
        self.back_ocr_text.setStyleSheet("QTextEdit { background-color: #f8f9fa; border: 1px solid #dee2e6; }")
        ocr_layout.addWidget(self.back_ocr_label, 1, 0)
        ocr_layout.addWidget(self.back_ocr_text, 1, 1)
        scroll_layout.addWidget(ocr_group)

        # Metadata and ISBNs
        data_group = QGroupBox("📋 Extracted Data")
        data_layout = QGridLayout(data_group)
        
        self.metadata_label = QLabel("Metadata (Front):")
        self.metadata_text = QTextEdit()
        self.metadata_text.setReadOnly(True)
        self.metadata_text.setFixedHeight(80)
        self.metadata_text.setStyleSheet("QTextEdit { background-color: #e8f5e8; border: 1px solid #c3e6c3; }")
        data_layout.addWidget(self.metadata_label, 0, 0)
        data_layout.addWidget(self.metadata_text, 0, 1)

        self.isbn_label = QLabel("ISBNs (Back):")
        self.isbn_text = QTextEdit()
        self.isbn_text.setReadOnly(True)
        self.isbn_text.setFixedHeight(80)
        self.isbn_text.setStyleSheet("QTextEdit { background-color: #e8f4fd; border: 1px solid #b3d9ff; }")
        data_layout.addWidget(self.isbn_label, 1, 0)
        data_layout.addWidget(self.isbn_text, 1, 1)
        scroll_layout.addWidget(data_group)

        # Combined Results
        combined_group = QGroupBox("🎯 Combined Results")
        combined_layout = QVBoxLayout(combined_group)
        
        self.combined_text = QTextEdit()
        self.combined_text.setReadOnly(True)
        self.combined_text.setFixedHeight(100)
        self.combined_text.setStyleSheet("QTextEdit { background-color: #fff3cd; border: 1px solid #ffeaa7; }")
        combined_layout.addWidget(self.combined_text)
        scroll_layout.addWidget(combined_group)

        # --- State ---
        self.front_image_np = None
        self.front_preprocessed_np = None
        self.back_image_np = None
        self.back_preprocessed_np = None
        self.front_ocr_result = None
        self.back_ocr_result = None
        self.metadata = None
        self.isbns = None
        self.current_mode = 'upload'

    def switch_mode(self, mode):
        """Switch between upload and camera modes"""
        if mode == self.current_mode:
            return
            
        self.current_mode = mode
        
        if mode == 'upload':
            self.upload_mode_btn.setChecked(True)
            self.camera_mode_btn.setChecked(False)
            self.input_group.setTitle("📷 Upload Book Covers")
            self.setup_upload_mode()
        else:
            self.upload_mode_btn.setChecked(False)
            self.camera_mode_btn.setChecked(True)
            self.input_group.setTitle("📷 Camera Capture")
            self.setup_camera_mode()

    def setup_upload_mode(self):
        """Setup the UI for file upload mode"""
        # Clear existing layout
        for i in reversed(range(self.input_layout.count())): 
            self.input_layout.itemAt(i).widget().setParent(None)
        
        # Front cover group
        front_group = QGroupBox("Front Cover")
        front_layout = QVBoxLayout(front_group)
        self.front_upload_btn = QPushButton("Upload Front Cover")
        self.front_upload_btn.clicked.connect(lambda: self.upload_image('front'))
        self.front_upload_btn.setStyleSheet("QPushButton { background-color: #3498db; color: white; padding: 8px; border-radius: 4px; }")
        front_layout.addWidget(self.front_upload_btn)
        
        # Front cover images
        front_images_layout = QHBoxLayout()
        self.front_original_label = QLabel("Original")
        self.front_original_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.front_original_label.setFixedSize(250, 350)
        self.front_original_label.setStyleSheet("border: 2px dashed #bdc3c7; background-color: #ecf0f1;")
        front_images_layout.addWidget(self.front_original_label)
        
        self.front_preprocessed_label = QLabel("Preprocessed")
        self.front_preprocessed_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.front_preprocessed_label.setFixedSize(250, 350)
        self.front_preprocessed_label.setStyleSheet("border: 2px dashed #bdc3c7; background-color: #ecf0f1;")
        front_images_layout.addWidget(self.front_preprocessed_label)
        front_layout.addLayout(front_images_layout)
        self.input_layout.addWidget(front_group)

        # Back cover group
        back_group = QGroupBox("Back Cover")
        back_layout = QVBoxLayout(back_group)
        self.back_upload_btn = QPushButton("Upload Back Cover")
        self.back_upload_btn.clicked.connect(lambda: self.upload_image('back'))
        self.back_upload_btn.setStyleSheet("QPushButton { background-color: #3498db; color: white; padding: 8px; border-radius: 4px; }")
        back_layout.addWidget(self.back_upload_btn)
        
        # Back cover images
        back_images_layout = QHBoxLayout()
        self.back_original_label = QLabel("Original")
        self.back_original_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.back_original_label.setFixedSize(250, 350)
        self.back_original_label.setStyleSheet("border: 2px dashed #bdc3c7; background-color: #ecf0f1;")
        back_images_layout.addWidget(self.back_original_label)
        
        self.back_preprocessed_label = QLabel("Preprocessed")
        self.back_preprocessed_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.back_preprocessed_label.setFixedSize(250, 350)
        self.back_preprocessed_label.setStyleSheet("border: 2px dashed #bdc3c7; background-color: #ecf0f1;")
        back_images_layout.addWidget(self.back_preprocessed_label)
        back_layout.addLayout(back_images_layout)
        self.input_layout.addWidget(back_group)

    def setup_camera_mode(self):
        """Setup the UI for camera capture mode"""
        # Clear existing layout
        for i in reversed(range(self.input_layout.count())): 
            self.input_layout.itemAt(i).widget().setParent(None)
        
        # Front cover camera
        front_group = QGroupBox("Front Cover")
        front_layout = QVBoxLayout(front_group)
        self.front_camera = CameraWidget(parent=front_group)
        front_layout.addWidget(self.front_camera)
        self.input_layout.addWidget(front_group)

        # Back cover camera
        back_group = QGroupBox("Back Cover")
        back_layout = QVBoxLayout(back_group)
        self.back_camera = CameraWidget(parent=back_group)
        back_layout.addWidget(self.back_camera)
        self.input_layout.addWidget(back_group)

        # Check if both covers are ready
        self.check_ready_to_process()

    def upload_image(self, which):
        file_path, _ = QFileDialog.getOpenFileName(
            self, f"Select {which.title()} Cover Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp)"
        )
        if not file_path:
            return
        image = cv2.imread(file_path)
        if image is None or image.size == 0:
            self._set_image_labels(which, failed=True)
            return
        self.process_image_data(image, which)
        # Check if both covers are ready
        self.check_ready_to_process()

    def process_image_data(self, image, which):
        """Process image data (from upload or camera)"""
        # Store the original full-size image for processing
        if which == 'front':
            self.front_image_np = image
        else:
            self.back_image_np = image
            
        # Display original (scaled for UI only)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        scaled_pixmap = pixmap.scaled(250, 350, Qt.AspectRatioMode.KeepAspectRatio)
        
        if which == 'front':
            if self.current_mode == 'upload':
                self.front_original_label.setPixmap(scaled_pixmap)
            else:
                self.front_camera.camera_label.setPixmap(scaled_pixmap)
        else:
            if self.current_mode == 'upload':
                self.back_original_label.setPixmap(scaled_pixmap)
            else:
                self.back_camera.camera_label.setPixmap(scaled_pixmap)
            
        # Preprocess using the original full-size image
        _, buffer = cv2.imencode('.jpg', image)
        preprocessed = preprocess_image(buffer.tobytes())
        
        # Store the preprocessed full-size image for processing
        if which == 'front':
            self.front_preprocessed_np = preprocessed
        else:
            self.back_preprocessed_np = preprocessed
            
        # Display preprocessed (scaled for UI only)
        if len(preprocessed.shape) == 2:
            preprocessed_rgb = cv2.cvtColor(preprocessed, cv2.COLOR_GRAY2RGB)
        else:
            preprocessed_rgb = cv2.cvtColor(preprocessed, cv2.COLOR_BGR2RGB)
            
        h2, w2, ch2 = preprocessed_rgb.shape
        bytes_per_line2 = ch2 * w2
        qt_image2 = QImage(preprocessed_rgb.data, w2, h2, bytes_per_line2, QImage.Format.Format_RGB888)
        pixmap2 = QPixmap.fromImage(qt_image2)
        scaled_pixmap2 = pixmap2.scaled(250, 350, Qt.AspectRatioMode.KeepAspectRatio)
        
        if which == 'front':
            if self.current_mode == 'upload':
                self.front_preprocessed_label.setPixmap(scaled_pixmap2)
            else:
                # In camera mode, we don't show preprocessed image
                pass
        else:
            if self.current_mode == 'upload':
                self.back_preprocessed_label.setPixmap(scaled_pixmap2)
            else:
                # In camera mode, we don't show preprocessed image
                pass
            
        # Enable process button if both images are loaded
        self.check_ready_to_process()

    def _set_image_labels(self, which, failed=False):
        if which == 'front':
            if self.current_mode == 'upload':
                self.front_original_label.setText("Failed to load image." if failed else "Original")
                self.front_preprocessed_label.setText("")
            else:
                self.front_camera.camera_label.setText("Failed to load image." if failed else "Camera Preview")
            self.front_image_np = None
            self.front_preprocessed_np = None
        else:
            if self.current_mode == 'upload':
                self.back_original_label.setText("Failed to load image." if failed else "Original")
                self.back_preprocessed_label.setText("")
            else:
                self.back_camera.camera_label.setText("Failed to load image." if failed else "Camera Preview")
            self.back_image_np = None
            self.back_preprocessed_np = None
        self.check_ready_to_process()

    def process_both(self):
        # Check if we have images from camera
        if self.current_mode == 'camera':
            if self.front_camera.captured_image is not None:
                self.process_image_data(self.front_camera.captured_image, 'front')
            if self.back_camera.captured_image is not None:
                self.process_image_data(self.back_camera.captured_image, 'back')
        
        # Disable button during processing
        self.process_btn.setEnabled(False)
        self.process_btn.setText("Processing...")
        
        try:
            # --- Process Front Cover ---
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmpf:
                cv2.imwrite(tmpf.name, self.front_preprocessed_np)  # Use preprocessed
                front_path = tmpf.name
            front_ocr = extract_text_with_confidence(front_path)
            self.front_ocr_result = front_ocr
            front_text = front_ocr.get('text', '')
            front_conf = front_ocr.get('confidence', 0.0)
            self.front_ocr_text.setPlainText(f"Confidence: {front_conf:.2f}\n\n{front_text}")

            # --- Process Back Cover ---
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmpb:
                cv2.imwrite(tmpb.name, self.back_preprocessed_np)  # Use preprocessed
                back_path = tmpb.name
            back_ocr = extract_text_with_confidence(back_path)
            self.back_ocr_result = back_ocr
            back_text = back_ocr.get('text', '')
            back_conf = back_ocr.get('confidence', 0.0)
            self.back_ocr_text.setPlainText(f"Confidence: {back_conf:.2f}\n\n{back_text}")

            # --- Metadata Extraction (Front) ---
            if front_text:
                metadata = extract_metadata_with_gemini(front_text)
                self.metadata = metadata
                if metadata:
                    title = metadata.get('title', 'Not found')
                    authors = ', '.join(metadata.get('authors', [])) if metadata.get('authors') else 'Not found'
                    self.metadata_text.setPlainText(f"Title: {title}\nAuthors: {authors}")
                else:
                    self.metadata_text.setPlainText("No metadata found.")
            else:
                self.metadata_text.setPlainText("No text to extract metadata.")

            # --- ISBN Extraction (Back) ---
            if back_text:
                isbns = extract_and_validate_isbns(back_text)
                self.isbns = isbns
                if isbns and 'isbns' in isbns and isbns['isbns']:
                    isbn_list = [isbn_data['isbn'] for isbn_data in isbns['isbns']]
                    self.isbn_text.setPlainText("\n".join(isbn_list))
                else:
                    self.isbn_text.setPlainText("No ISBNs found.")
            else:
                self.isbn_text.setPlainText("No text to extract ISBNs.")

            # --- Combine Results ---
            if self.metadata and self.isbns and 'isbns' in self.isbns:
                isbn_list = [isbn_data['isbn'] for isbn_data in self.isbns['isbns']]
                combined = metadata_combiner(self.metadata, isbn_list)
                self.combined_text.setPlainText(str(combined))
            else:
                self.combined_text.setPlainText("Not enough data to combine results.")
                
        except Exception as e:
            self.combined_text.setPlainText(f"Error during processing: {str(e)}")
        finally:
            # Re-enable button
            self.process_btn.setEnabled(True)
            self.process_btn.setText("🚀 Process Both Covers")

    def check_ready_to_process(self):
        """Enable process button only if both covers are ready (uploaded or captured)"""
        if self.current_mode == 'upload':
            ready = self.front_image_np is not None and self.back_image_np is not None
        else:
            ready = (self.front_camera.captured_image is not None and self.back_camera.captured_image is not None)
        self.process_btn.setEnabled(ready)

def main():
    app = QApplication(sys.argv)
    window = BookAcquisitionApp()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()