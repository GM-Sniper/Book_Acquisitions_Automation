import sys
import os
import platform
import threading
import tempfile
from typing import Optional

# Add the project root to sys.path so 'src' can be imported
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
    QLabel, QTextEdit, QGroupBox, QSizePolicy, QProgressBar, QMessageBox, QComboBox
)
from PyQt6.QtGui import QPixmap, QImage, QFont
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QThread

import cv2
import numpy as np
from src.vision.preprocessing import preprocess_image
from src.vision.OCR_Processing import extract_text_with_confidence
from src.metadata.metadata_extraction import extract_metadata_with_gemini, metadata_combiner
from src.utils.isbn_detection import extract_and_validate_isbns

class ProcessingThread(QThread):
    """Background thread for processing images"""
    processing_complete = pyqtSignal(dict)
    processing_error = pyqtSignal(str)
    progress_update = pyqtSignal(int)
    
    def __init__(self, front_image, back_image):
        super().__init__()
        self.front_image = front_image
        self.back_image = back_image
        
    def run(self):
        try:
            # Step 1: Preprocess front cover (10%)
            self.progress_update.emit(10)
            # Convert numpy array to bytes for preprocessing
            _, front_encoded = cv2.imencode('.jpg', self.front_image)
            front_bytes = front_encoded.tobytes()
            front_processed = preprocess_image(front_bytes)
            
            # Step 2: Preprocess back cover (20%)
            self.progress_update.emit(20)
            # Convert numpy array to bytes for preprocessing
            _, back_encoded = cv2.imencode('.jpg', self.back_image)
            back_bytes = back_encoded.tobytes()
            back_processed = preprocess_image(back_bytes)
            
            # Step 3: Extract text from front cover (30%)
            self.progress_update.emit(30)
            # Save processed image to temp file
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_front:
                cv2.imwrite(temp_front.name, front_processed)
                front_result = extract_text_with_confidence(temp_front.name)
            # Clean up temp file
            os.unlink(temp_front.name)
            
            # Step 4: Extract text from back cover (40%)
            self.progress_update.emit(40)
            # Save processed image to temp file
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_back:
                cv2.imwrite(temp_back.name, back_processed)
                back_result = extract_text_with_confidence(temp_back.name)
            # Clean up temp file
            os.unlink(temp_back.name)
            
            # Step 5: Extract metadata from front cover (50%)
            self.progress_update.emit(50)
            front_metadata = extract_metadata_with_gemini(front_result["text"])
            
            # Step 6: Extract ISBNs from back cover (70%)
            self.progress_update.emit(70)
            isbns = extract_and_validate_isbns(back_result["text"])
            
            # Step 7: Combine metadata with confidence info (90%)
            self.progress_update.emit(90)
            final_metadata = metadata_combiner(front_metadata, isbns)
            # Add confidence information
            final_metadata["front_confidence"] = front_result.get("confidence", 0.0)
            final_metadata["back_confidence"] = back_result.get("confidence", 0.0)
            final_metadata["front_word_count"] = front_result.get("word_count", 0)
            final_metadata["back_word_count"] = back_result.get("word_count", 0)
            
            # Step 8: Complete (100%)
            self.progress_update.emit(100)
            self.processing_complete.emit(final_metadata)
            
        except Exception as e:
            self.processing_error.emit(str(e))

class CameraWidget(QWidget):
    """Simple camera widget for live preview and capture"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.camera = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.captured_image = None
        self.selected_camera_index = 0
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout()
        
        # Camera selection dropdown
        self.camera_select = QComboBox()
        self.camera_select.setFixedWidth(180)
        self.camera_select.addItems(self.get_camera_list())
        self.camera_select.currentIndexChanged.connect(self.change_camera_index)
        layout.addWidget(self.camera_select)
        
        # Camera preview
        self.camera_label = QLabel("Camera Preview")
        self.camera_label.setMinimumSize(400, 300)
        self.camera_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.camera_label.setStyleSheet("border: 2px dashed #bdc3c7; background-color: #ecf0f1;")
        
        # Camera controls
        controls_layout = QHBoxLayout()
        self.start_btn = QPushButton("Start Camera")
        self.start_btn.clicked.connect(self.start_camera)
        self.start_btn.setStyleSheet("QPushButton { background-color: #27ae60; color: white; padding: 8px; border-radius: 4px; }")
        
        self.capture_btn = QPushButton("Capture")
        self.capture_btn.clicked.connect(self.capture_image)
        self.capture_btn.setEnabled(False)
        self.capture_btn.setStyleSheet("QPushButton { background-color: #3498db; color: white; padding: 8px; border-radius: 4px; }")
        
        self.stop_btn = QPushButton("Stop Camera")
        self.stop_btn.clicked.connect(self.stop_camera)
        self.stop_btn.setEnabled(False)
        self.stop_btn.setStyleSheet("QPushButton { background-color: #e74c3c; color: white; padding: 8px; border-radius: 4px; }")
        
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
            QMessageBox.warning(self, "Camera Error", "Could not open camera")

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
                return frame
        return None

class CleanBookAcquisitionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Book Acquisition Tool - Clean Version")
        self.setGeometry(200, 100, 1200, 800)
        
        # Store captured images
        self.front_image = None
        self.back_image = None
        self.processing_thread = None
        
        self.setup_ui()

    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Title
        title_label = QLabel("📚 Book Acquisition Tool")
        title_label.setFont(QFont("Arial", 24, QFont.Weight.Bold))
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setStyleSheet("color: #2c3e50; margin: 20px;")
        main_layout.addWidget(title_label)
        
        # Main content area
        content_layout = QHBoxLayout()
        
        # Left side - Camera and Capture
        left_panel = QVBoxLayout()
        
        # Camera widget
        camera_group = QGroupBox("📷 Camera Capture")
        camera_layout = QVBoxLayout(camera_group)
        self.camera_widget = CameraWidget()
        camera_layout.addWidget(self.camera_widget)
        left_panel.addWidget(camera_group)
        
        # Capture instructions
        instructions_group = QGroupBox("📋 Instructions")
        instructions_layout = QVBoxLayout(instructions_group)
        instructions_text = QLabel(
            "1. Start the camera\n"
            "2. Capture the FRONT cover of the book\n"
            "3. Capture the BACK cover of the book\n"
            "4. Processing will start automatically"
        )
        instructions_text.setStyleSheet("font-size: 14px; padding: 10px;")
        instructions_layout.addWidget(instructions_text)
        left_panel.addWidget(instructions_group)
        
        # Capture buttons
        capture_group = QGroupBox("📸 Capture Controls")
        capture_layout = QHBoxLayout(capture_group)
        
        self.capture_front_btn = QPushButton("Capture Front Cover")
        self.capture_front_btn.clicked.connect(self.capture_front)
        self.capture_front_btn.setStyleSheet("QPushButton { background-color: #3498db; color: white; padding: 10px; border-radius: 6px; font-size: 14px; }")
        
        self.capture_back_btn = QPushButton("Capture Back Cover")
        self.capture_back_btn.clicked.connect(self.capture_back)
        self.capture_back_btn.setStyleSheet("QPushButton { background-color: #e67e22; color: white; padding: 10px; border-radius: 6px; font-size: 14px; }")
        
        capture_layout.addWidget(self.capture_front_btn)
        capture_layout.addWidget(self.capture_back_btn)
        left_panel.addWidget(capture_group)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setStyleSheet("QProgressBar { border: 2px solid grey; border-radius: 5px; text-align: center; } QProgressBar::chunk { background-color: #3498db; border-radius: 3px; }")
        left_panel.addWidget(self.progress_bar)
        
        content_layout.addLayout(left_panel)
        
        # Right side - Results
        right_panel = QVBoxLayout()
        
        # Status
        self.status_label = QLabel("Ready to capture")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setStyleSheet("font-size: 16px; color: #27ae60; padding: 10px; background-color: #ecf0f1; border-radius: 5px;")
        right_panel.addWidget(self.status_label)
        
        # Results
        results_group = QGroupBox("📖 Extracted Metadata")
        results_layout = QVBoxLayout(results_group)
        self.results_text = QTextEdit()
        self.results_text.setPlaceholderText("Metadata will appear here after processing...")
        self.results_text.setStyleSheet("font-family: 'Courier New'; font-size: 12px;")
        results_layout.addWidget(self.results_text)
        right_panel.addWidget(results_group)
        
        content_layout.addLayout(right_panel)
        
        main_layout.addLayout(content_layout)

    def capture_front(self):
        if not self.camera_widget.camera or not self.camera_widget.camera.isOpened():
            QMessageBox.warning(self, "Camera Error", "Please start the camera first")
            return
            
        self.front_image = self.camera_widget.capture_image()
        if self.front_image is not None:
            self.status_label.setText("Front cover captured ✓")
            self.status_label.setStyleSheet("font-size: 16px; color: #27ae60; padding: 10px; background-color: #d5f4e6; border-radius: 5px;")
            self.check_ready_to_process()

    def capture_back(self):
        if not self.camera_widget.camera or not self.camera_widget.camera.isOpened():
            QMessageBox.warning(self, "Camera Error", "Please start the camera first")
            return
            
        self.back_image = self.camera_widget.capture_image()
        if self.back_image is not None:
            self.status_label.setText("Back cover captured ✓")
            self.status_label.setStyleSheet("font-size: 16px; color: #27ae60; padding: 10px; background-color: #d5f4e6; border-radius: 5px;")
            self.check_ready_to_process()

    def check_ready_to_process(self):
        if self.front_image is not None and self.back_image is not None:
            self.status_label.setText("Processing... Please wait")
            self.status_label.setStyleSheet("font-size: 16px; color: #f39c12; padding: 10px; background-color: #fef9e7; border-radius: 5px;")
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)
            self.start_processing()

    def start_processing(self):
        # Disable capture buttons during processing
        self.capture_front_btn.setEnabled(False)
        self.capture_back_btn.setEnabled(False)
        
        # Start background processing
        self.processing_thread = ProcessingThread(self.front_image, self.back_image)
        self.processing_thread.processing_complete.connect(self.on_processing_complete)
        self.processing_thread.processing_error.connect(self.on_processing_error)
        self.processing_thread.progress_update.connect(self.progress_bar.setValue)
        self.processing_thread.start()

    def on_processing_complete(self, metadata):
        self.progress_bar.setVisible(False)
        self.capture_front_btn.setEnabled(True)
        self.capture_back_btn.setEnabled(True)
        
        self.status_label.setText("Processing complete ✓")
        self.status_label.setStyleSheet("font-size: 16px; color: #27ae60; padding: 10px; background-color: #d5f4e6; border-radius: 5px;")
        
        # Display results
        if metadata:
            result_text = "📚 Book Metadata:\n\n"
            if metadata.get('title'):
                result_text += f"Title: {metadata['title']}\n"
            if metadata.get('authors'):
                result_text += f"Authors: {', '.join(metadata['authors'])}\n"
            if metadata.get('isbns'):
                result_text += f"ISBNs: {', '.join(metadata['isbns'])}\n"
            
            # Add confidence information with visual indicators
            result_text += "\n📊 OCR Confidence:\n"
            
            # Front cover confidence
            front_conf = metadata.get('front_confidence', 0.0)
            front_words = metadata.get('front_word_count', 0)
            front_indicator = self.get_confidence_indicator(front_conf)
            result_text += f"Front Cover: {front_indicator} {front_conf:.1%} ({front_words} words)\n"
            
            # Back cover confidence
            back_conf = metadata.get('back_confidence', 0.0)
            back_words = metadata.get('back_word_count', 0)
            back_indicator = self.get_confidence_indicator(back_conf)
            result_text += f"Back Cover: {back_indicator} {back_conf:.1%} ({back_words} words)\n"
            
            # Overall assessment
            avg_confidence = (front_conf + back_conf) / 2
            overall_indicator = self.get_confidence_indicator(avg_confidence)
            result_text += f"\nOverall: {overall_indicator} {avg_confidence:.1%}"
            
        else:
            result_text = "No metadata could be extracted. Please try again with clearer images."
        
        self.results_text.setText(result_text)

    def on_processing_error(self, error_message):
        self.progress_bar.setVisible(False)
        self.capture_front_btn.setEnabled(True)
        self.capture_back_btn.setEnabled(True)
        
        self.status_label.setText("Processing failed ✗")
        self.status_label.setStyleSheet("font-size: 16px; color: #e74c3c; padding: 10px; background-color: #fadbd8; border-radius: 5px;")
        
        QMessageBox.critical(self, "Processing Error", f"An error occurred during processing:\n{error_message}")

    def get_confidence_indicator(self, confidence):
        """Get visual indicator for confidence level"""
        if confidence >= 0.8:
            return "🟢"  # Green - High confidence
        elif confidence >= 0.6:
            return "🟡"  # Yellow - Medium confidence
        elif confidence >= 0.4:
            return "🟠"  # Orange - Low confidence
        else:
            return "🔴"  # Red - Very low confidence

def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # Modern look
    
    window = CleanBookAcquisitionApp()
    window.show()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main() 