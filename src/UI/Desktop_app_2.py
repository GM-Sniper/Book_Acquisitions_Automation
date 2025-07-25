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
    QLabel, QTextEdit, QGroupBox, QSizePolicy, QProgressBar, QMessageBox, QComboBox,
    QFrame, QScrollArea, QGridLayout, QSpacerItem
)
from PyQt6.QtGui import QPixmap, QImage, QFont, QPalette, QColor, QIcon, QPainter, QLinearGradient
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QThread, QPropertyAnimation, QEasingCurve, QRect, QPropertyAnimation, QParallelAnimationGroup

import cv2
import numpy as np
from src.vision.preprocessing import preprocess_image
from src.vision.OCR_Processing import extract_text_with_confidence
from src.metadata.metadata_extraction import extract_metadata_with_gemini, metadata_combiner
from src.utils.isbn_detection import extract_and_validate_isbns
from src.utils.database_cloud import create_table, insert_book, search_book

from src.vision.gemini_processing import process_book_images

class ModernButton(QPushButton):
    """Custom modern button with hover effects and animations (dark mode)"""
    def __init__(self, text, color="#1976d2", hover_color="#1565c0", icon=None):
        super().__init__(text)
        self.color = color
        self.hover_color = hover_color
        self.setFixedHeight(40)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setStyleSheet(f"""
            QPushButton {{
                background-color: {color};
                color: #fff;
                border: none;
                border-radius: 8px;
                font-size: 14px;
                font-weight: 600;
                padding: 10px 18px;
                text-align: center;
            }}
            QPushButton:hover {{
                background-color: {hover_color};
                filter: brightness(1.1);
            }}
            QPushButton:pressed {{
                background-color: {hover_color};
            }}
            QPushButton:disabled {{
                background-color: #333;
                color: #888;
            }}
        """)
        if icon:
            self.setIcon(icon)

class StatusCard(QFrame):
    """Modern status card with animations (dark mode)"""
    def __init__(self, title, status="ready", parent=None):
        super().__init__(parent)
        self.title = title
        self.status = status
        self.setup_ui()
    def setup_ui(self):
        self.setFrameStyle(QFrame.Shape.StyledPanel)
        self.setStyleSheet("""
            QFrame {
                background-color: #23272e;
                border-radius: 12px;
                border: 2px solid #333;
            }
        """)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 10, 16, 10)
        title_label = QLabel(self.title)
        title_label.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        title_label.setStyleSheet("color: #fff; margin-bottom: 2px;")
        self.status_label = QLabel("Ready")
        self.status_label.setFont(QFont("Segoe UI", 10))
        self.status_label.setStyleSheet("color: #4caf50; font-weight: 600;")
        layout.addWidget(title_label)
        layout.addWidget(self.status_label)
    def update_status(self, status, color="#4caf50"):
        self.status = status
        self.status_label.setText(status)
        self.status_label.setStyleSheet(f"color: {color}; font-weight: 600;")
        self.animate_status_change()
    def animate_status_change(self):
        animation = QPropertyAnimation(self, b"geometry")
        animation.setDuration(200)
        animation.setStartValue(self.geometry())
        animation.setEndValue(self.geometry())
        animation.setEasingCurve(QEasingCurve.Type.OutCubic)
        animation.start()

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

class ModernCameraWidget(QWidget):
    """Modern camera widget with sleek dark mode design"""
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
        layout.setContentsMargins(0, 0, 0, 0)
        camera_select_frame = QFrame()
        camera_select_frame.setStyleSheet("""
            QFrame {
                background-color: #23272e;
                border-radius: 8px;
                padding: 8px;
            }
        """)
        camera_select_layout = QHBoxLayout(camera_select_frame)
        camera_label = QLabel("📷 Camera:")
        camera_label.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        camera_label.setStyleSheet("color: #fff;")
        self.camera_select = QComboBox()
        self.camera_select.setFixedWidth(200)
        self.camera_select.addItems(self.get_camera_list())
        self.camera_select.currentIndexChanged.connect(self.change_camera_index)
        self.camera_select.setStyleSheet("""
            QComboBox {
                border: 2px solid #333;
                border-radius: 6px;
                padding: 7px 12px;
                background-color: #181c20;
                color: #fff;
                font-size: 12px;
            }
            QComboBox QAbstractItemView {
                background: #23272e;
                color: #fff;
            }
        """)
        camera_select_layout.addWidget(camera_label)
        camera_select_layout.addWidget(self.camera_select)
        camera_select_layout.addStretch()
        layout.addWidget(camera_select_frame)
        self.camera_label = QLabel("📷 Camera Preview")
        self.camera_label.setMinimumSize(400, 250)
        self.camera_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.camera_label.setStyleSheet("""
            QLabel {
                border: 2px dashed #444;
                border-radius: 12px;
                background: #181c20;
                color: #888;
                font-size: 15px;
                font-weight: 600;
            }
        """)
        layout.addWidget(self.camera_label)
        controls_frame = QFrame()
        controls_frame.setStyleSheet("""
            QFrame {
                background-color: #23272e;
                border-radius: 10px;
                border: 1px solid #333;
            }
        """)
        controls_layout = QHBoxLayout(controls_frame)
        controls_layout.setContentsMargins(12, 10, 12, 10)
        self.start_btn = ModernButton("▶ Start Camera", "#388e3c", "#2e7031")
        self.start_btn.clicked.connect(self.start_camera)
        self.capture_btn = ModernButton("📸 Capture", "#1976d2", "#1565c0")
        self.capture_btn.clicked.connect(self.capture_image)
        self.capture_btn.setEnabled(False)
        self.stop_btn = ModernButton("⏹ Stop Camera", "#d32f2f", "#b71c1c")
        self.stop_btn.clicked.connect(self.stop_camera)
        self.stop_btn.setEnabled(False)
        controls_layout.addWidget(self.start_btn)
        controls_layout.addWidget(self.capture_btn)
        controls_layout.addWidget(self.stop_btn)
        controls_layout.addStretch()
        layout.addWidget(controls_frame)
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
            
            # Add success animation
            self.animate_button_success(self.start_btn)
        else:
            QMessageBox.warning(self, "Camera Error", "Could not open camera")

    def stop_camera(self):
        if self.camera:
            self.timer.stop()
            self.camera.release()
            self.camera = None
            self.camera_label.clear()
            self.camera_label.setText("📷 Camera Preview")
            self.camera_label.setStyleSheet("""
                QLabel {
                    border: 2px dashed #444;
                    border-radius: 12px;
                    background: #181c20;
                    color: #888;
                    font-size: 15px;
                    font-weight: 600;
                }
            """)
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
                self.camera_label.setStyleSheet("border: 2px solid #388e3c; border-radius: 12px;")

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
                self.camera_label.setText("✅ Image Captured!")
                self.camera_label.setStyleSheet("""
                    QLabel {
                        border: 2px solid #388e3c;
                        border-radius: 12px;
                        background-color: #2e7031;
                        color: #fff;
                        font-size: 15px;
                        font-weight: 600;
                    }
                """)
                
                # Add capture animation
                self.animate_button_success(self.capture_btn)
                return frame
        return None

    def animate_button_success(self, button):
        """Animate button to show success"""
        animation = QPropertyAnimation(button, b"geometry")
        animation.setDuration(150)
        original_geometry = button.geometry()
        animation.setStartValue(original_geometry)
        animation.setEndValue(original_geometry)
        animation.setEasingCurve(QEasingCurve.Type.OutBounce)
        animation.start()

class ModernBookAcquisitionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("📚 Book Acquisition Tool - Modern Edition")
        self.setGeometry(100, 50, 1400, 900)
        # Store captured images (list)
        self.captured_images = []
        self.processing_thread = None
        # Setup modern styling
        self.setup_styles()
        self.setup_ui()

    def setup_styles(self):
        self.setStyleSheet("""
            QMainWindow {
                background: #181c20;
            }
            QGroupBox {
                font-weight: bold;
                font-size: 13px;
                color: #fff;
                border: 2px solid #333;
                border-radius: 10px;
                margin-top: 10px;
                padding-top: 10px;
                background-color: #23272e;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 8px 0 8px;
            }
            QTextEdit {
                border: 2px solid #333;
                border-radius: 8px;
                padding: 10px;
                background-color: #181c20;
                color: #fff;
                font-family: 'Segoe UI', 'Consolas', monospace;
                font-size: 12px;
            }
            QProgressBar {
                border: 2px solid #333;
                border-radius: 8px;
                text-align: center;
                font-weight: bold;
                background-color: #23272e;
                color: #fff;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #1976d2, stop:1 #1565c0);
                border-radius: 6px;
            }
            QLabel {
                color: #fff;
            }
        """)

    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(12)
        header_frame = QFrame()
        header_frame.setStyleSheet("""
            QFrame {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #1976d2, stop:1 #1565c0);
                border-radius: 12px;
                padding: 12px;
            }
        """)
        header_layout = QVBoxLayout(header_frame)
        title_label = QLabel("📚 Book Acquisition Tool")
        title_label.setFont(QFont("Segoe UI", 19, QFont.Weight.Bold))
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setStyleSheet("color: #fff; margin: 2px;")
        subtitle_label = QLabel("Modern AI-Powered Book Metadata Extraction")
        subtitle_label.setFont(QFont("Segoe UI", 11))
        subtitle_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle_label.setStyleSheet("color: #b3c6e0; margin: 0px;")
        header_layout.addWidget(title_label)
        header_layout.addWidget(subtitle_label)
        main_layout.addWidget(header_frame)
        content_layout = QHBoxLayout()
        content_layout.setSpacing(12)
        left_panel = QVBoxLayout()
        left_panel.setSpacing(10)
        camera_group = QGroupBox("📷 Camera Capture")
        camera_layout = QVBoxLayout(camera_group)
        self.camera_widget = ModernCameraWidget()
        camera_layout.addWidget(self.camera_widget)
        left_panel.addWidget(camera_group)
        # Multi-image capture status
        status_group = QGroupBox("📊 Capture Status")
        status_layout = QVBoxLayout(status_group)
        self.captured_count_label = QLabel("Captured Images: 0")
        self.captured_count_label.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        self.captured_count_label.setStyleSheet("color: #fff; margin-bottom: 2px;")
        status_layout.addWidget(self.captured_count_label)
        self.captured_list_label = QLabel("")
        self.captured_list_label.setFont(QFont("Segoe UI", 10))
        self.captured_list_label.setStyleSheet("color: #b3c6e0;")
        status_layout.addWidget(self.captured_list_label)
        left_panel.addWidget(status_group)
        # Single capture button
        capture_group = QGroupBox("📸 Capture Controls")
        capture_layout = QHBoxLayout(capture_group)
        self.capture_image_btn = ModernButton("📸 Capture Image", "#1976d2", "#1565c0")
        self.capture_image_btn.clicked.connect(self.capture_image)
        capture_layout.addWidget(self.capture_image_btn)
        # Process button
        self.process_btn = ModernButton("⚡ Process Images", "#388e3c", "#2e7031")
        self.process_btn.setEnabled(False)
        self.process_btn.clicked.connect(self.start_processing)
        capture_layout.addWidget(self.process_btn)
        left_panel.addWidget(capture_group)
        progress_group = QGroupBox("⚡ Processing Progress")
        progress_layout = QVBoxLayout(progress_group)
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setFixedHeight(22)
        self.progress_label = QLabel("Ready to process")
        self.progress_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.progress_label.setStyleSheet("color: #b3c6e0; font-size: 12px; margin-top: 2px;")
        progress_layout.addWidget(self.progress_bar)
        progress_layout.addWidget(self.progress_label)
        left_panel.addWidget(progress_group)
        content_layout.addLayout(left_panel, 2)
        right_panel = QVBoxLayout()
        right_panel.setSpacing(10)
        results_group = QGroupBox("📖 Extracted Metadata")
        results_layout = QVBoxLayout(results_group)
        self.results_text = QTextEdit()
        self.results_text.setPlaceholderText("📋 Metadata will appear here after processing...\n\n💡 Tips:\n• Ensure good lighting for better OCR results\n• Hold the book steady during capture\n• Include the entire cover in the frame")
        self.results_text.setMinimumHeight(300)
        self.results_text.setReadOnly(True)
        results_layout.addWidget(self.results_text)
        right_panel.addWidget(results_group)
        confidence_group = QGroupBox("📊 Confidence Analysis")
        confidence_layout = QVBoxLayout(confidence_group)
        self.confidence_text = QLabel("No data available")
        self.confidence_text.setStyleSheet("""
            QLabel {
                padding: 10px;
                background-color: #181c20;
                border-radius: 8px;
                color: #b3c6e0;
                font-size: 12px;
            }
        """)
        self.confidence_text.setWordWrap(True)
        confidence_layout.addWidget(self.confidence_text)
        right_panel.addWidget(confidence_group)
        content_layout.addLayout(right_panel, 3)
        main_layout.addLayout(content_layout)

    def capture_image(self):
        if not self.camera_widget.camera or not self.camera_widget.camera.isOpened():
            QMessageBox.warning(self, "Camera Error", "Please start the camera first")
            return
        image = self.camera_widget.capture_image()
        if image is not None:
            self.captured_images.append(image)
            self.update_capture_status()
            self.process_btn.setEnabled(True)

    def update_capture_status(self):
        count = len(self.captured_images)
        self.captured_count_label.setText(f"Captured Images: {count}")
        if count > 0:
            self.captured_list_label.setText(", ".join([f"Captured {i+1}" for i in range(count)]))
        else:
            self.captured_list_label.setText("")
        if count == 0:
            self.process_btn.setEnabled(False)

    def start_processing(self):
        if not self.captured_images:
            QMessageBox.warning(self, "No Images", "Please capture at least one image before processing.")
            return
        # Disable buttons during processing
        self.capture_image_btn.setEnabled(False)
        self.process_btn.setEnabled(False)
        self.progress_label.setText("Processing... Please wait")
        self.progress_label.setStyleSheet("color: #fd7e14; font-size: 12px; font-weight: 600; margin-top: 5px;")
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        # Start background processing
        self.processing_thread = GeminiProcessingThread(self.captured_images)
        self.processing_thread.processing_complete.connect(self.on_processing_complete)
        self.processing_thread.processing_error.connect(self.on_processing_error)
        self.processing_thread.progress_update.connect(self.progress_bar.setValue)
        self.processing_thread.start()

    def on_processing_complete(self, metadata):
        self.progress_bar.setVisible(False)
        self.capture_image_btn.setEnabled(True)
        self.process_btn.setEnabled(True)
        self.progress_label.setText("Processing complete ✓")
        self.progress_label.setStyleSheet("color: #28a745; font-size: 12px; font-weight: 600; margin-top: 5px;")
        # Display results
        if metadata:
            result_text = "📚 Book Metadata\n" + "="*50 + "\n\n"
            if metadata.get('title'):
                result_text += f"📖 Title: {metadata['title']}\n\n"
            if metadata.get('authors'):
                result_text += f"✍️ Authors: {', '.join(metadata['authors'])}\n\n"
            if metadata.get('isbn'):
                result_text += f"🔢 ISBN: {metadata['isbn']}\n\n"
            if metadata.get('isbn10'):
                result_text += f"🔢 ISBN-10: {metadata['isbn10']}\n\n"
            if metadata.get('isbn13'):
                result_text += f"🔢 ISBN-13: {metadata['isbn13']}\n\n"
            if metadata.get('publisher'):
                result_text += f"🏢 Publisher: {metadata['publisher']}\n\n"
            if metadata.get('year'):
                result_text += f"📅 Year: {metadata['year']}\n\n"
            if metadata.get('edition'):
                result_text += f"📚 Edition: {metadata['edition']}\n\n"
            if metadata.get('series'):
                result_text += f"🔗 Series: {metadata['series']}\n\n"
            if metadata.get('genre'):
                result_text += f"🏷️ Genre: {metadata['genre']}\n\n"
            if metadata.get('language'):
                result_text += f"🌐 Language: {metadata['language']}\n\n"
            if metadata.get('additional_text'):
                result_text += f"📝 Additional Text: {metadata['additional_text']}\n\n"
            self.results_text.setText(result_text)
            # Confidence display (Gemini doesn't provide a confidence score, so just show a message)
            confidence_html = f"""
            <div style=\"padding: 15px; background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); border-radius: 8px;\">
                <h3 style=\"color: #2c3e50; margin: 0 0 10px 0;\">Gemini Vision Analysis</h3>
                <div style=\"text-align: center; margin-top: 15px;\">
                    <strong>Processed {len(self.captured_images)} image(s) with Gemini Vision</strong>
                </div>
            </div>
            """
            self.confidence_text.setText(confidence_html)
            # Check if book exists in cloud DB
            existing = search_book(isbn=metadata.get('isbn'), title=metadata.get('title'), authors=metadata.get('authors'))

            if existing:
                QMessageBox.information(self, "Book Exists", "✅ This book already exists in the database.")
            else:
                insert_book(metadata)
                QMessageBox.information(self, "Book Added", "📚 Book metadata saved to the cloud database.")

        else:
            result_text = "❌ No metadata could be extracted.\n\n💡 Suggestions:\n• Ensure good lighting\n• Hold the book steady\n• Include the entire cover\n• Try different angles"
            self.confidence_text.setText("No confidence data available")
            self.results_text.setText(result_text)

    def on_processing_error(self, error_message):
        self.progress_bar.setVisible(False)
        self.capture_image_btn.setEnabled(True)
        self.process_btn.setEnabled(True)
        self.progress_label.setText("Processing failed ✗")
        self.progress_label.setStyleSheet("color: #dc3545; font-size: 12px; font-weight: 600; margin-top: 5px;")
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

# New GeminiProcessingThread for Gemini-based processing
from PyQt6.QtCore import QThread, pyqtSignal
class GeminiProcessingThread(QThread):
    processing_complete = pyqtSignal(dict)
    processing_error = pyqtSignal(str)
    progress_update = pyqtSignal(int)
    def __init__(self, image_list):
        super().__init__()
        self.image_list = image_list
    def run(self):
        try:
            self.progress_update.emit(10)
            # Use Gemini processing pipeline
            metadata = process_book_images(self.image_list, prompt_type="comprehensive", infer_missing=True)
            self.progress_update.emit(100)
            self.processing_complete.emit(metadata)
        except Exception as e:
            self.processing_error.emit(str(e))

def main():
    create_table()
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # Modern look
    
    # Set application icon and properties
    app.setApplicationName("Book Acquisition Tool")
    app.setApplicationVersion("2.0")
    
    window = ModernBookAcquisitionApp()
    window.show()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main() 