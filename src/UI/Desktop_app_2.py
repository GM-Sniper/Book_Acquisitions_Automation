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
    QFrame, QScrollArea, QGridLayout, QSpacerItem, QTabWidget, QDialog, QLineEdit,
    QFormLayout, QDialogButtonBox, QTextEdit, QCheckBox, QSpinBox, QDateEdit
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
from src.metadata.llm_metadata_combiner import llm_metadata_combiner
from src.utils.google_books import search_book_by_isbn, search_book_by_title_author, extract_book_metadata
from src.utils.openlibrary import OpenLibraryAPI
from src.utils.LOC import LOCConverter
from src.utils.isbnlib_service import ISBNService

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
        self.stop_btn = ModernButton("⏹ Stop Camera", "#d32f2f", "#b71c1c")
        self.stop_btn.clicked.connect(self.stop_camera)
        self.stop_btn.setEnabled(False)
        controls_layout.addWidget(self.start_btn)
        controls_layout.addWidget(self.stop_btn)
        controls_layout.addStretch()
        layout.addWidget(controls_frame)
        self.setLayout(layout)

    def get_camera_list(self):
        # Try up to 5 camera indices, only include if a frame can be read
        available = []
        for i in range(5):
            try:
                # Try different backends on Windows
                if platform.system() == 'Windows':
                    # Try DirectShow first
                    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
                    if not cap.isOpened():
                        # Try MSMF backend
                        cap = cv2.VideoCapture(i, cv2.CAP_MSMF)
                    if not cap.isOpened():
                        # Try default backend
                        cap = cv2.VideoCapture(i)
                else:
                    cap = cv2.VideoCapture(i)
                
                if cap is not None and cap.isOpened():
                    ret, _ = cap.read()
                    if ret:
                        available.append(f"Camera {i}")
                    cap.release()
            except Exception as e:
                print(f"Camera {i} test failed: {e}")
                continue
        if not available:
            available = ["Camera 0"]
        return available

    def change_camera_index(self, idx):
        self.selected_camera_index = idx
        self.stop_camera()

    def start_camera(self):
        # Try different backends on Windows for better compatibility
        if platform.system() == 'Windows':
            # Try DirectShow first
            self.camera = cv2.VideoCapture(self.selected_camera_index, cv2.CAP_DSHOW)
            if not self.camera.isOpened():
                # Try MSMF backend
                self.camera = cv2.VideoCapture(self.selected_camera_index, cv2.CAP_MSMF)
            if not self.camera.isOpened():
                # Try default backend
                self.camera = cv2.VideoCapture(self.selected_camera_index)
        else:
            self.camera = cv2.VideoCapture(self.selected_camera_index)
            
        if self.camera.isOpened():
            print(f"✅ Camera {self.selected_camera_index} opened successfully")
            self.timer.start(30)
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            
            # Add success animation
            self.animate_button_success(self.start_btn)
        else:
            print(f"❌ Failed to open camera {self.selected_camera_index}")
            QMessageBox.warning(self, "Camera Error", "Could not open camera. Please check if camera is connected and not in use by another application.")

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

class MetadataReviewDialog(QDialog):
    """Dialog for reviewing and editing metadata before database save"""
    def __init__(self, metadata, parent=None):
        super().__init__(parent)
        self.metadata = metadata.copy()
        self.setup_ui()
        self.populate_fields()
        
    def setup_ui(self):
        self.setWindowTitle("📚 Review & Edit Metadata")
        self.setGeometry(200, 100, 800, 600)
        self.setStyleSheet("""
            QDialog {
                background-color: #181c20;
                color: #fff;
            }
            QLabel {
                color: #fff;
                font-weight: 600;
            }
            QLineEdit, QTextEdit, QSpinBox, QDateEdit {
                background-color: #23272e;
                border: 2px solid #333;
                border-radius: 6px;
                padding: 8px;
                color: #fff;
                font-size: 12px;
            }
            QLineEdit:focus, QTextEdit:focus, QSpinBox:focus, QDateEdit:focus {
                border-color: #1976d2;
            }
            QPushButton {
                background-color: #1976d2;
                color: #fff;
                border: none;
                border-radius: 6px;
                padding: 10px 20px;
                font-weight: 600;
                min-width: 100px;
            }
            QPushButton:hover {
                background-color: #1565c0;
            }
            QPushButton:pressed {
                background-color: #0d47a1;
            }
            QGroupBox {
                font-weight: bold;
                font-size: 13px;
                color: #fff;
                border: 2px solid #333;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
                background-color: #23272e;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 8px 0 8px;
            }
        """)
        
        layout = QVBoxLayout(self)
        
        # Header
        header_label = QLabel("📚 Review & Edit Book Metadata")
        header_label.setFont(QFont("Segoe UI", 16, QFont.Weight.Bold))
        header_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header_label.setStyleSheet("color: #fff; margin-bottom: 10px;")
        layout.addWidget(header_label)
        
        # Instructions
        instructions = QLabel("💡 Review the extracted metadata below. You can edit any field or add missing information before saving to the database.")
        instructions.setWordWrap(True)
        instructions.setStyleSheet("color: #b3c6e0; font-size: 12px; margin-bottom: 15px; padding: 10px; background-color: #23272e; border-radius: 6px;")
        layout.addWidget(instructions)
        
        # Create scrollable form
        scroll_area = QScrollArea()
        scroll_widget = QWidget()
        form_layout = QFormLayout(scroll_widget)
        form_layout.setSpacing(15)
        
        # Basic Information Group
        basic_group = QGroupBox("📖 Basic Information")
        basic_layout = QFormLayout(basic_group)
        
        self.title_edit = QLineEdit()
        self.title_edit.setPlaceholderText("Enter book title...")
        basic_layout.addRow("📖 Title:", self.title_edit)
        
        self.authors_edit = QLineEdit()
        self.authors_edit.setPlaceholderText("Enter authors (separate with commas)...")
        basic_layout.addRow("✍️ Authors:", self.authors_edit)
        
        self.publisher_edit = QLineEdit()
        self.publisher_edit.setPlaceholderText("Enter publisher...")
        basic_layout.addRow("🏢 Publisher:", self.publisher_edit)
        
        self.year_spinbox = QSpinBox()
        self.year_spinbox.setRange(0, 2100)  # Allow 0 for empty/null
        self.year_spinbox.setValue(0)  # Start with 0 (empty)
        self.year_spinbox.setSpecialValueText("")  # Show empty text when value is 0
        basic_layout.addRow("📅 Publication Year:", self.year_spinbox)
        
        self.edition_edit = QLineEdit()
        self.edition_edit.setPlaceholderText("Enter edition (e.g., 1st, 2nd, etc.)...")
        basic_layout.addRow("📚 Edition:", self.edition_edit)
        
        form_layout.addRow(basic_group)
        
        # ISBN Information Group
        isbn_group = QGroupBox("🔢 ISBN Information")
        isbn_layout = QFormLayout(isbn_group)
        
        self.isbn10_edit = QLineEdit()
        self.isbn10_edit.setPlaceholderText("Enter ISBN-10...")
        isbn_layout.addRow("🔢 ISBN-10:", self.isbn10_edit)
        
        self.isbn13_edit = QLineEdit()
        self.isbn13_edit.setPlaceholderText("Enter ISBN-13...")
        isbn_layout.addRow("🔢 ISBN-13:", self.isbn13_edit)
        
        form_layout.addRow(isbn_group)
        
        # Additional Information Group
        additional_group = QGroupBox("📋 Additional Information")
        additional_layout = QFormLayout(additional_group)
        
        self.series_edit = QLineEdit()
        self.series_edit.setPlaceholderText("Enter series name...")
        additional_layout.addRow("🔗 Series:", self.series_edit)
        
        self.genre_edit = QLineEdit()
        self.genre_edit.setPlaceholderText("Enter genre...")
        additional_layout.addRow("🏷️ Genre:", self.genre_edit)
        
        self.language_edit = QLineEdit()
        self.language_edit.setPlaceholderText("Enter language...")
        additional_layout.addRow("🌐 Language:", self.language_edit)
        
        self.lccn_edit = QLineEdit()
        self.lccn_edit.setPlaceholderText("Enter LCCN...")
        additional_layout.addRow("📋 LCCN:", self.lccn_edit)
        
        self.oclc_edit = QLineEdit()
        self.oclc_edit.setPlaceholderText("Enter OCLC number...")
        additional_layout.addRow("🔢 OCLC:", self.oclc_edit)
        
        self.additional_text_edit = QTextEdit()
        self.additional_text_edit.setMaximumHeight(80)
        self.additional_text_edit.setPlaceholderText("Enter any additional notes or information...")
        additional_layout.addRow("📝 Additional Notes:", self.additional_text_edit)
        
        form_layout.addRow(additional_group)
        
        # Quality Check Group
        quality_group = QGroupBox("✅ Quality Check")
        quality_layout = QFormLayout(quality_group)
        
        self.confidence_label = QLabel("Extraction Confidence")
        self.confidence_label.setStyleSheet("color: #4caf50; font-weight: 600;")
        quality_layout.addRow("🎯 Confidence:", self.confidence_label)
        
        self.word_count_label = QLabel("0 words extracted")
        self.word_count_label.setStyleSheet("color: #b3c6e0;")
        quality_layout.addRow("📊 Words Extracted:", self.word_count_label)
        
        form_layout.addRow(quality_group)
        
        scroll_area.setWidget(scroll_widget)
        scroll_area.setWidgetResizable(True)
        layout.addWidget(scroll_area)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        self.save_btn = ModernButton("💾 Save to Database", "#28a745", "#218838")
        self.save_btn.clicked.connect(self.accept)
        
        self.cancel_btn = ModernButton("❌ Cancel", "#dc3545", "#c82333")
        self.cancel_btn.clicked.connect(self.reject)
        
        self.preview_btn = ModernButton("👁️ Preview", "#17a2b8", "#138496")
        self.preview_btn.clicked.connect(self.show_preview)
        
        button_layout.addWidget(self.preview_btn)
        button_layout.addStretch()
        button_layout.addWidget(self.cancel_btn)
        button_layout.addWidget(self.save_btn)
        
        layout.addLayout(button_layout)
    
    def populate_fields(self):
        """Populate form fields with extracted metadata"""
        if self.metadata.get('title'):
            self.title_edit.setText(self.metadata['title'])
        
        if self.metadata.get('authors'):
            if isinstance(self.metadata['authors'], list):
                self.authors_edit.setText(', '.join(self.metadata['authors']))
            else:
                self.authors_edit.setText(str(self.metadata['authors']))
        
        if self.metadata.get('publisher'):
            self.publisher_edit.setText(self.metadata['publisher'])
        
        if self.metadata.get('year'):
            try:
                year = int(self.metadata['year'])
                if year > 0:
                    self.year_spinbox.setValue(year)
                else:
                    self.year_spinbox.setValue(0)  # Set to 0 (empty) for invalid years
            except (ValueError, TypeError):
                self.year_spinbox.setValue(0)  # Set to 0 (empty) for invalid years
        
        if self.metadata.get('edition'):
            self.edition_edit.setText(self.metadata['edition'])
        
        if self.metadata.get('isbn10'):
            self.isbn10_edit.setText(self.metadata['isbn10'])
        
        if self.metadata.get('isbn13'):
            self.isbn13_edit.setText(self.metadata['isbn13'])
        
        if self.metadata.get('series'):
            self.series_edit.setText(self.metadata['series'])
        
        if self.metadata.get('genre'):
            # Convert list to comma-separated string if it's a list
            genre_value = self.metadata['genre']
            if isinstance(genre_value, list):
                genre_value = ', '.join(genre_value)
            self.genre_edit.setText(genre_value)
        
        if self.metadata.get('language'):
            self.language_edit.setText(self.metadata['language'])
        
        if self.metadata.get('lccn'):
            self.lccn_edit.setText(self.metadata['lccn'])
        
        if self.metadata.get('oclc_no'):
            self.oclc_edit.setText(self.metadata['oclc_no'])
        
        if self.metadata.get('additional_text'):
            self.additional_text_edit.setText(self.metadata['additional_text'])
        
        # Set confidence and word count
        confidence = self.metadata.get('confidence', 0.0)
        if confidence > 0.8:
            confidence_color = "#4caf50"
            confidence_text = "High"
        elif confidence > 0.6:
            confidence_color = "#ff9800"
            confidence_text = "Medium"
        else:
            confidence_color = "#f44336"
            confidence_text = "Low"
        
        self.confidence_label.setText(f"{confidence_text} ({confidence:.1%})")
        self.confidence_label.setStyleSheet(f"color: {confidence_color}; font-weight: 600;")
        
        word_count = self.metadata.get('word_count', 0)
        self.word_count_label.setText(f"{word_count} words extracted")
    
    def get_edited_metadata(self):
        """Get the edited metadata from form fields"""
        edited_metadata = self.metadata.copy()
        
        edited_metadata['title'] = self.title_edit.text().strip()
        edited_metadata['authors'] = [author.strip() for author in self.authors_edit.text().split(',') if author.strip()]
        edited_metadata['publisher'] = self.publisher_edit.text().strip()
        year_value = self.year_spinbox.value()
        if year_value > 0:
            edited_metadata['year'] = str(year_value)
        else:
            edited_metadata['year'] = None  # Set to None for empty/null
        edited_metadata['edition'] = self.edition_edit.text().strip()
        edited_metadata['isbn10'] = self.isbn10_edit.text().strip()
        edited_metadata['isbn13'] = self.isbn13_edit.text().strip()
        edited_metadata['series'] = self.series_edit.text().strip()
        edited_metadata['genre'] = self.genre_edit.text().strip()
        edited_metadata['language'] = self.language_edit.text().strip()
        edited_metadata['lccn'] = self.lccn_edit.text().strip()
        edited_metadata['oclc_no'] = self.oclc_edit.text().strip()
        edited_metadata['additional_text'] = self.additional_text_edit.toPlainText().strip()
        
        # Remove empty fields
        edited_metadata = {k: v for k, v in edited_metadata.items() if v}
        
        return edited_metadata
    
    def show_preview(self):
        """Show a preview of the metadata as it will appear in the database"""
        edited_metadata = self.get_edited_metadata()
        
        preview_text = "📚 Metadata Preview\n" + "="*50 + "\n\n"
        
        if edited_metadata.get('title'):
            preview_text += f"📖 Title: {edited_metadata['title']}\n\n"
        if edited_metadata.get('authors'):
            preview_text += f"✍️ Authors: {', '.join(edited_metadata['authors'])}\n\n"
        if edited_metadata.get('publisher'):
            preview_text += f"🏢 Publisher: {edited_metadata['publisher']}\n\n"
        if edited_metadata.get('year'):
            preview_text += f"📅 Year: {edited_metadata['year']}\n\n"
        if edited_metadata.get('edition'):
            preview_text += f"📚 Edition: {edited_metadata['edition']}\n\n"
        if edited_metadata.get('isbn10'):
            preview_text += f"🔢 ISBN-10: {edited_metadata['isbn10']}\n\n"
        if edited_metadata.get('isbn13'):
            preview_text += f"🔢 ISBN-13: {edited_metadata['isbn13']}\n\n"
        if edited_metadata.get('series'):
            preview_text += f"🔗 Series: {edited_metadata['series']}\n\n"
        if edited_metadata.get('genre'):
            preview_text += f"🏷️ Genre: {edited_metadata['genre']}\n\n"
        if edited_metadata.get('language'):
            preview_text += f"🌐 Language: {edited_metadata['language']}\n\n"
        if edited_metadata.get('lccn'):
            preview_text += f"📋 LCCN: {edited_metadata['lccn']}\n\n"
        if edited_metadata.get('oclc_no'):
            preview_text += f"🔢 OCLC: {edited_metadata['oclc_no']}\n\n"
        if edited_metadata.get('additional_text'):
            preview_text += f"📝 Additional Notes: {edited_metadata['additional_text']}\n\n"
        
        QMessageBox.information(self, "Metadata Preview", preview_text)

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
        
        # Add helpful tip label
        self.capture_tip_label = QLabel("💡 Tip: Capture front cover, back cover, and any pages with text for best results!")
        self.capture_tip_label.setFont(QFont("Segoe UI", 9))
        self.capture_tip_label.setStyleSheet("color: #4caf50; font-style: italic; margin-bottom: 5px;")
        self.capture_tip_label.setWordWrap(True)
        status_layout.addWidget(self.capture_tip_label)
        
        self.captured_list_label = QLabel("")
        self.captured_list_label.setFont(QFont("Segoe UI", 10))
        self.captured_list_label.setStyleSheet("color: #b3c6e0;")
        status_layout.addWidget(self.captured_list_label)
        left_panel.addWidget(status_group)
        # Single capture button
        capture_group = QGroupBox("📸 Capture Controls")
        capture_layout = QHBoxLayout(capture_group)
        self.capture_image_btn = ModernButton("📸 Capture Image", "#28a745", "#218838")
        self.capture_image_btn.clicked.connect(self.capture_image)
        self.capture_image_btn.setToolTip("Click to capture the current camera frame. You can capture multiple images!")
        capture_layout.addWidget(self.capture_image_btn)
        # Process button
        self.process_btn = ModernButton("⚡ Process Images", "#388e3c", "#2e7031")
        self.process_btn.setEnabled(False)
        self.process_btn.clicked.connect(self.start_processing)
        capture_layout.addWidget(self.process_btn)
        
        # Review Metadata button
        self.review_btn = ModernButton("📝 Review Metadata", "#17a2b8", "#138496")
        self.review_btn.setEnabled(False)
        self.review_btn.clicked.connect(self.review_metadata)
        capture_layout.addWidget(self.review_btn)
        
        # Next Book/Reset button
        self.reset_btn = ModernButton("🔄 Next Book / Reset", "#6f42c1", "#5a32a3")
        self.reset_btn.setEnabled(False)
        self.reset_btn.clicked.connect(self.reset_for_next_book)
        self.reset_btn.setToolTip("Clear current data and prepare for the next book")
        capture_layout.addWidget(self.reset_btn)
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
        
        # Create tab widget for metadata display
        metadata_tabs = QTabWidget()
        metadata_tabs.setStyleSheet("""
            QTabWidget::pane {
                border: 2px solid #333;
                border-radius: 8px;
                background-color: #23272e;
            }
            QTabBar::tab {
                background-color: #181c20;
                color: #fff;
                padding: 8px 16px;
                margin-right: 2px;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
            }
            QTabBar::tab:selected {
                background-color: #1976d2;
            }
            QTabBar::tab:hover {
                background-color: #1565c0;
            }
        """)
        
        # Gemini Metadata Tab
        gemini_tab = QWidget()
        gemini_layout = QVBoxLayout(gemini_tab)
        self.gemini_results_text = QTextEdit()
        self.gemini_results_text.setPlaceholderText("🤖 Gemini Vision metadata will appear here...\n\n💡 This shows the initial extraction from the book images")
        self.gemini_results_text.setMinimumHeight(300)
        self.gemini_results_text.setReadOnly(True)
        gemini_layout.addWidget(self.gemini_results_text)
        metadata_tabs.addTab(gemini_tab, "🤖 Gemini Vision")
        
        # Final Metadata Tab
        final_tab = QWidget()
        final_layout = QVBoxLayout(final_tab)
        self.final_results_text = QTextEdit()
        self.final_results_text.setPlaceholderText("📚 Final unified metadata will appear here...\n\n💡 This shows the best possible metadata from all sources")
        self.final_results_text.setMinimumHeight(300)
        self.final_results_text.setReadOnly(True)
        final_layout.addWidget(self.final_results_text)
        metadata_tabs.addTab(final_tab, "📚 Final Metadata")
        
        right_panel.addWidget(metadata_tabs)
        
        # Database Status Group
        db_group = QGroupBox("🗄️ Database Status")
        db_layout = QVBoxLayout(db_group)
        self.db_status_text = QLabel("No data available")
        self.db_status_text.setStyleSheet("""
            QLabel {
                padding: 10px;
                background-color: #181c20;
                border-radius: 8px;
                color: #b3c6e0;
                font-size: 12px;
            }
        """)
        self.db_status_text.setWordWrap(True)
        db_layout.addWidget(self.db_status_text)
        right_panel.addWidget(db_group)
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
            
            # Show success notification
            self.show_capture_notification(f"✅ Image {len(self.captured_images)} captured successfully!")
            
            # Prompt for more images if less than 2 captured
            if len(self.captured_images) < 2:
                QMessageBox.information(self, "Capture More Images", 
                    f"Great! You've captured {len(self.captured_images)} image(s).\n\n"
                    "💡 Tip: Capture multiple images for better results:\n"
                    "• Front cover\n"
                    "• Back cover\n"
                    "• Any additional pages with text\n\n"
                    "You can capture as many images as you want!")
            elif len(self.captured_images) == 2:
                QMessageBox.information(self, "Good Progress!", 
                    "Excellent! You've captured 2 images.\n\n"
                    "You can continue capturing more images if needed, "
                    "or click 'Process Images' when you're ready.")
            else:
                QMessageBox.information(self, "Multiple Images Captured", 
                    f"Perfect! You've captured {len(self.captured_images)} images.\n\n"
                    "You can continue capturing more or click 'Process Images' when ready.")
    
    def show_capture_notification(self, message):
        """Show a green notification popup"""
        notification = QMessageBox(self)
        notification.setIcon(QMessageBox.Icon.Information)
        notification.setText(message)
        notification.setWindowTitle("Image Captured")
        notification.setStyleSheet("""
            QMessageBox {
                background-color: #23272e;
                color: #fff;
            }
            QMessageBox QLabel {
                color: #fff;
                font-size: 14px;
                font-weight: 600;
            }
            QPushButton {
                background-color: #28a745;
                color: #fff;
                border: none;
                border-radius: 6px;
                padding: 8px 16px;
                font-weight: 600;
                min-width: 80px;
            }
            QPushButton:hover {
                background-color: #218838;
            }
        """)
        notification.exec()

    def update_capture_status(self):
        count = len(self.captured_images)
        self.captured_count_label.setText(f"Captured Images: {count}")
        
        # Update tip based on number of images
        if count == 0:
            self.capture_tip_label.setText("💡 Tip: Capture front cover, back cover, and any pages with text for best results!")
            self.capture_tip_label.setStyleSheet("color: #4caf50; font-style: italic; margin-bottom: 5px;")
        elif count == 1:
            self.capture_tip_label.setText("✅ Good start! Now capture the back cover for ISBN information.")
            self.capture_tip_label.setStyleSheet("color: #ff9800; font-style: italic; margin-bottom: 5px;")
        elif count == 2:
            self.capture_tip_label.setText("🎉 Excellent! You have front and back covers. You can capture more or process now!")
            self.capture_tip_label.setStyleSheet("color: #28a745; font-style: italic; margin-bottom: 5px;")
        else:
            self.capture_tip_label.setText(f"🌟 Perfect! You have {count} images. Ready to process or capture more!")
            self.capture_tip_label.setStyleSheet("color: #28a745; font-style: italic; margin-bottom: 5px;")
        
        if count > 0:
            image_types = []
            if count >= 1:
                image_types.append("Front cover")
            if count >= 2:
                image_types.append("Back cover")
            if count > 2:
                image_types.append(f"{count-2} additional image(s)")
            
            self.captured_list_label.setText(" | ".join(image_types))
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
        self.review_btn.setEnabled(False)
        self.reset_btn.setEnabled(False)
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

    def on_processing_complete(self, gemini_metadata, unified_metadata):
        self.progress_bar.setVisible(False)
        self.capture_image_btn.setEnabled(True)
        self.process_btn.setEnabled(True)
        self.review_btn.setEnabled(True)
        self.reset_btn.setEnabled(True)
        self.progress_label.setText("Processing complete ✓")
        self.progress_label.setStyleSheet("color: #28a745; font-size: 12px; font-weight: 600; margin-top: 5px;")
        
        # Store the metadata for later review
        self.current_unified_metadata = unified_metadata
        
        # Display Gemini metadata
        if gemini_metadata:
            gemini_text = "🤖 Gemini Vision Metadata\n" + "="*50 + "\n\n"
            if gemini_metadata.get('title'):
                gemini_text += f"📖 Title: {gemini_metadata['title']}\n\n"
            if gemini_metadata.get('authors'):
                gemini_text += f"✍️ Authors: {', '.join(gemini_metadata['authors'])}\n\n"
            if gemini_metadata.get('isbn'):
                gemini_text += f"🔢 ISBN: {gemini_metadata['isbn']}\n\n"
            if gemini_metadata.get('isbn10'):
                gemini_text += f"🔢 ISBN-10: {gemini_metadata['isbn10']}\n\n"
            if gemini_metadata.get('isbn13'):
                gemini_text += f"🔢 ISBN-13: {gemini_metadata['isbn13']}\n\n"
            if gemini_metadata.get('publisher'):
                gemini_text += f"🏢 Publisher: {gemini_metadata['publisher']}\n\n"
            if gemini_metadata.get('year'):
                gemini_text += f"📅 Year: {gemini_metadata['year']}\n\n"
            if gemini_metadata.get('edition'):
                gemini_text += f"📚 Edition: {gemini_metadata['edition']}\n\n"
            if gemini_metadata.get('series'):
                gemini_text += f"🔗 Series: {gemini_metadata['series']}\n\n"
            if gemini_metadata.get('genre'):
                gemini_text += f"🏷️ Genre: {gemini_metadata['genre']}\n\n"
            if gemini_metadata.get('language'):
                gemini_text += f"🌐 Language: {gemini_metadata['language']}\n\n"
            if gemini_metadata.get('additional_text'):
                gemini_text += f"📝 Additional Text: {gemini_metadata['additional_text']}\n\n"
            self.gemini_results_text.setText(gemini_text)
        else:
            self.gemini_results_text.setText("❌ No Gemini metadata could be extracted.")
        
        # Display final unified metadata
        if unified_metadata:
            final_text = "📚 Final Unified Metadata\n" + "="*50 + "\n\n"
            if unified_metadata.get('title'):
                final_text += f"📖 Title: {unified_metadata['title']}\n\n"
            if unified_metadata.get('authors'):
                final_text += f"✍️ Authors: {', '.join(unified_metadata['authors'])}\n\n"
            if unified_metadata.get('isbn'):
                final_text += f"🔢 ISBN: {unified_metadata['isbn']}\n\n"
            if unified_metadata.get('isbn10'):
                final_text += f"🔢 ISBN-10: {unified_metadata['isbn10']}\n\n"
            if unified_metadata.get('isbn13'):
                final_text += f"🔢 ISBN-13: {unified_metadata['isbn13']}\n\n"
            if unified_metadata.get('publisher'):
                final_text += f"🏢 Publisher: {unified_metadata['publisher']}\n\n"
            if unified_metadata.get('published_date'):
                final_text += f"📅 Published Date: {unified_metadata['published_date']}\n\n"
            if unified_metadata.get('edition'):
                final_text += f"📚 Edition: {unified_metadata['edition']}\n\n"
            if unified_metadata.get('series'):
                final_text += f"🔗 Series: {unified_metadata['series']}\n\n"
            if unified_metadata.get('genre'):
                final_text += f"🏷️ Genre: {unified_metadata['genre']}\n\n"
            if unified_metadata.get('language'):
                final_text += f"🌐 Language: {unified_metadata['language']}\n\n"
            if unified_metadata.get('lccn'):
                final_text += f"📋 LCCN: {unified_metadata['lccn']}\n\n"
            if unified_metadata.get('oclc_no'):
                final_text += f"🔢 OCLC: {unified_metadata['oclc_no']}\n\n"
            if unified_metadata.get('additional_text'):
                final_text += f"📝 Additional Text: {unified_metadata['additional_text']}\n\n"
            self.final_results_text.setText(final_text)
            
            # Show metadata review dialog before database operations
            review_dialog = MetadataReviewDialog(unified_metadata, self)
            if review_dialog.exec() == QDialog.DialogCode.Accepted:
                # User clicked "Save to Database"
                edited_metadata = review_dialog.get_edited_metadata()
                
                # Update the stored metadata with edited version
                self.current_unified_metadata = edited_metadata
                
                # Update the final results text with edited metadata
                self.update_final_metadata_display(edited_metadata)
                
                # Database operations with edited metadata
                try:
                    existing = search_book(
                        isbn=edited_metadata.get('isbn') or edited_metadata.get('isbn13') or edited_metadata.get('isbn10'), 
                        title=edited_metadata.get('title'), 
                        authors=edited_metadata.get('authors')
                    )
                    
                    if existing:
                        db_status = "✅ Book already exists in database"
                        QMessageBox.information(self, "Book Exists", "✅ This book already exists in the database.")
                    else:
                        insert_success = insert_book(edited_metadata)
                        if insert_success:
                            db_status = "📚 Book metadata saved to database"
                            QMessageBox.information(self, "Book Added", "📚 Book metadata saved to the cloud database.")
                        else:
                            db_status = "⚠️ Database not available - book not saved"
                            QMessageBox.warning(self, "Database Error", "⚠️ Database not available - book metadata not saved.")
                    
                    self.db_status_text.setText(db_status)
                except Exception as e:
                    db_status = f"❌ Database error: {str(e)}"
                    self.db_status_text.setText(db_status)
                    QMessageBox.warning(self, "Database Error", f"Database operation failed: {str(e)}")
            else:
                # User clicked "Cancel"
                db_status = "❌ Metadata review cancelled - book not saved"
                self.db_status_text.setText(db_status)
                QMessageBox.information(self, "Review Cancelled", "📝 Metadata review was cancelled. Book was not saved to database.")
        else:
            self.final_results_text.setText("❌ No unified metadata could be generated.")
            self.db_status_text.setText("❌ No data to save to database")

    def on_processing_error(self, error_message):
        self.progress_bar.setVisible(False)
        self.capture_image_btn.setEnabled(True)
        self.process_btn.setEnabled(True)
        self.review_btn.setEnabled(False)
        self.reset_btn.setEnabled(True)
        self.progress_label.setText("Processing failed ✗")
        self.progress_label.setStyleSheet("color: #dc3545; font-size: 12px; font-weight: 600; margin-top: 5px;")
        QMessageBox.critical(self, "Processing Error", f"An error occurred during processing:\n{error_message}")

    def review_metadata(self):
        """Open metadata review dialog for manual editing"""
        if not hasattr(self, 'current_unified_metadata') or not self.current_unified_metadata:
            QMessageBox.warning(self, "No Metadata", "No metadata available for review. Please process images first.")
            return
        
        review_dialog = MetadataReviewDialog(self.current_unified_metadata, self)
        if review_dialog.exec() == QDialog.DialogCode.Accepted:
            # User clicked "Save to Database"
            edited_metadata = review_dialog.get_edited_metadata()
            
            # Update the stored metadata
            self.current_unified_metadata = edited_metadata
            
            # Update the final results text with edited metadata
            self.update_final_metadata_display(edited_metadata)
            
            # Database operations with edited metadata
            try:
                existing = search_book(
                    isbn=edited_metadata.get('isbn') or edited_metadata.get('isbn13') or edited_metadata.get('isbn10'), 
                    title=edited_metadata.get('title'), 
                    authors=edited_metadata.get('authors')
                )
                
                if existing:
                    db_status = "✅ Book already exists in database"
                    QMessageBox.information(self, "Book Exists", "✅ This book already exists in the database.")
                else:
                    insert_success = insert_book(edited_metadata)
                    if insert_success:
                        db_status = "📚 Book metadata saved to database"
                        QMessageBox.information(self, "Book Added", "📚 Book metadata saved to the cloud database.")
                    else:
                        db_status = "⚠️ Database not available - book not saved"
                        QMessageBox.warning(self, "Database Error", "⚠️ Database not available - book metadata not saved.")
                
                self.db_status_text.setText(db_status)
            except Exception as e:
                db_status = f"❌ Database error: {str(e)}"
                self.db_status_text.setText(db_status)
                QMessageBox.warning(self, "Database Error", f"Database operation failed: {str(e)}")
        else:
            # User clicked "Cancel"
            QMessageBox.information(self, "Review Cancelled", "📝 Metadata review was cancelled. No changes were saved.")

    def update_final_metadata_display(self, metadata):
        """Update the final metadata display with the given metadata"""
        if metadata:
            final_text = "📚 Final Unified Metadata\n" + "="*50 + "\n\n"
            if metadata.get('title'):
                final_text += f"📖 Title: {metadata['title']}\n\n"
            if metadata.get('authors'):
                final_text += f"✍️ Authors: {', '.join(metadata['authors'])}\n\n"
            if metadata.get('isbn'):
                final_text += f"🔢 ISBN: {metadata['isbn']}\n\n"
            if metadata.get('isbn10'):
                final_text += f"🔢 ISBN-10: {metadata['isbn10']}\n\n"
            if metadata.get('isbn13'):
                final_text += f"🔢 ISBN-13: {metadata['isbn13']}\n\n"
            if metadata.get('publisher'):
                final_text += f"🏢 Publisher: {metadata['publisher']}\n\n"
            if metadata.get('published_date'):
                final_text += f"📅 Published Date: {metadata['published_date']}\n\n"
            if metadata.get('year'):
                final_text += f"📅 Year: {metadata['year']}\n\n"
            if metadata.get('edition'):
                final_text += f"📚 Edition: {metadata['edition']}\n\n"
            if metadata.get('series'):
                final_text += f"🔗 Series: {metadata['series']}\n\n"
            if metadata.get('genre'):
                final_text += f"🏷️ Genre: {metadata['genre']}\n\n"
            if metadata.get('language'):
                final_text += f"🌐 Language: {metadata['language']}\n\n"
            if metadata.get('lccn'):
                final_text += f"📋 LCCN: {metadata['lccn']}\n\n"
            if metadata.get('oclc_no'):
                final_text += f"🔢 OCLC: {metadata['oclc_no']}\n\n"
            if metadata.get('additional_text'):
                final_text += f"📝 Additional Text: {metadata['additional_text']}\n\n"
            self.final_results_text.setText(final_text)
        else:
            self.final_results_text.setText("❌ No unified metadata could be generated.")

    def reset_for_next_book(self):
        """Reset the application state for processing the next book"""
        # Clear captured images
        self.captured_images.clear()
        
        # Reset UI state
        self.capture_image_btn.setEnabled(True)
        self.process_btn.setEnabled(False)
        self.review_btn.setEnabled(False)
        self.reset_btn.setEnabled(False)
        
        # Clear progress indicators
        self.progress_bar.setVisible(False)
        self.progress_label.setText("Ready to process")
        self.progress_label.setStyleSheet("color: #b3c6e0; font-size: 12px; margin-top: 2px;")
        
        # Clear metadata results
        self.gemini_results_text.clear()
        self.gemini_results_text.setPlaceholderText("🤖 Gemini Vision metadata will appear here...\n\n💡 This shows the initial extraction from the book images")
        
        self.final_results_text.clear()
        self.final_results_text.setPlaceholderText("📚 Final unified metadata will appear here...\n\n💡 This shows the best possible metadata from all sources")
        
        # Clear database status
        self.db_status_text.setText("No data available")
        
        # Update capture status
        self.update_capture_status()
        
        # Clear stored metadata
        if hasattr(self, 'current_unified_metadata'):
            delattr(self, 'current_unified_metadata')
        
        # Show confirmation message
        QMessageBox.information(self, "Reset Complete", 
            "🔄 Application reset successfully!\n\n"
            "✅ Ready to process the next book.\n"
            "📸 You can now capture new images.")

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

# Enhanced GeminiProcessingThread with external API integration
from PyQt6.QtCore import QThread, pyqtSignal
class GeminiProcessingThread(QThread):
    processing_complete = pyqtSignal(dict, dict)  # (gemini_metadata, unified_metadata)
    processing_error = pyqtSignal(str)
    progress_update = pyqtSignal(int)
    
    def __init__(self, image_list):
        super().__init__()
        self.image_list = image_list
    
    def extract_all_isbns(self, metadata):
        """Extract all ISBNs from metadata dictionary"""
        isbns = []
        if metadata.get('isbn'):
            isbns.append(metadata['isbn'])
        if metadata.get('isbn10'):
            isbns.append(metadata['isbn10'])
        if metadata.get('isbn13'):
            isbns.append(metadata['isbn13'])
        return isbns
    
    def run(self):
        try:
            # Step 1: Process images with Gemini (20%)
            self.progress_update.emit(20)
            gemini_metadata = process_book_images(self.image_list, prompt_type="comprehensive", infer_missing=True)
            
            # Step 2: Extract ISBNs (30%)
            self.progress_update.emit(30)
            isbns = self.extract_all_isbns(gemini_metadata)
            
            # Step 3: Query external APIs (60%)
            self.progress_update.emit(60)
            gb_data = {}
            ol_data = {}
            loc_data = {}
            isbnlib_data = {}
            
            if isbns:
                # Google Books
                try:
                    for isbn in isbns:
                        gb_result = search_book_by_isbn(isbn)
                        if gb_result:
                            gb_data = extract_book_metadata(gb_result)
                            break
                except Exception as e:
                    print(f"Google Books API error: {e}")
                
                # OpenLibrary
                try:
                    ol_api = OpenLibraryAPI()
                    for isbn in isbns:
                        ol_result = ol_api.search_by_isbn(isbn)
                        if ol_result:
                            ol_data = ol_result
                            break
                except Exception as e:
                    print(f"OpenLibrary API error: {e}")
                
                # LOC for LCCN
                try:
                    loc_converter = LOCConverter()
                    loc_results_raw = loc_converter.get_lccn_for_isbns(isbns)
                    lccn_value = next((lccn for lccn in loc_results_raw.values() if lccn), None)
                    loc_data = {'lccn': lccn_value} if lccn_value else {}
                except Exception as e:
                    print(f"LOC API error: {e}")
                
                # isbnlib
                try:
                    isbn_service = ISBNService()
                    for isbn in isbns:
                        isbnlib_result = isbn_service.search_by_isbn(isbn)
                        if isbnlib_result:
                            isbnlib_data = isbnlib_result
                            break
                except Exception as e:
                    print(f"isbnlib error: {e}")
            
            # Step 4: Merge metadata with LLM (90%)
            self.progress_update.emit(90)
            try:
                unified_metadata = llm_metadata_combiner(
                    gemini_metadata, gb_data, ol_data, loc_data, isbnlib_data, debug=False
                )
            except Exception as e:
                print(f"LLM combiner error: {e}")
                # Fallback to simple merge
                unified_metadata = gemini_metadata.copy()
                if gb_data:
                    unified_metadata.update(gb_data)
                if ol_data:
                    unified_metadata.update(ol_data)
                if loc_data:
                    unified_metadata.update(loc_data)
                if isbnlib_data:
                    unified_metadata.update(isbnlib_data)
            
            # Step 5: Complete (100%)
            self.progress_update.emit(100)
            self.processing_complete.emit(gemini_metadata, unified_metadata)
            
        except Exception as e:
            self.processing_error.emit(str(e))

def main():
    # Try to create table, but don't fail if database is unavailable
    db_success = create_table()
    if not db_success:
        print("⚠️ Database not available - app will run without database functionality")
    
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