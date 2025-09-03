import sys
import os
import json
import platform

# Add the project root to sys.path so 'src' can be imported
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
    QLabel, QTextEdit, QGroupBox, QProgressBar, QMessageBox, QComboBox,
    QFrame, QScrollArea, QTabWidget, QDialog, QLineEdit,
    QFormLayout, QDialogButtonBox, QTextEdit, QSpinBox
)
from PyQt6.QtGui import QPixmap, QImage, QFont
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QThread, QPropertyAnimation, QEasingCurve

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import re

from src.vision.gemini_processing import process_book_images
from src.metadata.llm_metadata_combiner import llm_metadata_combiner
from src.utils.google_books import search_book_by_isbn, extract_book_metadata
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
        
        self.save_btn = ModernButton("💾 Save to Excel", "#28a745", "#218838")
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
        
        # Handle the general 'isbn' field - populate into isbn13 if it's 13 digits, otherwise isbn10
        if self.metadata.get('isbn'):
            isbn = self.metadata['isbn']
            if isbn and len(str(isbn).replace('-', '').replace(' ', '')) == 13:
                # If no isbn13 is set, use the general isbn
                if not self.metadata.get('isbn13'):
                    self.isbn13_edit.setText(isbn)
            else:
                # If no isbn10 is set, use the general isbn
                if not self.metadata.get('isbn10'):
                    self.isbn10_edit.setText(isbn)
        
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
            additional = self.metadata['additional_text']
            if isinstance(additional, dict):
                try:
                    self.additional_text_edit.setPlainText(json.dumps(additional, ensure_ascii=False, indent=2))
                except Exception:
                    self.additional_text_edit.setPlainText(str(additional))
            else:
                self.additional_text_edit.setText(str(additional))
        
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
        
        # Set the general 'isbn' field based on isbn13 or isbn10
        isbn13 = edited_metadata['isbn13']
        isbn10 = edited_metadata['isbn10']
        if isbn13 and isbn13.strip():
            edited_metadata['isbn'] = isbn13
        elif isbn10 and isbn10.strip():
            edited_metadata['isbn'] = isbn10
        else:
            edited_metadata['isbn'] = None
        
        # Remove completely empty fields (but keep fields with None values for display purposes)
        # Only remove fields that are empty strings, empty lists, or None
        cleaned_metadata = {}
        for k, v in edited_metadata.items():
            if v is not None and v != "" and v != []:
                cleaned_metadata[k] = v
            elif v is None:
                # Keep None values for fields that were explicitly cleared
                cleaned_metadata[k] = None
        
        return cleaned_metadata
    
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
        # Initialize Excel storage
        self.excel_path = self.get_default_excel_path()
        self.ensure_excel_file_exists(self.excel_path)

    # =====================
    # Excel helper methods
    # =====================
    def get_default_excel_path(self) -> str:
        base_dir = Path(project_root) / "database"
        base_dir.mkdir(parents=True, exist_ok=True)
        return str(base_dir / "AUC_records_005.xlsx")

    def ensure_excel_file_exists(self, excel_path: str):
        path_obj = Path(excel_path)
        if not path_obj.exists():
            df = pd.DataFrame(columns=["TITLE", "AUTHOR", "PUBLISHED", "D.O. Pub.", "OCLC no.", "LC no.", "ISBN", "AUC no."])
            df.to_excel(path_obj, index=False)

    def read_excel(self) -> pd.DataFrame:
        try:
            return pd.read_excel(self.excel_path, dtype=str).fillna("")
        except Exception:
            return pd.DataFrame(columns=["TITLE", "AUTHOR", "PUBLISHED", "D.O. Pub.", "OCLC no.", "LC no.", "ISBN", "AUC no."])  # fallback

    # Datavbase integration (root folder with existing records for duplicate checking)
    def get_datavbase_dir(self) -> Path:
        return Path(project_root) / "database"

    def _standardize_external_df(self, df: pd.DataFrame) -> pd.DataFrame:
        # Attempt to map varying column names to author/title/isbn
        cols_lower = {c.lower(): c for c in df.columns}
        def pick(*cands):
            for cand in cands:
                for key, orig in cols_lower.items():
                    if cand in key:
                        return orig
            return None
        author_col = pick("author", "authors", "creator", "writer", "by", "author")
        title_col = pick("title", "book title", "name", "title")
        isbn_col = pick("isbn13", "isbn-13", "isbn10", "isbn-10", "isbn", "isbn")
        published_col = pick("published", "publisher", "pub", "publishing")
        year_col = pick("year", "date", "d.o. pub.", "publication year")
        oclc_col = pick("oclc", "oclc no.", "oclc_no", "oclc number")
        lc_col = pick("lc", "lc no.", "lccn", "lc_no", "library of congress")
        out = pd.DataFrame({
            "TITLE": df.get(title_col, "").astype(str),
            "AUTHOR": df.get(author_col, "").astype(str),
            "PUBLISHED": df.get(published_col, "").astype(str),
            "D.O. Pub.": df.get(year_col, "").astype(str),
            "OCLC no.": df.get(oclc_col, "").astype(str),
            "LC no.": df.get(lc_col, "").astype(str),
            "ISBN": df.get(isbn_col, "").astype(str),
            "AUC no.": "",  # Will be auto-generated
        })
        return out.fillna("")

    def load_datavbase_records(self) -> pd.DataFrame:
        base = self.get_datavbase_dir()
        if not base.exists():
            return pd.DataFrame(columns=["TITLE", "AUTHOR", "PUBLISHED", "D.O. Pub.", "OCLC no.", "LC no.", "ISBN", "AUC no."]).fillna("")
        frames = []
        for path in base.rglob("*"):
            try:
                if path.suffix.lower() in [".xlsx", ".xlsm", ".xlsb", ".xls"]:
                    df = pd.read_excel(path, dtype=str)
                    frames.append(self._standardize_external_df(df))
                elif path.suffix.lower() in [".csv", ".tsv"]:
                    sep = "\t" if path.suffix.lower() == ".tsv" else ","
                    df = pd.read_csv(path, dtype=str, sep=sep, encoding_errors="ignore")
                    frames.append(self._standardize_external_df(df))
            except Exception:
                # Skip unreadable files
                continue
        if frames:
            return pd.concat(frames, ignore_index=True).fillna("")
        return pd.DataFrame(columns=["TITLE", "AUTHOR", "PUBLISHED", "D.O. Pub.", "OCLC no.", "LC no.", "ISBN", "AUC no."]).fillna("")

    def normalize_author(self, authors_value) -> str:
        if isinstance(authors_value, list):
            authors_text = ", ".join([str(a).strip() for a in authors_value if str(a).strip()])
        else:
            authors_text = str(authors_value or "").strip()
        return authors_text.lower()

    def normalize_title(self, title_value) -> str:
        raw = str(title_value or "").lower()
        # Remove bracketed/parenthetical content and subtitles after colon
        raw = re.sub(r"\([^)]*\)", " ", raw)
        raw = raw.split(":")[0]
        # Remove non-alphanumeric characters
        raw = re.sub(r"[^a-z0-9\s]", " ", raw)
        # Collapse whitespace
        raw = re.sub(r"\s+", " ", raw).strip()
        return raw

    def normalize_isbn(self, isbn_value) -> str:
        isbn_text = str(isbn_value or "")
        return "".join(ch for ch in isbn_text if ch.isdigit())

    def extract_year_from_text(self, value) -> str:
        text = str(value or "")
        # Match a 4-digit year between 1000 and 2099 anywhere in the string
        m = re.search(r"\b(1\d{3}|20\d{2})\b", text)
        return m.group(1) if m else ""

    def build_record_from_metadata(self, metadata: dict) -> dict:
        authors_text = ", ".join(metadata.get("authors", [])) if isinstance(metadata.get("authors"), list) else str(metadata.get("authors", "")).strip()
        isbn_value = metadata.get("isbn") or metadata.get("isbn13") or metadata.get("isbn10") or ""
        
        # Build publisher string
        publisher = metadata.get("publisher", "")
        raw_year_or_date = metadata.get("year", metadata.get("published_date", ""))
        published_str = f"{publisher}: {raw_year_or_date}" if publisher and raw_year_or_date else (publisher or raw_year_or_date or "")
        
        # Extract clean 4-digit year
        year_only = self.extract_year_from_text(raw_year_or_date)
        
        # Generate AUC number (simple increment based on existing records)
        existing_df = self.read_excel()
        next_auc_num = len(existing_df) + 1
        auc_number = f"b{next_auc_num:07d}x"
        
        return {
            "TITLE": str(metadata.get("title", "")).strip(),
            "AUTHOR": authors_text.strip(),
            "PUBLISHED": published_str,
            "D.O. Pub.": year_only,
            "OCLC no.": str(metadata.get("oclc_no", "")).strip(),
            "LC no.": str(metadata.get("lccn", "")).strip(),
            "ISBN": self.normalize_isbn(isbn_value),
            "AUC no.": auc_number,
        }

    def _tokenize(self, text: str) -> set:
        if not text:
            return set()
        text = re.sub(r"[^a-z0-9\s]", " ", text.lower())
        return set([tok for tok in text.split() if tok])

    def _jaccard(self, a: set, b: set) -> float:
        if not a or not b:
            return 0.0
        inter = len(a & b)
        union = len(a | b)
        return inter / union if union else 0.0

    def _author_tokens(self, author_text: str) -> set:
        # Normalize common separators and invert "Last, First"
        text = author_text.replace("&", ",").replace(" and ", ", ")
        parts = [p.strip() for p in text.split(",") if p.strip()]
        tokens = []
        for p in parts:
            # If looks like Lastname Firstname, keep tokens anyway
            tokens.extend(self._tokenize(p))
        return set(tokens)

    def is_duplicate_record(self, record: dict, df: pd.DataFrame) -> bool:
        if df.empty:
            return False
        # Normalize dataframe for comparison
        df_norm = df.copy()
        df_norm["author_norm"] = df_norm["AUTHOR"].astype(str).str.strip().str.lower()
        df_norm["title_norm"] = df_norm["TITLE"].astype(str).str.strip().str.lower()
        df_norm["isbn_norm"] = df_norm["ISBN"].astype(str).apply(lambda x: "".join(ch for ch in x if ch.isdigit()))

        isbn_norm = self.normalize_isbn(record.get("ISBN", ""))
        author_norm = self.normalize_author(record.get("AUTHOR", ""))
        title_norm = self.normalize_title(record.get("TITLE", ""))

        # Prefer ISBN match if available
        if isbn_norm:
            if (df_norm["isbn_norm"] == isbn_norm).any():
                return True
        # Fallback to fuzzy author+title match
        if author_norm and title_norm:
            # Precompute tokens
            rec_title_tokens = self._tokenize(title_norm)
            rec_author_tokens = self._author_tokens(author_norm)
            # Iterate rows (vectorizing true fuzzy is complex without extra deps)
            for _, row in df_norm.iterrows():
                row_title = self.normalize_title(row.get("TITLE", ""))
                row_author = str(row.get("AUTHOR", "")).lower().strip()
                title_sim = self._jaccard(rec_title_tokens, self._tokenize(row_title))
                author_sim = self._jaccard(rec_author_tokens, self._author_tokens(row_author))
                # Heuristic thresholds: tolerate messy data
                if title_sim >= 0.6 and author_sim >= 0.5:
                    return True
                # Extra rule: strong title match and any author last-name overlap
                if title_sim >= 0.75 and (rec_author_tokens & self._author_tokens(row_author)):
                    return True
        return False

    def append_record_to_excel(self, record: dict) -> bool:
        try:
            # Combine existing local Excel with datavbase records for duplicate checks
            df_local = self.read_excel()
            df_ext = self.load_datavbase_records()
            df_all = pd.concat([df_local, df_ext], ignore_index=True)
            if self.is_duplicate_record(record, df_all):
                return False  # duplicate, not appended

            new_df = pd.concat([df_local, pd.DataFrame([record])], ignore_index=True)

            # Ensure directory exists
            target_path = Path(self.excel_path)
            target_path.parent.mkdir(parents=True, exist_ok=True)

            # Enforce .xlsx extension (avoid accidental '.xls?' or other invalid suffixes)
            if target_path.suffix.lower() not in (".xlsx", ".xlsm"):
                target_path = target_path.with_suffix(".xlsx")

            try:
                new_df.to_excel(target_path, index=False)
                self.excel_path = str(target_path)
                return True
            except PermissionError:
                # If the file is open in Excel or locked, write to a fallback file
                from datetime import datetime
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                fallback = target_path.with_name(f"{target_path.stem}_{ts}{target_path.suffix}")
                try:
                    new_df.to_excel(fallback, index=False)
                    QMessageBox.information(self, "Excel Locked",
                                            f"The main file is locked. Saved to fallback: {fallback}")
                    # Keep original path for future attempts
                    return True
                except Exception as e2:
                    QMessageBox.warning(self, "Excel Error",
                                         f"Failed to write to Excel (locked?): {target_path}\n"
                                         f"Also failed fallback: {fallback}\nError: {e2}\n\n"
                                         f"Tip: Close the Excel workbook if it's open and try again.")
                    return False
        except Exception as e:
            QMessageBox.warning(self, "Excel Error", f"Failed to write to Excel: {e}")
            return False

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
        
        # Excel Status Group
        db_group = QGroupBox("📄 Excel Status")
        db_layout = QVBoxLayout(db_group)
        self.db_status_text = QLabel("No Excel activity yet")
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
            
            # Show metadata review dialog before saving to Excel
            review_dialog = MetadataReviewDialog(unified_metadata, self)
            if review_dialog.exec() == QDialog.DialogCode.Accepted:
                # User clicked "Save to Excel"
                edited_metadata = review_dialog.get_edited_metadata()
                
                # Update the stored metadata with edited version
                self.current_unified_metadata = edited_metadata
                
                # Update the final results text with edited metadata
                self.update_final_metadata_display(edited_metadata)
                
                # Save to Excel with duplicate check
                record = self.build_record_from_metadata(edited_metadata)
                df_existing = pd.concat([self.read_excel(), self.load_datavbase_records()], ignore_index=True)
                if self.is_duplicate_record(record, df_existing):
                    status_text = "✅ Duplicate detected. Not added to Excel."
                    QMessageBox.information(self, "Duplicate", "This book already exists in the Excel file (matched by ISBN or Author+Title).")
                else:
                    if self.append_record_to_excel(record):
                        status_text = "📄 Book saved to Excel."
                        QMessageBox.information(self, "Saved", "Book metadata saved to Excel.")
                    else:
                        status_text = "❌ Failed to save to Excel."
                self.db_status_text.setText(status_text)
            else:
                # User clicked "Cancel"
                status_text = "❌ Metadata review cancelled - not saved"
                self.db_status_text.setText(status_text)
                QMessageBox.information(self, "Review Cancelled", "📝 Metadata review was cancelled. Book was not saved to Excel.")
        else:
            self.final_results_text.setText("❌ No unified metadata could be generated.")
            self.db_status_text.setText("❌ No data to save to Excel")

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
            # User clicked "Save to Excel"
            edited_metadata = review_dialog.get_edited_metadata()
            
            # Update the stored metadata
            self.current_unified_metadata = edited_metadata
            
            # Update the final results text with edited metadata
            self.update_final_metadata_display(edited_metadata)
            
            # Excel operations with edited metadata
            record = self.build_record_from_metadata(edited_metadata)
            df_existing = pd.concat([self.read_excel(), self.load_datavbase_records()], ignore_index=True)
            if self.is_duplicate_record(record, df_existing):
                status_text = "✅ Duplicate detected. Not added to Excel."
                QMessageBox.information(self, "Duplicate", "This book already exists in the Excel file (matched by ISBN or Author+Title).")
            else:
                if self.append_record_to_excel(record):
                    status_text = "📄 Book saved to Excel."
                    QMessageBox.information(self, "Saved", "Book metadata saved to Excel.")
                else:
                    status_text = "❌ Failed to save to Excel."
            # Debug print of what would be saved
            print(f"DEBUG: Attempting to save record to Excel: {record}")
            self.db_status_text.setText(status_text)
        else:
            # User clicked "Cancel"
            QMessageBox.information(self, "Review Cancelled", "📝 Metadata review was cancelled. No changes were saved.")

    def update_final_metadata_display(self, metadata):
        """Update the final metadata display with the given metadata"""
        if metadata:
            final_text = "📚 Final Unified Metadata\n" + "="*50 + "\n\n"
            
            # Show all fields, including those that were cleared (None values)
            if 'title' in metadata:
                if metadata['title']:
                    final_text += f"📖 Title: {metadata['title']}\n\n"
                else:
                    final_text += f"📖 Title: [Cleared]\n\n"
            
            if 'authors' in metadata:
                if metadata['authors']:
                    final_text += f"✍️ Authors: {', '.join(metadata['authors'])}\n\n"
                else:
                    final_text += f"✍️ Authors: [Cleared]\n\n"
            
            if 'isbn' in metadata:
                if metadata['isbn']:
                    final_text += f"🔢 ISBN: {metadata['isbn']}\n\n"
                else:
                    final_text += f"🔢 ISBN: [Cleared]\n\n"
            
            if 'isbn10' in metadata:
                if metadata['isbn10']:
                    final_text += f"🔢 ISBN-10: {metadata['isbn10']}\n\n"
                else:
                    final_text += f"🔢 ISBN-10: [Cleared]\n\n"
            
            if 'isbn13' in metadata:
                if metadata['isbn13']:
                    final_text += f"🔢 ISBN-13: {metadata['isbn13']}\n\n"
                else:
                    final_text += f"🔢 ISBN-13: [Cleared]\n\n"
            
            if 'publisher' in metadata:
                if metadata['publisher']:
                    final_text += f"🏢 Publisher: {metadata['publisher']}\n\n"
                else:
                    final_text += f"🏢 Publisher: [Cleared]\n\n"
            
            if 'published_date' in metadata:
                if metadata['published_date']:
                    final_text += f"📅 Published Date: {metadata['published_date']}\n\n"
                else:
                    final_text += f"📅 Published Date: [Cleared]\n\n"
            
            if 'year' in metadata:
                if metadata['year']:
                    final_text += f"📅 Year: {metadata['year']}\n\n"
                else:
                    final_text += f"📅 Year: [Cleared]\n\n"
            
            if 'edition' in metadata:
                if metadata['edition']:
                    final_text += f"📚 Edition: {metadata['edition']}\n\n"
                else:
                    final_text += f"📚 Edition: [Cleared]\n\n"
            
            if 'series' in metadata:
                if metadata['series']:
                    final_text += f"🔗 Series: {metadata['series']}\n\n"
                else:
                    final_text += f"🔗 Series: [Cleared]\n\n"
            
            if 'genre' in metadata:
                if metadata['genre']:
                    final_text += f"🏷️ Genre: {metadata['genre']}\n\n"
                else:
                    final_text += f"🏷️ Genre: [Cleared]\n\n"
            
            if 'language' in metadata:
                if metadata['language']:
                    final_text += f"🌐 Language: {metadata['language']}\n\n"
                else:
                    final_text += f"🌐 Language: [Cleared]\n\n"
            
            if 'lccn' in metadata:
                if metadata['lccn']:
                    final_text += f"📋 LCCN: {metadata['lccn']}\n\n"
                else:
                    final_text += f"📋 LCCN: [Cleared]\n\n"
            
            if 'oclc_no' in metadata:
                if metadata['oclc_no']:
                    final_text += f"🔢 OCLC: {metadata['oclc_no']}\n\n"
                else:
                    final_text += f"🔢 OCLC: [Cleared]\n\n"
            
            if 'additional_text' in metadata:
                if metadata['additional_text']:
                    final_text += f"📝 Additional Text: {metadata['additional_text']}\n\n"
                else:
                    final_text += f"📝 Additional Text: [Cleared]\n\n"
            
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
        if not metadata:
            return []
        
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
            
            # Check if Gemini processing returned valid metadata
            if not gemini_metadata:
                gemini_metadata = {}  # Initialize empty dict if None
                print("Warning: Gemini processing returned None, using empty metadata")
            
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
            
            # Ensure we have a valid metadata dictionary
            if not unified_metadata:
                unified_metadata = {}
                print("Warning: No unified metadata generated, using empty dict")
            
            # Step 5: Complete (100%)
            self.progress_update.emit(100)
            self.processing_complete.emit(gemini_metadata, unified_metadata)
            
        except Exception as e:
            self.processing_error.emit(str(e))

def main():
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