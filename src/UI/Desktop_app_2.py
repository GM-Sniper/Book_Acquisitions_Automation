import sys
import os
import platform
import tempfile
from typing import Optional

# Add the project root to sys.path so 'src' can be imported
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QTextEdit, QGroupBox, QProgressBar, QMessageBox, QComboBox,
    QFrame
)
from PyQt6.QtGui import QPixmap, QImage, QFont
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QThread, QPropertyAnimation, QEasingCurve

import cv2
import numpy as np

from src.vision.preprocessing import preprocess_image
from src.vision.OCR_Processing import extract_text_with_confidence
from src.metadata.metadata_extraction import extract_metadata_with_gemini, metadata_combiner
from src.utils.isbn_detection import extract_and_validate_isbns
from src.utils.database_cloud import create_table, insert_book, search_book
from src.vision.gemini_processing import process_book_images

# Call number generator
from src.utils.call_number_generator import generate_call_number_with_gemini


# --------------------------- UI Pieces ---------------------------

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


# --------------------------- Gemini wrapper (multi-image path) ---------------------------

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
            metadata = process_book_images(self.image_list, prompt_type="comprehensive", infer_missing=True)
            self.progress_update.emit(100)
            self.processing_complete.emit(metadata)
        except Exception as e:
            self.processing_error.emit(str(e))


# --------------------------- Camera widget ---------------------------

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
        if platform.system() == 'Windows':
            self.camera = cv2.VideoCapture(self.selected_camera_index, cv2.CAP_DSHOW)
        else:
            self.camera = cv2.VideoCapture(self.selected_camera_index)

        if self.camera.isOpened():
            self.timer.start(30)
            self.start_btn.setEnabled(False)
            self.capture_btn.setEnabled(True)
            self.stop_btn.setEnabled(True)
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
                scaled_pixmap = pixmap.scaled(
                    self.camera_label.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation
                )
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
                scaled_pixmap = pixmap.scaled(
                    self.camera_label.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation
                )
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
                self.animate_button_success(self.capture_btn)
                return frame
        return None

    def animate_button_success(self, button):
        animation = QPropertyAnimation(button, b"geometry")
        animation.setDuration(150)
        original_geometry = button.geometry()
        animation.setStartValue(original_geometry)
        animation.setEndValue(original_geometry)
        animation.setEasingCurve(QEasingCurve.Type.OutBounce)
        animation.start()


# --------------------------- Main Window ---------------------------

class ModernBookAcquisitionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("📚 Book Acquisition Tool - Modern Edition")
        self.setGeometry(100, 50, 1400, 900)
        self.captured_images = []
        self.processing_thread = None
        self.setup_styles()
        self.setup_ui()

    def setup_styles(self):
        self.setStyleSheet("""
            QMainWindow { background: #181c20; }
            QGroupBox {
                font-weight: bold; font-size: 13px; color: #fff;
                border: 2px solid #333; border-radius: 10px; margin-top: 10px;
                padding-top: 10px; background-color: #23272e;
            }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 8px; }
            QTextEdit {
                border: 2px solid #333; border-radius: 8px; padding: 10px;
                background-color: #181c20; color: #fff;
                font-family: 'Segoe UI','Consolas', monospace; font-size: 12px;
            }
            QProgressBar {
                border: 2px solid #333; border-radius: 8px; text-align: center;
                font-weight: bold; background-color: #23272e; color: #fff;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #1976d2, stop:1 #1565c0);
                border-radius: 6px;
            }
            QLabel { color: #fff; }
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
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #1976d2, stop:1 #1565c0);
                border-radius: 12px; padding: 12px;
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

        # Left panel
        left_panel = QVBoxLayout()
        left_panel.setSpacing(10)

        camera_group = QGroupBox("📷 Camera Capture")
        camera_layout = QVBoxLayout(camera_group)
        self.camera_widget = ModernCameraWidget()
        camera_layout.addWidget(self.camera_widget)
        left_panel.addWidget(camera_group)

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

        capture_group = QGroupBox("📸 Capture Controls")
        capture_layout = QHBoxLayout(capture_group)
        self.capture_image_btn = ModernButton("📸 Capture Image", "#1976d2", "#1565c0")
        self.capture_image_btn.clicked.connect(self.capture_image)
        capture_layout.addWidget(self.capture_image_btn)

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

        # Right panel
        right_panel = QVBoxLayout()
        right_panel.setSpacing(10)

        results_group = QGroupBox("📖 Extracted Metadata")
        results_layout = QVBoxLayout(results_group)
        self.results_text = QTextEdit()
        self.results_text.setPlaceholderText(
            "📋 Metadata will appear here after processing...\n\n💡 Tips:\n"
            "• Ensure good lighting for better OCR results\n"
            "• Hold the book steady during capture\n"
            "• Include the entire cover in the frame"
        )
        self.results_text.setMinimumHeight(300)
        self.results_text.setReadOnly(True)
        results_layout.addWidget(self.results_text)
        right_panel.addWidget(results_group)

        confidence_group = QGroupBox("📊 Confidence Analysis")
        confidence_layout = QVBoxLayout(confidence_group)
        self.confidence_text = QLabel("No data available")
        self.confidence_text.setStyleSheet("""
            QLabel {
                padding: 10px; background-color: #181c20; border-radius: 8px;
                color: #b3c6e0; font-size: 12px;
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
        self.capture_image_btn.setEnabled(False)
        self.process_btn.setEnabled(False)
        self.progress_label.setText("Processing... Please wait")
        self.progress_label.setStyleSheet("color: #fd7e14; font-size: 12px; font-weight: 600; margin-top: 5px;")
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)

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

        if metadata:
            # Generate/attach call number immediately
            call_number = None
            try:
                call_number = generate_call_number_with_gemini(metadata)  # returns string always (fallbacks included)
            except Exception as e:
                print(f"Call number generation failed: {e}")

            if call_number:
                metadata["call_number"] = call_number

            # ---- Build results text (Call Number shown right after language/genre) ----
            lines = ["📚 Book Metadata", "="*50, ""]
            if metadata.get('title'):
                lines.append(f"📖 Title: {metadata['title']}\n")
            if metadata.get('authors'):
                lines.append(f"✍️ Authors: {', '.join(metadata['authors'])}\n")
            if metadata.get('isbn'):
                lines.append(f"🔢 ISBN: {metadata['isbn']}\n")
            if metadata.get('isbn10'):
                lines.append(f"🔢 ISBN-10: {metadata['isbn10']}\n")
            if metadata.get('isbn13'):
                lines.append(f"🔢 ISBN-13: {metadata['isbn13']}\n")
            if metadata.get('publisher'):
                lines.append(f"🏢 Publisher: {metadata['publisher']}\n")
            if metadata.get('year'):
                lines.append(f"📅 Year: {metadata['year']}\n")
            if metadata.get('edition'):
                lines.append(f"📚 Edition: {metadata['edition']}\n")
            if metadata.get('series'):
                lines.append(f"🔗 Series: {metadata['series']}\n")
            if metadata.get('genre'):
                lines.append(f"🏷️ Genre: {metadata['genre']}\n")
            if metadata.get('language'):
                lines.append(f"🌐 Language: {metadata['language']}\n")

            # 👉 Call Number line appears *after* main descriptive metadata
            if metadata.get("call_number"):
                lines.append(f"🗂️ Call Number: {metadata['call_number']}\n")

            # If we used a fallback, show a small note
            note = metadata.get("call_number_note")
            if note:
                lines.append(f"ℹ️ Call number note: {note}\n")

            if metadata.get('additional_text'):
                lines.append(f"📝 Additional Text: {metadata['additional_text']}\n")

            self.results_text.setText("\n".join(lines))

            # Confidence box
            confidence_html = f"""
            <div style="padding: 15px; background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); border-radius: 8px;">
                <h3 style="color: #2c3e50; margin: 0 0 10px 0;">Gemini Vision Analysis</h3>
                <div style="text-align: center; margin-top: 15px;">
                    <strong>Processed {len(self.captured_images)} image(s) with Gemini Vision</strong>
                </div>
            </div>
            """
            self.confidence_text.setText(confidence_html)

            # Check DB & insert (with call_number if supported)
            existing = search_book(
                isbn=metadata.get('isbn'),
                title=metadata.get('title'),
                authors=metadata.get('authors')
            )

            if existing:
                QMessageBox.information(self, "Book Exists", "✅ This book already exists in the database.")
            else:
                try:
                    insert_book(metadata)  # expects your DB layer to store "call_number" if present
                except TypeError:
                    md_copy = dict(metadata)
                    md_copy.pop("call_number", None)
                    md_copy.pop("call_number_note", None)
                    insert_book(md_copy)

                QMessageBox.information(self, "Book Added", "📚 Book metadata saved to the cloud database.")
        else:
            self.confidence_text.setText("No confidence data available")
            self.results_text.setText(
                "❌ No metadata could be extracted.\n\n💡 Suggestions:\n"
                "• Ensure good lighting\n"
                "• Hold the book steady\n"
                "• Include the entire cover\n"
                "• Try different angles"
            )

    def on_processing_error(self, error_message):
        self.progress_bar.setVisible(False)
        self.capture_image_btn.setEnabled(True)
        self.process_btn.setEnabled(True)
        self.progress_label.setText("Processing failed ✗")
        self.progress_label.setStyleSheet("color: #dc3545; font-size: 12px; font-weight: 600; margin-top: 5px;")
        QMessageBox.critical(self, "Processing Error", f"An error occurred during processing:\n{error_message}")

    def get_confidence_indicator(self, confidence):
        if confidence >= 0.8:
            return "🟢"
        elif confidence >= 0.6:
            return "🟡"
        elif confidence >= 0.4:
            return "🟠"
        else:
            return "🔴"


# --------------------------- App entry ---------------------------

def main():
    create_table()
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    app.setApplicationName("Book Acquisition Tool")
    app.setApplicationVersion("2.0")
    window = ModernBookAcquisitionApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
