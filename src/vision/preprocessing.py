import cv2
import numpy as np


def preprocess_image(image_bytes, steps=None):
    """
    Preprocess an image for OCR using OpenCV.
    Args:
        image_bytes (bytes): The image data in bytes (from Streamlit uploader).
        steps (list, optional): List of preprocessing steps to apply. If None, applies default steps.
    Returns:
        np.ndarray: The preprocessed image ready for OCR for google vision.
    """
    # Convert bytes to numpy array
    file_bytes = np.asarray(bytearray(image_bytes), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Default preprocessing steps - OCR optimized
    if steps is None:
        steps = [to_grayscale, enhance_contrast, denoise]

    for step in steps:
        img = step(img)
    return img

# --- Preprocessing steps ---
def to_grayscale(img):
    """Convert image to grayscale."""
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def enhance_contrast_gentle(img):
    """Enhance contrast using CLAHE with gentler parameters for better OCR."""
    if len(img.shape) == 2:  # Grayscale
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8,8))  # Gentler clipLimit
        return clahe.apply(img)
    else:  # Color image
        # Convert to LAB color space
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8,8))  # Gentler clipLimit
        lab[:,:,0] = clahe.apply(lab[:,:,0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

def optional_denoise(img):
    """Light denoising that preserves text details."""
    if len(img.shape) == 2:  # Grayscale
        return cv2.fastNlMeansDenoising(img, None, 10, 7, 21)  # Gentler parameters
    else:
        return cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)  # Gentler parameters

# --- Legacy functions (kept for compatibility) ---
def enhance_contrast(img):
    """Original CLAHE enhancement (more aggressive)."""
    if len(img.shape) == 2:  # Grayscale
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        return clahe.apply(img)
    else:  # Color image
        # Convert to LAB color space
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        lab[:,:,0] = clahe.apply(lab[:,:,0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

def threshold(img):
    """Apply adaptive thresholding to binarize the image."""
    return cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    ) if len(img.shape) == 2 else img

def denoise(img):
    """Original denoising (more aggressive)."""
    if len(img.shape) == 2:
        return cv2.fastNlMeansDenoising(img, None, 21, 7, 21)
    else:
        return cv2.fastNlMeansDenoisingColored(img, None, 30, 30, 7, 21)

# We can experment with more preprocessing steps if needed (e.g., deskew, morphological ops, etc.)
# For now, we will use these default preprocessing steps.
