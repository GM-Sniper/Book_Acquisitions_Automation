import sys
import os

# Ensure project root is in sys.path for config import
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config.config import Config
from google.cloud import vision
import io
import re
import cv2

def extract_text_from_image(image_np):
    """
    Extracts text from a preprocessed image using Google Vision API.
    Args:
        image_np (np.ndarray): Preprocessed image as a NumPy array.
    Returns:
        str: Extracted text.
    """
    # Convert NumPy array to bytes (JPEG/PNG encoding)
    _, encoded_image = cv2.imencode('.png', image_np)
    content = encoded_image.tobytes()

    client = vision.ImageAnnotatorClient()
    image = vision.Image(content=content)
    response = client.text_detection(image=image)
    texts = response.text_annotations

    if texts:
        return texts[0].description  # The first item is the full text
    return ""

def extract_text_with_confidence(image_path):
    """
    Enhanced text extraction with confidence scoring using document text detection.
    Args:
        image_path (str): Path to the image file.
    Returns:
        dict: Text and confidence information.
    """
    try:
        with open(image_path, 'rb') as image_file:
            content = image_file.read()
        
        client = vision.ImageAnnotatorClient()
        image = vision.Image(content=content)
        
        # Use document text detection (better for books)
        response = client.document_text_detection(image=image)
        
        if response.error.message:
            raise Exception(f'Vision API error: {response.error.message}')
        
        document = response.full_text_annotation
        
        if not document.text:
            return {
                "text": "",
                "confidence": 0.0,
                "word_count": 0
            }
        
        # Calculate average confidence
        total_confidence = 0
        word_count = 0
        
        for page in document.pages:
            for block in page.blocks:
                for paragraph in block.paragraphs:
                    for word in paragraph.words:
                        word_confidence = word.confidence
                        total_confidence += word_confidence
                        word_count += 1
        
        avg_confidence = total_confidence / word_count if word_count > 0 else 0
        
        return {
            "text": document.text,
            "confidence": avg_confidence,
            "word_count": word_count
        }
        
    except Exception as e:
        print(f"Error extracting text: {e}")
        return {
            "text": "",
            "confidence": 0.0,
            "error": str(e)
        }

