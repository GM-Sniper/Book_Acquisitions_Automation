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

