import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Google Vision
    GOOGLE_CLOUD_PROJECT = os.getenv('GOOGLE_CLOUD_PROJECT')
    GOOGLE_CREDENTIALS_PATH = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')

    # Gemini (Generative AI)
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

    # Processing Settings
    IMAGE_MAX_SIZE = (1024, 1024)
    CONFIDENCE_THRESHOLD = 0.7

    # File Paths
    RAW_IMAGES_DIR = 'data/raw_images'
    PROCESSED_DIR = 'data/processed'
    METADATA_DIR = 'data/metadata'
