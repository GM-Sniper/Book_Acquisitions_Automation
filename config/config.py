import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Google Vision
    GOOGLE_CLOUD_PROJECT = os.getenv('GOOGLE_CLOUD_PROJECT')
    GOOGLE_CREDENTIALS_PATH = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
    GOOGLE_BOOKS_API_KEY = os.getenv('GOOGLE_BOOKS_API_KEY')

    # Gemini (Generative AI)
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

    # WorldCat API
    WORLDCAT_CLIENT_ID = os.getenv("WORLDCAT_CLIENT_ID")
    WORLDCAT_CLIENT_SECRET = os.getenv("WORLDCAT_CLIENT_SECRET")

   # Library of Congress APIs only
    LOC_LCCN_BASE_URL = os.getenv('LOC_LCCN_BASE_URL', 'http://lccn.loc.gov')
    LOC_SRU_BASE_URL = os.getenv('LOC_SRU_BASE_URL', 'http://lx2.loc.gov:210/lcdb')
    
    # Request settings
    REQUEST_DELAY = int(os.getenv('REQUEST_DELAY', 3))
    MAX_RETRIES = int(os.getenv('MAX_RETRIES', 3))
    TIMEOUT = int(os.getenv('TIMEOUT', 15))


    # Processing Settings
    IMAGE_MAX_SIZE = (1024, 1024)
    CONFIDENCE_THRESHOLD = 0.7

    # File Paths
    RAW_IMAGES_DIR = 'data/raw_images'
    PROCESSED_DIR = 'data/processed'
    METADATA_DIR = 'data/metadata'
