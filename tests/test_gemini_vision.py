import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.vision.gemini_processing import process_book_images
from src.vision.preprocessing import preprocess_image
from PIL import Image
import numpy as np

def test_image_processing():
    """
    Test: Give front file paths without preprocessing.
    """
    print("\n=== Front Cover Test ===")
    
    front_path = "data/raw_images/book4_front.jpg"
    
    if os.path.exists(front_path):
        try:
            #(with preprocessing)
            with open(front_path, 'rb') as f:
                front_bytes = f.read()
            front_processed = preprocess_image(front_bytes)
            metadata = process_book_images([front_processed], prompt_type="detailed")
            print("Front metadata:", metadata)

            #  Process with Gemini (no preprocessing)
            # metadata = process_book_images([front_path], prompt_type="detailed")
            # print("Front metadata:", metadata)
            
        except Exception as e:
            print(f"Error: {e}")


def test_front_back_covers():
    """
    Test: Give front and back cover file paths without preprocessing.
    """
    print("\n=== Front and Back Covers Test ===")
    
    front_path = "data/raw_images/book1_front.jpg"
    back_path = "data/raw_images/book1_back.jpg"
    
    if os.path.exists(front_path) and os.path.exists(back_path):
        try:
            #(with preprocessing)
            with open(front_path, 'rb') as f:
                front_bytes = f.read()
            with open(back_path, 'rb') as f:
                back_bytes = f.read()
            
            front_processed = preprocess_image(front_bytes)
            back_processed = preprocess_image(back_bytes)
            metadata = process_book_images([front_processed, back_processed], prompt_type="detailed")
            print("Front and back covers metadata:", metadata)

            #(without preprocessing)
            # metadata = process_book_images([front_path, back_path], prompt_type="detailed")
            # print("Front and back covers metadata:", metadata)
            
        except Exception as e:
            print(f"Error: {e}")
    else:
        print("Please place book3_front.jpg and book3_back.jpg in data/raw_images/")


if __name__ == "__main__":
    print("Gemini Vision Test")
    print("=" * 30)
    
    # Check if API key is available
    from config.config import Config
    if not Config.GEMINI_API_KEY:
        print("Warning: GEMINI_API_KEY not found in environment variables.")
        print("Please set your Gemini API key to test the functionality.")
    else:
        print("API key found. Running tests...")
        #test_image_processing()
        test_front_back_covers() 