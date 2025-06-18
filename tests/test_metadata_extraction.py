import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.metadata.metadata_extraction import extract_metadata_with_gemini

def test_metadata_extraction():
    # Example OCR text from a book cover
    ocr_text = """
    Modern Physics for Scientists and Engineers
    Stephen T. Thornton
    Andrew Rex
    4th edition
    """
    result = extract_metadata_with_gemini(ocr_text)
    print("Extracted Metadata:")
    print(result)

if __name__ == "__main__":
    test_metadata_extraction()
