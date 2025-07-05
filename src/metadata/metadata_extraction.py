from config.config import Config
from google import genai
import json
import re

def extract_json_from_text(text):
    """
    Extracts the first JSON object from a string using regex and parses it.
    Returns the parsed dict or None if not found/invalid.
    """
    match = re.search(r'\{.*?\}', text, re.DOTALL)
    if match:
        json_str = match.group(0)
        try:
            return json.loads(json_str)
        except Exception as e:
            print(f"Failed to parse JSON block: {e}")
    return None

def extract_metadata_with_gemini(ocr_text):
    """
    Uses Gemini to extract book metadata (title, authors, isbn10, isbn13) from OCR text.
    Args:
        ocr_text (str): The OCR-extracted text from the book cover.
    Returns:
        dict: {"title": ..., "authors": [...]} or None if error.
    """
    if not Config.GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY not found in environment variables.")

    client = genai.Client(api_key=Config.GEMINI_API_KEY)
    prompt = f'''
    Extract the following information from this text:
    - Book title
    - Author(s)

    Text:
    """{ocr_text}"""

    Return the result as JSON with keys: title, authors.
    If the text does not contain any of the information, return None for that key.
    '''
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        # Use the robust JSON extraction
        return extract_json_from_text(response.text)
    except Exception as e:
        print(f"Gemini metadata extraction failed: {e}")
        return None

def metadata_combiner(front_metadata, isbns):
    """
    Combines front cover metadata and ISBNs into a single dict.
    Args:
        front_metadata (dict): Should have 'title' and 'authors' keys.
        isbns (list): List of ISBN strings.
    Returns:
        dict: Combined metadata with 'title', 'authors', and 'isbns'.
    """
    return {
        "title": front_metadata.get("title"),
        "authors": front_metadata.get("authors"),
        "isbns": isbns
    } 