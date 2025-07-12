import base64
import json
import re
import io
from PIL import Image
import numpy as np
from google import genai
from config.config import Config


def encode_image_to_base64(image_data):
    """
    Encode image data to base64 string for Gemini API.
    Args:
        image_data: Can be bytes, PIL Image, numpy array, or file path
    Returns:
        str: Base64 encoded image string
    """
    if isinstance(image_data, str):
        # Assume it's a file path
        with open(image_data, 'rb') as f:
            image_bytes = f.read()
    elif isinstance(image_data, bytes):
        image_bytes = image_data
    elif isinstance(image_data, Image.Image):
        # Convert PIL Image to bytes
        buffer = io.BytesIO()
        image_data.save(buffer, format='JPEG')
        image_bytes = buffer.getvalue()
    elif isinstance(image_data, np.ndarray):
        # Convert numpy array to PIL Image then to bytes
        if image_data.dtype != np.uint8:
            image_data = (image_data * 255).astype(np.uint8)
        pil_image = Image.fromarray(image_data)
        buffer = io.BytesIO()
        pil_image.save(buffer, format='JPEG')
        image_bytes = buffer.getvalue()
    else:
        raise ValueError(f"Unsupported image data type: {type(image_data)}")
    
    return base64.b64encode(image_bytes).decode('utf-8')


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


def extract_book_metadata_from_images(image_data_list, prompt_type="detailed"):
    """
    Extract book metadata from multiple images using Gemini Vision.
    Args:
        image_data_list: List of image data (bytes, PIL Image, numpy array, or file path)
        prompt_type: "basic", "detailed", or "comprehensive"
    Returns:
        dict: Extracted metadata in JSON format
    """
    if not Config.GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY not found in environment variables.")

    # Encode all images to base64
    base64_images = []
    for image_data in image_data_list:
        base64_image = encode_image_to_base64(image_data)
        base64_images.append(base64_image)
    
    # Define prompts based on type
    prompts = {
        "basic": f"""
        Analyze these {len(base64_images)} book cover images and extract the following information:
        - Book title
        - Author(s)
        
        Return the result as a JSON object with keys: title, authors.
        If any information is not visible or unclear, set that value to null.
        """,
        
        "detailed": f"""
        Analyze these {len(base64_images)} book cover images and extract the following information:
        - Book title
        - Author(s)
        - Publisher (if visible)
        - Publication year (if visible)
        - ISBN (if visible)
        - Edition (if visible)
        - Series information (if part of a series)
        
        Return the result as a JSON object with keys: title, authors, publisher, year, isbn, edition, series.
        If any information is not visible or unclear, set that value to null.
        """,
        
        "comprehensive": f"""
        Perform a comprehensive analysis of these {len(base64_images)} book cover images and extract:
        - Book title
        - Author(s)
        - Publisher
        - Publication year
        - ISBN (both ISBN-10 and ISBN-13 if available)
        - Edition
        - Series information
        - Genre/category (if evident from cover)
        - Language (if not English)
        - Any additional text visible on the covers
        
        Return the result as a JSON object with keys: title, authors, publisher, year, isbn10, isbn13, edition, series, genre, language, additional_text.
        If any information is not visible or unclear, set that value to null.
        """
    }
    
    prompt = prompts.get(prompt_type, prompts["detailed"])
    
    try:
        client = genai.Client(api_key=Config.GEMINI_API_KEY)
        
        # Create the content with multiple images
        parts = [{"text": prompt}]
        for base64_image in base64_images:
            parts.append({
                "inline_data": {
                    "mime_type": "image/jpeg",
                    "data": base64_image
                }
            })
        
        content = [
            {
                "role": "user",
                "parts": parts
            }
        ]
        
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=content
        )
        
        # Extract JSON from response
        return extract_json_from_text(response.text)
        
    except Exception as e:
        print(f"Gemini multi-image processing failed: {e}")
        return None


def extract_book_metadata_from_image(image_data, prompt_type="detailed"):
    """
    Extract book metadata from a single image using Gemini Vision.
    Args:
        image_data: Image data (bytes, PIL Image, numpy array, or file path)
        prompt_type: "basic", "detailed", or "comprehensive"
    Returns:
        dict: Extracted metadata in JSON format
    """
    return extract_book_metadata_from_images([image_data], prompt_type)


def infer_missing_metadata(metadata, image_data_list=None):
    """
    Use Gemini's knowledge to fill in missing metadata gaps.
    Args:
        metadata (dict): Initial metadata from image analysis
        image_data_list: List of image data for visual context
    Returns:
        dict: Enhanced metadata with inferred information
    """
    if not Config.GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY not found in environment variables.")
    
    if not metadata or not metadata.get('title'):
        return metadata
    
    # Create a more conservative prompt for inference with web search
    prompt = f"""
    Based on your knowledge of books and publishing, help fill in missing information for this book. Be very conservative and only provide information you are highly confident about.
    
    Current metadata: {json.dumps(metadata, indent=2)}
    
    IMPORTANT GUIDELINES:
    - Only fill in information you are 90%+ confident about
    - If you're unsure, leave it as null
    - Do NOT guess publication years unless you have strong evidence
    - Do NOT guess publishers unless you have strong evidence
    - Do NOT guess ISBNs - these are unique identifiers that cannot be inferred
    - For genre, only infer if the title/author clearly indicates it
    - Be very careful with children's literature classification - only if clearly a children's book
    
    Use your knowledge and any websearch and available information to:
    - Look up publication years for books you recognize
    - Find publisher information for known books
    - Verify ISBNs if visible on the cover
    - Get accurate genre classifications
    - Find edition information
    
    Focus on:
    - Publication year: Use web search to find accurate publication dates
    - Publisher: Use web search to find the correct publisher
    - Genre: Use web search to verify genre classifications
    - Language: Only if title clearly indicates non-English
    - Edition: Use web search to find edition information
    - ISBN: Use web search to verify ISBNs if visible
    
    Return the enhanced metadata as a JSON object. Only fill in fields that are currently null or missing AND that you are highly confident about.
    If you cannot reasonably infer a value with high confidence, keep it as null.
    """
    
    try:
        client = genai.Client(api_key=Config.GEMINI_API_KEY)
        
        if image_data_list:
            # Include images in the prompt for visual context
            parts = [{"text": prompt}]
            for image_data in image_data_list:
                base64_image = encode_image_to_base64(image_data)
                parts.append({
                    "inline_data": {
                        "mime_type": "image/jpeg",
                        "data": base64_image
                    }
                })
            
            content = [
                {
                    "role": "user",
                    "parts": parts
                }
            ]
        else:
            # Text-only prompt
            content = [
                {
                    "role": "user",
                    "parts": [{"text": prompt}]
                }
            ]
        
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=content
        )
        
        # Extract enhanced metadata
        enhanced_metadata = extract_json_from_text(response.text)
        
        if enhanced_metadata:
            # Merge the enhanced metadata with original, preferring enhanced values
            merged = metadata.copy()
            for key, value in enhanced_metadata.items():
                if value and value != "null" and value != "None":
                    merged[key] = value
            return merged
        
        return metadata
        
    except Exception as e:
        print(f"Metadata inference failed: {e}")
        return metadata


def validate_book_metadata(metadata):
    """
    Validate and clean extracted metadata.
    Args:
        metadata (dict): Raw metadata from Gemini
    Returns:
        dict: Cleaned and validated metadata
    """
    if not metadata:
        return None
    
    # Clean and validate fields
    cleaned = {}
    
    # Title validation
    if metadata.get('title'):
        title = str(metadata['title']).strip()
        if title and title.lower() not in ['null', 'none', 'unknown']:
            cleaned['title'] = title
    
    # Authors validation
    if metadata.get('authors'):
        authors = metadata['authors']
        if isinstance(authors, list):
            authors = [str(author).strip() for author in authors if author and str(author).strip()]
        elif isinstance(authors, str):
            authors = [author.strip() for author in authors.split(',') if author.strip()]
        else:
            authors = []
        
        if authors:
            cleaned['authors'] = authors
    
    # Other fields
    for field in ['publisher', 'year', 'isbn', 'isbn10', 'isbn13', 'edition', 'series', 'genre', 'language']:
        if metadata.get(field):
            value = str(metadata[field]).strip()
            if value and value.lower() not in ['null', 'none', 'unknown']:
                cleaned[field] = value
    
    return cleaned


def process_book_images(image_data_list, prompt_type="detailed", infer_missing=True):
    """
    Main function to process multiple book images with Gemini and extract metadata.
    Args:
        image_data_list: List of image data (bytes, PIL Image, numpy array, or file path)
        prompt_type: "basic", "detailed", or "comprehensive"
        infer_missing: Whether to use Gemini's knowledge to fill missing gaps
    Returns:
        dict: Extracted and validated metadata
    """
    metadata = extract_book_metadata_from_images(image_data_list, prompt_type)
    validated_metadata = validate_book_metadata(metadata)
    
    if infer_missing and validated_metadata:
        enhanced_metadata = infer_missing_metadata(validated_metadata, image_data_list)
        return enhanced_metadata
    
    return validated_metadata


def process_book_image(image_data, prompt_type="detailed", infer_missing=True):
    """
    Main function to process a single book image with Gemini and extract metadata.
    Args:
        image_data: Image data (bytes, PIL Image, numpy array, or file path)
        prompt_type: "basic", "detailed", or "comprehensive"
        infer_missing: Whether to use Gemini's knowledge to fill missing gaps
    Returns:
        dict: Extracted and validated metadata
    """
    return process_book_images([image_data], prompt_type, infer_missing)
