import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import Config
from google import genai

def test_gemini_api():
    """Test Gemini API connection and basic functionality"""
    
    # Check if API key is loaded
    if not Config.GEMINI_API_KEY:
        print(" GEMINI_API_KEY not found in environment variables")
        print("Make sure to add GEMINI_API_KEY=your-api-key to your .env file")
        return False
    
    print(" Gemini API key loaded successfully")
    print(f"API key starts with: {Config.GEMINI_API_KEY[:10]}...")
    
    try:
        # Initialize the client
        client = genai.Client(api_key=Config.GEMINI_API_KEY)
        print(" Gemini client initialized successfully")
        
        # Test basic text generation
        print("\n Testing basic text generation...")
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents="Generate a brief description of what metadata extraction means in library science."
        )
        
        print(" Gemini API call successful!")
        print(f"Response length: {len(response.text)} characters")
        print(f"Response preview: {response.text[:200]}...")
        
        # Test with book-related prompt (relevant to your project)
        print("\n Testing book metadata extraction prompt...")
        book_prompt = """
        From this sample book information, extract structured metadata:
        "The Great Gatsby by F. Scott Fitzgerald, published by Scribner, 1925"
        
        Return as JSON with fields: title, author, publisher, year
        """
        
        metadata_response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=book_prompt
        )
        
        print(" Book metadata extraction test successful!")
        print(f"Metadata response: {metadata_response.text[:300]}...")
        
        return True
        
    except Exception as e:
        print(f" Gemini API test failed: {str(e)}")
        return False

if __name__ == "__main__":
    print("=== Gemini API Test ===")
    success = test_gemini_api()
    
    if success:
        print("\n All Gemini API tests passed!")
        print("Your Gemini integration is ready for the book processing pipeline.")
    else:
        print("\n Gemini API test failed. Check your API key and internet connection.")
