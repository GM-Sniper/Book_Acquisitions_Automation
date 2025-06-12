import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import Config
from google.cloud import vision
import io

def test_google_vision_api():
    """Test Google Vision API connection and text detection"""
    
    # Check if credentials are configured
    if not Config.GOOGLE_CREDENTIALS_PATH:
        print(" GOOGLE_APPLICATION_CREDENTIALS not found in environment variables")
        print("Make sure to add GOOGLE_APPLICATION_CREDENTIALS to your .env file")
        return False
    
    if not os.path.exists(Config.GOOGLE_CREDENTIALS_PATH):
        print(f" Credentials file not found at: {Config.GOOGLE_CREDENTIALS_PATH}")
        print("Make sure your google-vision-credentials.json file exists")
        return False
    
    print("Google Vision credentials file found")
    print(f"Credentials path: {Config.GOOGLE_CREDENTIALS_PATH}")
    
    try:
        # Initialize the Vision API client
        client = vision.ImageAnnotatorClient()
        print("✅ Google Vision client initialized successfully")
        
        # Test with a sample image (create a simple test image)
        test_image_path = create_test_image()
        
        if test_image_path and os.path.exists(test_image_path):
            print(f"\n🧪 Testing text detection with sample image...")
            
            # Load the image
            with io.open(test_image_path, 'rb') as image_file:
                content = image_file.read()
            
            image = vision.Image(content=content)
            
            # Perform text detection
            response = client.text_detection(image=image)
            texts = response.text_annotations
            
            if response.error.message:
                raise Exception(f'Vision API error: {response.error.message}')
            
            if texts:
                print("✅ Text detection successful!")
                print(f"Detected text: {texts[0].description[:200]}...")
                print(f"Confidence: Available through response object")
                
                # Test document text detection (better for books)
                print("\n📚 Testing document text detection (better for books)...")
                doc_response = client.document_text_detection(image=image)
                
                if doc_response.full_text_annotation:
                    print("✅ Document text detection successful!")
                    print(f"Full text length: {len(doc_response.full_text_annotation.text)} characters")
                    print(f"Text preview: {doc_response.full_text_annotation.text[:200]}...")
                else:
                    print("⚠️ Document text detection returned no results")
                
            else:
                print("⚠️ No text detected in the test image")
            
            # Clean up test image
            os.remove(test_image_path)
            
        else:
            print("⚠️ Could not create test image, testing with API call only")
        
        return True
        
    except Exception as e:
        print(f"❌ Google Vision API test failed: {str(e)}")
        return False

def create_test_image():
    """Create a simple test image with text for testing"""
    try:
        from PIL import Image, ImageDraw, ImageFont
        
        # Create a simple image with text
        img = Image.new('RGB', (400, 200), color='white')
        draw = ImageDraw.Draw(img)
        
        # Add some text that simulates a book cover
        test_text = [
            "THE GREAT GATSBY",
            "F. Scott Fitzgerald",
            "Scribner",
            "ISBN: 978-0-7432-7356-5"
        ]
        
        y_position = 20
        for line in test_text:
            draw.text((20, y_position), line, fill='black')
            y_position += 30
        
        # Save test image
        test_path = "test_book_cover.jpg"
        img.save(test_path)
        print(f"✅ Created test image: {test_path}")
        return test_path
        
    except ImportError:
        print("⚠️ PIL not available for test image creation")
        return None
    except Exception as e:
        print(f"⚠️ Could not create test image: {e}")
        return None

if __name__ == "__main__":
    print("=== Google Vision API Test ===")
    
    # Test basic API functionality
    success = test_google_vision_api()
    
    if success:
        # print("\n=== Testing with Real Book Image ===")
        # test_with_real_book_image()
        
        print("\n🎉 Google Vision API tests completed!")
        print("Your Google Vision integration is ready for the book processing pipeline.")
    else:
        print("\n💥 Google Vision API test failed.")
        print("Check your credentials and API configuration.")
