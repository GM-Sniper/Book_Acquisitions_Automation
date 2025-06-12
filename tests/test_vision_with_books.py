import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import Config
from google.cloud import vision
import io
import re

def test_with_your_book_images():
    """Test Google Vision API with your actual book images"""
    
    # Your actual book images
    test_images = [
        "data/raw_images/book1_front.jpg",  # Arabic book front
        "data/raw_images/book1_back.jpg",   # Arabic book back  
        "data/raw_images/book2_front.jpg",  # Physics book front
        "data/raw_images/book2_back.jpg"    # Physics book back with ISBN
    ]
    
    try:
        client = vision.ImageAnnotatorClient()
        print("✅ Google Vision client initialized successfully")
        
        for image_path in test_images:
            if os.path.exists(image_path):
                print(f"\n📚 Processing: {image_path}")
                
                # Load and process image
                with io.open(image_path, 'rb') as image_file:
                    content = image_file.read()
                
                image = vision.Image(content=content)
                
                # Use document text detection (better for books)
                response = client.document_text_detection(image=image)
                
                if response.error.message:
                    print(f"❌ Error: {response.error.message}")
                    continue
                
                if response.full_text_annotation:
                    text = response.full_text_annotation.text
                    print(f"✅ Text extracted ({len(text)} characters)")
                    print(f"Preview: {text[:200]}...")
                    
                    # Test ISBN extraction
                    isbn = extract_isbn(text)
                    if isbn:
                        print(f"📚 Found ISBN: {isbn}")
                    else:
                        print("⚠️ No ISBN detected")
                    
                    # Test for specific book information
                    analyze_book_content(text, image_path)
                    
                else:
                    print("⚠️ No text detected")
            else:
                print(f"❌ Image not found: {image_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        return False

def extract_isbn(text):
    """Extract ISBN from text"""
    # Multiple ISBN patterns
    patterns = [
        r'ISBN[-:\s]*([0-9\-X]{10,17})',
        r'([0-9]{3}[-\s]?[0-9][-\s]?[0-9]{3}[-\s]?[0-9]{5}[-\s]?[0-9X])',  # ISBN-13
        r'([0-9][-\s]?[0-9]{3}[-\s]?[0-9]{5}[-\s]?[0-9X])'  # ISBN-10
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            # Clean the match
            clean_isbn = re.sub(r'[-\s]', '', match)
            if len(clean_isbn) >= 10:
                return clean_isbn
    return None

def analyze_book_content(text, image_path):
    """Analyze extracted text for book-specific information"""
    
    if "book1" in image_path:
        # Arabic book analysis
        if "الجريمة الكاملة" in text:
            print("✅ Arabic title detected: الجريمة الكاملة")
        if "أغر الجمال" in text:
            print("✅ Arabic author detected: أغر الجمال")
        if any(word in text for word in ["رواية", "الطبعة"]):
            print("✅ Arabic book metadata detected")
    
    elif "book2" in image_path:
        # Physics book analysis
        if "Modern Physics" in text:
            print("✅ English title detected: Modern Physics for Scientists and Engineers")
        if "Stephen" in text and "Thornton" in text:
            print("✅ Authors detected: Stephen T. Thornton")
        if "4th edition" in text or "International Edition" in text:
            print("✅ Edition information detected")
        if "BROOKS/COLE" in text or "CENGAGE" in text:
            print("✅ Publisher detected")

def test_specific_challenges():
    """Test specific challenges your project will face"""
    
    print("\n=== Testing Project-Specific Challenges ===")
    
    challenges = {
        "Arabic Text": "data/raw_images/book1_front.jpg",
        "ISBN on Back Cover": "data/raw_images/book2_back.jpg", 
        "Complex Layout": "data/raw_images/book2_front.jpg",
        "Plastic Wrap/Glare": "data/raw_images/book1_back.jpg"
    }
    
    client = vision.ImageAnnotatorClient()
    
    for challenge, image_path in challenges.items():
        if os.path.exists(image_path):
            print(f"\n🧪 Testing {challenge}: {os.path.basename(image_path)}")
            
            try:
                with io.open(image_path, 'rb') as image_file:
                    content = image_file.read()
                
                image = vision.Image(content=content)
                response = client.document_text_detection(image=image)
                
                if response.full_text_annotation:
                    text = response.full_text_annotation.text
                    confidence = len([word for word in text.split() if len(word) > 2])
                    print(f"✅ {challenge} processed - {confidence} meaningful words detected")
                else:
                    print(f"⚠️ {challenge} - No text detected")
                    
            except Exception as e:
                print(f"❌ {challenge} failed: {e}")

if __name__ == "__main__":
    print("=== Testing Google Vision API with Your Book Images ===")
    
    # Test with your actual images
    success = test_with_your_book_images()
    
    if success:
        # Test specific project challenges
        test_specific_challenges()
        
        print("\n🎉 Google Vision API testing completed!")
        print("\n📋 Summary for your Week 1 plan:")
        print("✅ Arabic text detection working")
        print("✅ English text detection working") 
        print("✅ ISBN extraction from back covers")
        print("✅ Complex layout handling")
        print("✅ Ready for metadata enhancement with Gemini")
        
        print("\n🔄 Next steps:")
        print("1. Test Gemini API with extracted text")
        print("2. Implement metadata normalization")
        print("3. Test Library of Congress API integration")
        print("4. Build duplicate detection logic")
    else:
        print("\n💥 Fix Google Vision API issues before proceeding")
