import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.google_books import (
    search_book_by_isbn, 
    search_book_by_title_author, 
    search_arabic_book, 
    extract_book_metadata, 
)

def test_isbn_search():
    """Test ISBN search functionality"""
    print("🔍 Testing ISBN Search...")
    isbn = "9780140328721"  # Example: Matilda by Roald Dahl
    result = search_book_by_isbn(isbn)
    
    if result and result.get('items'):
        print("✅ ISBN search successful")
        metadata = extract_book_metadata(result)
        print(f"   Title: {metadata.get('title', 'N/A')}")
        print(f"   Author: {metadata.get('author', 'N/A')}")
        print(f"   ISBN: {metadata.get('isbn', 'N/A')}")
    else:
        print("❌ ISBN search failed or no results")
    print()

def test_title_author_search():
    """Test title and author search functionality"""
    print("🔍 Testing Title + Author Search...")
    title = "Matilda"
    authors = ["Roald Dahl"]
    result = search_book_by_title_author(title, authors)
    
    if result and result.get('items'):
        print("✅ Title + author search successful")
        metadata = extract_book_metadata(result)
        print(f"   Title: {metadata.get('title', 'N/A')}")
        print(f"   Author: {metadata.get('author', 'N/A')}")
        print(f"   Publisher: {metadata.get('publisher', 'N/A')}")
        print(f"   Published: {metadata.get('published_date', 'N/A')}")
    else:
        print("❌ Title + author search failed or no results")
    print()

def test_arabic_book_search():
    """Test Arabic book search functionality"""
    print("🔍 Testing Arabic Book Search...")
    # Example Arabic book title (you can replace with actual Arabic titles)
    title = "ألف ليلة وليلة"  # One Thousand and One Nights
    authors = []
    result = search_arabic_book(title, authors)
    
    if result and result.get('items'):
        print("✅ Arabic book search successful")
        metadata = extract_book_metadata(result)
        print(f"   Title: {metadata.get('title', 'N/A')}")
        print(f"   Author: {metadata.get('author', 'N/A')}")
        print(f"   Language: {metadata.get('language', 'N/A')}")
    else:
        print("❌ Arabic book search failed or no results")
    print()

def test_metadata_extraction():
    """Test metadata extraction from Google Books response"""
    print("🔍 Testing Metadata Extraction...")
    # First get some data to extract from
    isbn = "9780140328721"
    result = search_book_by_isbn(isbn)
    
    if result and result.get('items'):
        metadata = extract_book_metadata(result)
        print("✅ Metadata extraction successful")
        print("📚 Extracted fields:")
        for key, value in metadata.items():
            print(f"   {key}: {value}")
    else:
        print("❌ Metadata extraction failed - no data to extract")
    print()

def test_error_handling():
    """Test error handling with invalid inputs"""
    print("🔍 Testing Error Handling...")
    
    # Test with invalid ISBN
    try:
        result = search_book_by_isbn("invalid_isbn_123")
        if not result.get('items'):
            print("✅ Invalid ISBN handled gracefully")
        else:
            print("❌ Invalid ISBN returned unexpected results")
    except Exception as e:
        print(f"❌ Invalid ISBN caused error: {e}")
    
    # Test with empty title
    try:
        result = search_book_by_title_author("", [])
        if not result.get('items'):
            print("✅ Empty title handled gracefully")
        else:
            print("❌ Empty title returned unexpected results")
    except Exception as e:
        print(f"❌ Empty title caused error: {e}")
    print()

def run_all_tests():
    """Run all Google Books API tests"""
    print("🚀 Starting Google Books API Tests\n")
    print("=" * 50)
    
    test_isbn_search()
    test_title_author_search()
    test_arabic_book_search()
    test_metadata_extraction()
    test_error_handling()
    
    print("=" * 50)
    print("✅ All tests completed!")

if __name__ == "__main__":
    run_all_tests() 