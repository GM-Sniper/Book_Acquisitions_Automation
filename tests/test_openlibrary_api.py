import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.openlibrary import OpenLibraryAPI

def test_openlibrary_isbn_search():
    """Test OpenLibrary ISBN search functionality"""
    print("🔍 Testing OpenLibrary ISBN Search...")
    api = OpenLibraryAPI()
    isbn = "9781691706631"  # Example: Control Your Mind and Master Your Feelings
    result = api.search_by_isbn(isbn)
    
    if result:
        print("✅ OpenLibrary ISBN search successful")
        print(f"   Title: {result.get('title', 'N/A')}")
        print(f"   Author: {result.get('author', 'N/A')}")
        print(f"   Publisher: {result.get('publisher', 'N/A')}")
        print(f"   Published: {result.get('published_date', 'N/A')}")
        print(f"   ISBN: {result.get('isbn', 'N/A')}")
        print(f"   Source: {result.get('source', 'N/A')}")
    else:
        print("❌ OpenLibrary ISBN search failed or no results")
    print()

def test_openlibrary_title_author_search():
    """Test OpenLibrary title and author search functionality"""
    print("🔍 Testing OpenLibrary Title + Author Search...")
    api = OpenLibraryAPI()
    title = "Control Your Mind and Master Your Feelings"
    authors = ["Eric Robertson"]
    result = api.search_by_title_author(title, authors)
    
    if result:
        print("✅ OpenLibrary title + author search successful")
        print(f"   Title: {result.get('title', 'N/A')}")
        print(f"   Author: {result.get('author', 'N/A')}")
        print(f"   Publisher: {result.get('publisher', 'N/A')}")
        print(f"   Published: {result.get('published_date', 'N/A')}")
        print(f"   Source: {result.get('source', 'N/A')}")
    else:
        print("❌ OpenLibrary title + author search failed or no results")
    print()

def test_openlibrary_arabic_search():
    """Test OpenLibrary Arabic book search functionality"""
    print("🔍 Testing OpenLibrary Arabic Book Search...")
    api = OpenLibraryAPI()
    # Example Arabic book title
    title = "مولانا"  # One Thousand and One Nights
    authors = ["إبراهيم عيسى"]
    result = api.search_arabic_book(title, authors)
    
    if result:
        print("✅ OpenLibrary Arabic book search successful")
        print(f"   Title: {result.get('title', 'N/A')}")
        print(f"   Author: {result.get('author', 'N/A')}")
        print(f"   Publisher: {result.get('publisher', 'N/A')}")
        print(f"   Source: {result.get('source', 'N/A')}")
    else:
        print("❌ OpenLibrary Arabic book search failed or no results")
    print()



def test_openlibrary_error_handling():
    """Test OpenLibrary error handling with invalid inputs"""
    print("🔍 Testing OpenLibrary Error Handling...")
    api = OpenLibraryAPI()
    
    # Test with invalid ISBN
    try:
        result = api.search_by_isbn("invalid_isbn_123")
        if not result:
            print("✅ Invalid ISBN handled gracefully")
        else:
            print("❌ Invalid ISBN returned unexpected results")
    except Exception as e:
        print(f"❌ Invalid ISBN caused error: {e}")
    
    # Test with empty title
    try:
        result = api.search_by_title_author("", [])
        if not result:
            print("✅ Empty title handled gracefully")
        else:
            print("❌ Empty title returned unexpected results")
    except Exception as e:
        print(f"❌ Empty title caused error: {e}")
    print()

def test_openlibrary_metadata_format():
    """Test that OpenLibrary metadata matches our library catalog format"""
    print("🔍 Testing OpenLibrary Metadata Format...")
    api = OpenLibraryAPI()
    isbn = "9781691706631"
    result = api.search_by_isbn(isbn)
    
    if result:
        print("✅ OpenLibrary metadata format check")
        required_fields = ['title', 'author', 'publisher', 'published_date', 'd_o_pub', 'oclc_no', 'lc_no', 'isbn_10', 'isbn_13', 'isbn', 'source']
        missing_fields = []
        
        for field in required_fields:
            if field not in result:
                missing_fields.append(field)
        
        if missing_fields:
            print(f"   ❌ Missing fields: {missing_fields}")
        else:
            print("   ✅ All required fields present")
            print("   📚 Library catalog format:")
            for field in required_fields:
                print(f"      {field}: {result.get(field, 'N/A')}")
    else:
        print("❌ No data to check format")
    print()

def run_all_openlibrary_tests():
    """Run all OpenLibrary API tests"""
    print("🚀 Starting OpenLibrary API Tests\n")
    print("=" * 50)
    
    test_openlibrary_isbn_search()
    test_openlibrary_title_author_search()
    test_openlibrary_arabic_search()
    test_openlibrary_error_handling()
    test_openlibrary_metadata_format()
    
    print("=" * 50)
    print("✅ All OpenLibrary tests completed!")

if __name__ == "__main__":
    run_all_openlibrary_tests() 