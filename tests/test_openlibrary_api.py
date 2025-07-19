import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.openlibrary import OpenLibraryAPI

def test_openlibrary_isbn_search():
    """Test OpenLibrary ISBN search functionality"""
    print("🔍 Testing OpenLibrary ISBN Search...")
    api = OpenLibraryAPI(debug=True, rate_limit=1.0)  # Enable debug mode with rate limiting
    isbn = "9780140328721"  # Fantastic Mr. Fox
    result = api.search_by_isbn(isbn)
    
    if result:
        print("✅ OpenLibrary ISBN search successful")
        print(f"   📚 Title: {result.get('title', 'N/A')}")
        print(f"   👤 Author: {result.get('author', 'N/A')}")
        print(f"   🏢 Publisher: {result.get('publisher', 'N/A')}")
        print(f"   📅 Published: {result.get('published_date', 'N/A')}")
        print(f"   📖 ISBN-10: {result.get('isbn_10', 'N/A')}")
        print(f"   📖 ISBN-13: {result.get('isbn_13', 'N/A')}")
        print(f"   🔗 Combined ISBN: {result.get('isbn', 'N/A')}")
        print(f"   📋 OCLC: {result.get('oclc_no', 'N/A')}")
        print(f"   📋 LC: {result.get('lc_no', 'N/A')}")
        print(f"   🔗 Source: {result.get('source', 'N/A')}")
        
        # Verify ISBN matches
        if isbn in result.get('isbn', ''):
            print("   ✅ ISBN verification successful")
        else:
            print("   ⚠️  ISBN verification failed - ISBN not found in result")
    else:
        print("❌ OpenLibrary ISBN search failed or no results")
    print()

def test_openlibrary_title_author_search():
    """Test OpenLibrary title and author search functionality"""
    print("🔍 Testing OpenLibrary Title + Author Search...")
    api = OpenLibraryAPI(debug=True, rate_limit=1.0)
    title = "Fantastic Mr. Fox"
    authors = ["Roald Dahl"]
    result = api.search_by_title_author(title, authors)
    
    if result:
        print("✅ OpenLibrary title + author search successful")
        print(f"   📚 Title: {result.get('title', 'N/A')}")
        print(f"   👤 Author: {result.get('author', 'N/A')}")
        print(f"   🏢 Publisher: {result.get('publisher', 'N/A')}")
        print(f"   📅 Published: {result.get('published_date', 'N/A')}")
        print(f"   📖 ISBN-10: {result.get('isbn_10', 'N/A')}")
        print(f"   📖 ISBN-13: {result.get('isbn_13', 'N/A')}")
        print(f"   📋 OCLC: {result.get('oclc_no', 'N/A')}")
        print(f"   📋 LC: {result.get('lc_no', 'N/A')}")
        print(f"   🔗 Source: {result.get('source', 'N/A')}")
    else:
        print("❌ OpenLibrary title + author search failed or no results")
    print()

def test_openlibrary_arabic_search():
    """Test OpenLibrary Arabic book search functionality"""
    print("🔍 Testing OpenLibrary Arabic Book Search...")
    api = OpenLibraryAPI()
    # Test with the actual Arabic book
    title = "مولانا"
    authors = ["إبراهيم عيسى"]
    result = api.search_by_title_author(title, authors)
    
    if result:
        print("✅ OpenLibrary Arabic book search successful")
        print(f"   Title: {result.get('title', 'N/A')}")
        print(f"   Author: {result.get('author', 'N/A')}")
        print(f"   Publisher: {result.get('publisher', 'N/A')}")
        print(f"   Published: {result.get('published_date', 'N/A')}")
        print(f"   ISBN-10: {result.get('isbn_10', 'N/A')}")
        print(f"   ISBN-13: {result.get('isbn_13', 'N/A')}")
        print(f"   Source: {result.get('source', 'N/A')}")
    else:
        print("❌ OpenLibrary Arabic book search failed or no results")
    print()

def test_openlibrary_isbn_vs_title_search():
    """Compare ISBN search vs title search for the same book"""
    print("🔍 Testing ISBN Search vs Title Search Comparison...")
    api = OpenLibraryAPI()
    isbn = "9780140328721"
    title = "Fantastic Mr. Fox"
    authors = ["Roald Dahl"]
    
    # ISBN search
    isbn_result = api.search_by_isbn(isbn)
    print("📚 ISBN Search Results:")
    if isbn_result:
        print(f"   Title: {isbn_result.get('title', 'N/A')}")
        print(f"   ISBN-10: {isbn_result.get('isbn_10', 'N/A')}")
        print(f"   ISBN-13: {isbn_result.get('isbn_13', 'N/A')}")
        print(f"   Publisher: {isbn_result.get('publisher', 'N/A')}")
        print(f"   Source: {isbn_result.get('source', 'N/A')}")
    else:
        print("   ❌ No results from ISBN search")
    
    # Title search
    title_result = api.search_by_title_author(title, authors)
    print("📚 Title Search Results:")
    if title_result:
        print(f"   Title: {title_result.get('title', 'N/A')}")
        print(f"   ISBN-10: {title_result.get('isbn_10', 'N/A')}")
        print(f"   ISBN-13: {title_result.get('isbn_13', 'N/A')}")
        print(f"   Publisher: {title_result.get('publisher', 'N/A')}")
        print(f"   Source: {title_result.get('source', 'N/A')}")
    else:
        print("   ❌ No results from title search")
    
    # Compare completeness
    if isbn_result and title_result:
        isbn_fields = sum(1 for v in isbn_result.values() if v)
        title_fields = sum(1 for v in title_result.values() if v)
        print(f"📊 Data Completeness: ISBN search has {isbn_fields} fields, Title search has {title_fields} fields")
        if isbn_fields > title_fields:
            print("   ✅ ISBN search provides more complete data")
        elif title_fields > isbn_fields:
            print("   ✅ Title search provides more complete data")
        else:
            print("   ⚖️  Both searches provide similar data completeness")
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
    
    # Test with None values
    try:
        result = api.search_by_title_author(None, None)
        if not result:
            print("✅ None values handled gracefully")
        else:
            print("❌ None values returned unexpected results")
    except Exception as e:
        print(f"❌ None values caused error: {e}")
    print()

def test_openlibrary_metadata_format():
    """Test that OpenLibrary metadata matches our library catalog format"""
    print("🔍 Testing OpenLibrary Metadata Format...")
    api = OpenLibraryAPI()
    isbn = "9780140328721"
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
                value = result.get(field, 'N/A')
                print(f"      {field}: {value}")
        
        # Check data quality
        print("   📊 Data Quality Check:")
        if result.get('title'):
            print("      ✅ Title present")
        else:
            print("      ❌ Title missing")
            
        if result.get('isbn_10') or result.get('isbn_13'):
            print("      ✅ ISBN data present")
        else:
            print("      ❌ ISBN data missing")
            
        if result.get('publisher'):
            print("      ✅ Publisher present")
        else:
            print("      ⚠️  Publisher missing")
            
        if result.get('published_date'):
            print("      ✅ Publication date present")
        else:
            print("      ⚠️  Publication date missing")
    else:
        print("❌ No data to check format")
    print()

def test_openlibrary_multiple_isbns():
    """Test OpenLibrary with multiple ISBNs to see which one works best"""
    print("🔍 Testing OpenLibrary with Multiple ISBNs...")
    api = OpenLibraryAPI(debug=False, rate_limit=1.0)  # Disable debug for cleaner output
    test_isbns = [
        "9780140328721",  # Fantastic Mr. Fox
        "9789992195239",  # مولانا (Mowlana)
        "9781691706631",  # Control Your Mind and Master Your Feelings
        "9780141439518",  # Pride and Prejudice
    ]
    
    for isbn in test_isbns:
        print(f"   Testing ISBN: {isbn}")
        result = api.search_by_isbn(isbn)
        if result:
            print(f"      ✅ Found: {result.get('title', 'Unknown')}")
            print(f"      👤 Author: {result.get('author', 'Unknown')}")
            print(f"      🏢 Publisher: {result.get('publisher', 'Unknown')}")
            print(f"      📅 Published: {result.get('published_date', 'Unknown')}")
            print(f"      📖 ISBN-10: {result.get('isbn_10', 'Unknown')}")
            print(f"      📖 ISBN-13: {result.get('isbn_13', 'Unknown')}")
            print(f"      📋 OCLC: {result.get('oclc_no', 'Unknown')}")
            print(f"      📋 LC: {result.get('lc_no', 'Unknown')}")
        else:
            print(f"      ❌ Not found")
        print()
    print()

def run_all_openlibrary_tests():
    """Run all OpenLibrary API tests"""
    print("🚀 Starting OpenLibrary API Tests\n")
    print("=" * 60)
    
    test_openlibrary_isbn_search()
    test_openlibrary_title_author_search()
    test_openlibrary_arabic_search()
    test_openlibrary_isbn_vs_title_search()
    test_openlibrary_error_handling()
    test_openlibrary_metadata_format()
    test_openlibrary_multiple_isbns()
    
    print("=" * 60)
    print("✅ All OpenLibrary tests completed!")

if __name__ == "__main__":
    run_all_openlibrary_tests() 