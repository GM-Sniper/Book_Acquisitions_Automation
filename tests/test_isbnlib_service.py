import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.isbnlib_service import ISBNService, quick_isbn_search, validate_isbn
from src.utils.isbnlib_service import quick_title_search

def test_merge_functionality():
    """Test merge functionality with multiple services"""
    print("🔍 Testing Merge Functionality...")
    
    service = ISBNService(debug=True)
    result = service.search_by_isbn("9780140328721", merge_results=True)
    
    if result:
        print("✅ Merge functionality successful!")
        print(f"   📚 Title: {result.get('title', 'N/A')}")
        print(f"   👤 Author: {result.get('author', 'N/A')}")
        print(f"   🏢 Publisher: {result.get('publisher', 'N/A')}")
        print(f"   🌐 Language: {result.get('language', 'N/A')}")
        print(f"   📖 Format: {result.get('format', 'N/A')}")
        print(f"   📋 Subjects: {result.get('subjects', 'N/A')}")
        print(f"   🔗 Source: {result.get('source', 'N/A')}")
        
        # Show merge info if available
        # (No LLM merge info now)
    else:
        print("❌ Merge functionality failed")
    print()

def test_quick_functions():
    """Test quick functions without LLM parsing"""
    print("🔍 Testing Quick Functions (No LLM)...")
    
    # Test quick ISBN search
    result = quick_isbn_search("9780140328721", debug=True, merge_results=True)
    if result:
        print("✅ Quick ISBN search works!")
        print(f"   📚 Title: {result.get('title', 'Unknown')}")
        print(f"   🔗 Source: {result.get('source', 'N/A')}")
    else:
        print("❌ Quick ISBN search failed")
    print()

def test_title_author_search():
    """Test searching by title and author only (no ISBN)"""
    print("🔍 Testing Title/Author Search (No ISBN)...")
    title = "Matilda"
    authors = ["Roald Dahl"]
    result = quick_title_search(title, authors, debug=True)
    if result:
        print("✅ Title/Author search successful!")
        print(f"   📚 Title: {result.get('title', 'Unknown')}")
        print(f"   👤 Author: {result.get('author', 'Unknown')}")
        print(f"   🔗 Source: {result.get('source', 'N/A')}")
    else:
        print("❌ Title/Author search failed (isbnlib could not find by title/author)")
    print()

def run_no_llm_tests():
    """Run non-LLM-focused test suite"""
    print("🚀 Starting ISBNlib Tests (No LLM)\n")
    print("=" * 50)
    
    test_merge_functionality()
    test_quick_functions()
    test_title_author_search()
    
    print("=" * 50)
    print("✅ All non-LLM tests completed!")

if __name__ == "__main__":
    run_no_llm_tests()