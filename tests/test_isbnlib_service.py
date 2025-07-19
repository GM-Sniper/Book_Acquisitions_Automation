import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.isbnlib_service import ISBNService, quick_isbn_search, validate_isbn

def test_llm_parsing():
    """Test LLM parsing functionality"""
    print("🔍 Testing LLM Parsing...")
    
    service = ISBNService(debug=True, use_llm=True)
    result = service.search_by_isbn("9780140328721")
    
    if result:
        print("✅ LLM parsing successful!")
        print(f"   📚 Title: {result.get('title', 'N/A')}")
        print(f"   👤 Author: {result.get('author', 'N/A')}")
        print(f"   🏢 Publisher: {result.get('publisher', 'N/A')}")
        print(f"   🌐 Language: {result.get('language', 'N/A')}")
        print(f"   📖 Format: {result.get('format', 'N/A')}")
        print(f"   📋 Subjects: {result.get('subjects', 'N/A')}")
        print(f"   🤖 LLM Enhanced: {result.get('llm_enhanced', False)}")
        print(f"   🔗 Source: {result.get('source', 'N/A')}")
        
        # Debug: Show all keys in result
        print(f"   🔍 DEBUG - All keys in result: {list(result.keys())}")
        print(f"   🔍 DEBUG - Full result: {result}")
    else:
        print("❌ LLM parsing failed")
    print()

def test_arabic_book_llm():
    """Test Arabic book with LLM parsing"""
    print("🔍 Testing Arabic Book with LLM...")
    
    service = ISBNService(debug=True, use_llm=True)
    result = service.search_by_isbn("9789992195239")  # Don't merge to see pure LLM result
    
    if result:
        print("✅ Arabic book LLM parsing successful!")
        print(f"   📚 Title: {result.get('title', 'N/A')}")
        print(f"   👤 Author: {result.get('author', 'N/A')}")
        print(f"   🏢 Publisher: {result.get('publisher', 'N/A')}")
        print(f"   🌐 Language: {result.get('language', 'N/A')}")
        print(f"   📖 Format: {result.get('format', 'N/A')}")
        print(f"   📋 Subjects: {result.get('subjects', 'N/A')}")
        print(f"   🤖 LLM Enhanced: {result.get('llm_enhanced', False)}")
        print(f"   🔗 Source: {result.get('source', 'N/A')}")
    else:
        print("❌ Arabic book LLM parsing failed")
    print()

def test_merge_functionality():
    """Test merge functionality with multiple services"""
    print("🔍 Testing Merge Functionality...")
    
    service = ISBNService(debug=True, use_llm=True)
    result = service.search_by_isbn("9780140328721", merge_results=True)
    
    if result:
        print("✅ Merge functionality successful!")
        print(f"   📚 Title: {result.get('title', 'N/A')}")
        print(f"   👤 Author: {result.get('author', 'N/A')}")
        print(f"   🏢 Publisher: {result.get('publisher', 'N/A')}")
        print(f"   🌐 Language: {result.get('language', 'N/A')}")
        print(f"   📖 Format: {result.get('format', 'N/A')}")
        print(f"   📋 Subjects: {result.get('subjects', 'N/A')}")
        print(f"   🤖 LLM Enhanced: {result.get('llm_enhanced', True)}")
        print(f"   🔗 Source: {result.get('source', 'N/A')}")
        
        # Show merge info if available
        merge_info = result.get('merge_info', {})
        if merge_info:
            print(f"   📊 Merge Info:")
            print(f"      Total Results: {merge_info.get('total_results', 'N/A')}")
            print(f"      Enhanced Fields: {merge_info.get('enhanced_fields', [])}")
            print(f"      Sources: {merge_info.get('sources', [])}")
    else:
        print("❌ Merge functionality failed")
    print()

def test_quick_functions_llm():
    """Test quick functions with LLM parsing"""
    print("🔍 Testing Quick Functions with LLM...")
    
    # Test quick ISBN search with LLM
    result = quick_isbn_search("9780140328721", debug=True, use_llm=True, merge_results=True)
    if result:
        print("✅ Quick ISBN search with LLM works!")
        print(f"   📚 Title: {result.get('title', 'Unknown')}")
        print(f"   🤖 LLM Enhanced: {result.get('llm_enhanced', False)}")
        print(f"   🔗 Source: {result.get('source', 'N/A')}")
    else:
        print("❌ Quick ISBN search with LLM failed")
    print()

def test_llm_vs_no_llm():
    """Test LLM vs no LLM to see the difference"""
    print("🔍 Testing LLM vs No LLM...")
    
    # Test with LLM
    service_llm = ISBNService(debug=True, use_llm=True)
    result_llm = service_llm.search_by_isbn("9780140328721")
    
    # Test without LLM
    service_no_llm = ISBNService(debug=True, use_llm=False)
    result_no_llm = service_no_llm.search_by_isbn("9780140328721")
    
    print("🤖 WITH LLM:")
    if result_llm:
        print(f"   📚 Title: {result_llm.get('title', 'N/A')}")
        print(f"   🌐 Language: {result_llm.get('language', 'N/A')}")
        print(f"   📖 Format: {result_llm.get('format', 'N/A')}")
        print(f"   📋 Subjects: {result_llm.get('subjects', 'N/A')}")
        print(f"   🤖 LLM Enhanced: {result_llm.get('llm_enhanced', False)}")
    else:
        print("   ❌ LLM parsing failed")
    
    print("\n📚 WITHOUT LLM:")
    if result_no_llm:
        print(f"   📚 Title: {result_no_llm.get('title', 'N/A')}")
        print(f"   🌐 Language: {result_no_llm.get('language', 'N/A')}")
        print(f"   📖 Format: {result_no_llm.get('format', 'N/A')}")
        print(f"   📋 Subjects: {result_no_llm.get('subjects', 'N/A')}")
        print(f"   🤖 LLM Enhanced: {result_no_llm.get('llm_enhanced', False)}")
    else:
        print("   ❌ Regular parsing failed")
    print()

def run_llm_tests():
    """Run LLM-focused test suite"""
    print("🚀 Starting LLM ISBNlib Tests\n")
    print("=" * 50)
    
    test_llm_parsing()
    test_arabic_book_llm()
    test_merge_functionality()
    test_quick_functions_llm()
    test_llm_vs_no_llm()
    
    print("=" * 50)
    print("✅ All LLM tests completed!")

if __name__ == "__main__":
    run_llm_tests() 