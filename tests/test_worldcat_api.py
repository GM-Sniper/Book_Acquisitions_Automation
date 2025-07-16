"""
WorldCat API v2 Test Suite
=========================

Purpose:
--------
This test suite evaluates the WorldCat Search API v2 integration for book metadata retrieval,
specifically focusing on ISBN and title/author search using the new OAuth2/JSON API.

Key Functionality Tested:
------------------------
1. ISBN Search (search_by_isbn)
2. Title/Author Search (search_by_title_author)
3. Multiple ISBN Search (search_multiple_isbns)

Test Data:
---------
Uses well-known books for testing:
- To Kill a Mockingbird (ISBN: 9780446310789)
- Pride and Prejudice (ISBN: 9780141439518)
- The Great Gatsby (ISBN: 9780743273565)

Note:
-----
This test suite requires valid WorldCat v2 credentials to be set in environment variables.
Set WORLDCAT_CLIENT_ID and WORLDCAT_CLIENT_SECRET in your .env file before running tests.
"""

import sys
import os
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.utils.worldcat import WorldCatAPIv2
from config.config import Config

def test_worldcat_api_v2_initialization():
    """Test WorldCatAPIv2 client initialization"""
    print("Testing WorldCatAPIv2 initialization...")
    try:
        api = WorldCatAPIv2()
        print(f"✓ WorldCatAPIv2 client initialized successfully")
        print(f"  - API Base URL: {api.api_base_url}")
        print(f"  - Client ID configured: {'Yes' if api.client_id else 'No'}")
        print(f"  - Client Secret configured: {'Yes' if api.client_secret else 'No'}")
        return True
    except Exception as e:
        print(f"✗ WorldCatAPIv2 initialization failed: {e}")
        return False

def test_isbn_search_v2():
    """Test ISBN search functionality (v2)"""
    print("\nTesting WorldCat v2 ISBN search...")
    api = WorldCatAPIv2()
    test_isbns = [
        "9780446310789",  # To Kill a Mockingbird
        "9780141439518",  # Pride and Prejudice
        "9780743273565"   # The Great Gatsby
    ]
    for isbn in test_isbns:
        print(f"\nSearching for ISBN: {isbn}")
        try:
            metadata = api.search_by_isbn(isbn)
            if metadata:
                print(f"✓ Found metadata for ISBN {isbn}")
                print(f"  - Title: {metadata.get('title', 'N/A')}")
                print(f"  - Author: {metadata.get('author', 'N/A')}")
                print(f"  - Publisher: {metadata.get('publisher', 'N/A')}")
                print(f"  - Published Date: {metadata.get('published_date', 'N/A')}")
                print(f"  - OCLC Number: {metadata.get('oclc_no', 'N/A')}")
                print(f"  - ISBN: {metadata.get('isbn', 'N/A')}")
                print(f"  - Source: {metadata.get('source', 'N/A')}")
            else:
                print(f"✗ No metadata found for ISBN {isbn}")
        except Exception as e:
            print(f"✗ Error searching for ISBN {isbn}: {e}")
        time.sleep(2)

def test_title_author_search_v2():
    """Test title and author search (v2)"""
    print("\nTesting WorldCat v2 title/author search...")
    api = WorldCatAPIv2()
    test_books = [
        {
            "title": "To Kill a Mockingbird",
            "authors": ["Harper Lee"]
        },
        {
            "title": "Pride and Prejudice",
            "authors": ["Jane Austen"]
        }
    ]
    for book in test_books:
        print(f"\nSearching for: {book['title']} by {', '.join(book['authors'])}")
        try:
            metadata = api.search_by_title_author(book['title'], book['authors'])
            if metadata:
                print(f"✓ Found metadata")
                print(f"  - Title: {metadata.get('title', 'N/A')}")
                print(f"  - Author: {metadata.get('author', 'N/A')}")
                print(f"  - Publisher: {metadata.get('publisher', 'N/A')}")
                print(f"  - OCLC Number: {metadata.get('oclc_no', 'N/A')}")
                print(f"  - ISBN: {metadata.get('isbn', 'N/A')}")
            else:
                print(f"✗ No metadata found")
        except Exception as e:
            print(f"✗ Error searching: {e}")
        time.sleep(2)

def test_multiple_isbn_search_v2():
    """Test searching multiple ISBNs at once (v2)"""
    print("\nTesting WorldCat v2 multiple ISBN search...")
    api = WorldCatAPIv2()
    test_isbns = [
        "9780446310789",  # To Kill a Mockingbird
        "9780141439518",  # Pride and Prejudice
        "9780743273565"   # The Great Gatsby
    ]
    try:
        results = api.search_multiple_isbns(test_isbns)
        print(f"✓ Searched {len(test_isbns)} ISBNs")
        for isbn, metadata in results.items():
            if metadata:
                print(f"  - {isbn}: Found ({metadata.get('title', 'N/A')})")
            else:
                print(f"  - {isbn}: Not found")
    except Exception as e:
        print(f"✗ Error in multiple ISBN search: {e}")

def run_all_tests_v2():
    """Run all WorldCat API v2 tests"""
    print("=" * 60)
    print("WORLDCAT API v2 TEST SUITE")
    print("=" * 60)
    # Check if credentials are configured
    config = Config()
    if not hasattr(config, 'WORLDCAT_CLIENT_ID') or not config.WORLDCAT_CLIENT_ID:
        print("⚠️  Warning: WORLDCAT_CLIENT_ID not configured in environment variables")
        print("   Set WORLDCAT_CLIENT_ID in your .env file to run full tests")
    if not hasattr(config, 'WORLDCAT_CLIENT_SECRET') or not config.WORLDCAT_CLIENT_SECRET:
        print("⚠️  Warning: WORLDCAT_CLIENT_SECRET not configured in environment variables")
        print("   Set WORLDCAT_CLIENT_SECRET in your .env file to run full tests")
    if (not hasattr(config, 'WORLDCAT_CLIENT_ID') or not config.WORLDCAT_CLIENT_ID or
        not hasattr(config, 'WORLDCAT_CLIENT_SECRET') or not config.WORLDCAT_CLIENT_SECRET):
        print("   Some tests may fail without valid credentials")
    tests = [
        test_worldcat_api_v2_initialization,
        test_isbn_search_v2,
        test_title_author_search_v2,
        test_multiple_isbn_search_v2
    ]
    passed = 0
    total = len(tests)
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with exception: {e}")
    print("\n" + "=" * 60)
    print(f"TEST RESULTS: {passed}/{total} tests passed")
    print("=" * 60)
    return passed == total

if __name__ == "__main__":
    success = run_all_tests_v2()
    sys.exit(0 if success else 1) 