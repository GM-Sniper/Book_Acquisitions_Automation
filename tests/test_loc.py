"""
Library of Congress API Test Suite
=================================

Purpose:
--------
This test suite evaluates the Library of Congress (LoC) API integration for book metadata retrieval,
specifically focusing on the ISBN to LCCN (Library of Congress Control Number) conversion workflow.

Key Functionality Tested:
------------------------
1. ISBN to LCCN Conversion
   - Tests the SRU (Search/Retrieve via URL) service
   - Validates ISBN cleaning and formatting
   - Checks LCCN extraction from XML responses

2. Title/Author to LCCN Conversion
   - Tests title-based searches
   - Tests title + author combined searches
   - Validates query formatting and escaping

Test Data:
---------
Uses well-known books for testing:
- To Kill a Mockingbird (ISBN: 9780446310789, Author: Harper Lee)
- Pride and Prejudice (ISBN: 9780141439518, Author: Jane Austen)
- The Great Gatsby (Author: F. Scott Fitzgerald)

Note:
-----
This test suite was primarily created to evaluate the reliability and functionality
of the Library of Congress API for automated book metadata retrieval. It helps
identify potential issues with:
- API accessibility
- Response formats
- Data extraction
- Error handling
- Rate limiting
"""

import sys
import os
import time
import requests
import xml.etree.ElementTree as ET
import re
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from config.config import Config

class LOCConverter:
    def __init__(self, debug: bool = False):
        self.config = Config()
        self.debug = debug
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'LibraryAutomation/1.0 (Educational Project)'
        })
    
    def isbn_to_lccn(self, isbn):
        """Convert ISBN to LCCN using LOC SRU service"""
        try:
            # Clean ISBN (remove hyphens, spaces)
            clean_isbn = re.sub(r'[-\s]', '', isbn)
            
            # LOC SRU endpoint for ISBN search
            url = "http://lx2.loc.gov:210/lcdb"
            params = {
                'version': '1.1',
                'operation': 'searchRetrieve',
                'query': f'bath.isbn={clean_isbn}',
                'maximumRecords': '1',
                'recordSchema': 'mods'
            }
            
            response = self.session.get(url, params=params, timeout=15)
            
            if response.status_code == 200:
                lccn = self._extract_lccn(response.text)
                if lccn:
                    if self.debug:
                        print(f"✅ Found LCCN: {lccn}")
                    return lccn
                else:
                    if self.debug:
                        print("❌ No LCCN found in response")
                    return None
            else:
                if self.debug:
                    print(f"❌ SRU request failed: {response.status_code}")
                return None
                
        except Exception as e:
            if self.debug:
                print(f"❌ Error converting ISBN to LCCN: {e}")
            return None
    
    def title_author_to_lccn(self, title, author=None):
        """Convert Title (and optional Author) to LCCN using LOC SRU service"""
        if self.debug:
            print(f"🔍 Searching LCCN for title: '{title}'" + (f", author: '{author}'" if author else ""))
        
        try:
            url = "http://lx2.loc.gov:210/lcdb"
            
            # Fix query syntax - use proper CQL format
            if author:
                # Escape quotes and use proper CQL syntax
                clean_title = title.replace('"', '\\"')
                clean_author = author.replace('"', '\\"')
                # Use more specific search to avoid movies
                query = f'bath.title="{clean_title}" AND bath.name="{clean_author}"'
            else:
                clean_title = title.replace('"', '\\"')
                query = f'bath.title="{clean_title}"'
            
            params = {
                'version': '1.1',
                'operation': 'searchRetrieve',
                'query': query,
                'maximumRecords': '10',  # Get more results to find the book
                'recordSchema': 'mods'
            }
            
            if self.debug:
                print(f"🌐 Requesting: {url} with params: {params}")
            
            response = self.session.get(url, params=params, timeout=15)
            
            if self.debug:
                print(f"📡 SRU Status: {response.status_code}")
                print(f"📄 Response length: {len(response.text)} characters")
            
            if response.status_code == 200:
                lccn = self._extract_lccn_from_multiple_records(response.text)
                if lccn:
                    if self.debug:
                        print(f"✅ Found LCCN: {lccn}")
                    return lccn
                else:
                    if self.debug:
                        print("❌ No LCCN found in response")
                    return None
            else:
                if self.debug:
                    print(f"❌ SRU request failed: {response.status_code}")
                return None
                
        except Exception as e:
            if self.debug:
                print(f"❌ Error converting title '{title}' to LCCN: {e}")
            return None
    
    def _extract_lccn_from_multiple_records(self, xml_text):
        """Extract LCCN from multiple records, preferring books over movies"""
        # Split into individual records
        record_pattern = r'<zs:record>.*?</zs:record>'
        records = re.findall(record_pattern, xml_text, re.DOTALL)
        
        if self.debug:
            print(f"🔍 Found {len(records)} records to check")
        
        for i, record in enumerate(records):
            if self.debug:
                print(f"📖 Checking record {i+1}/{len(records)}")
            
            # Skip movies and videos
            if '<typeOfResource>moving image</typeOfResource>' in record:
                if self.debug:
                    print(f"   ❌ Record {i+1}: Skipping movie/video")
                continue
            
            if '<typeOfResource>sound recording</typeOfResource>' in record:
                if self.debug:
                    print(f"   ❌ Record {i+1}: Skipping audio recording")
                continue
            
            # Look for LCCN in this record
            lccn_match = re.search(r'<mods:identifier[^>]*type="lccn"[^>]*>([^<]+)</mods:identifier>',
                                   record, re.IGNORECASE)
            if lccn_match:
                lccn = lccn_match.group(1).strip()
                if self.debug:
                    print(f"   ✅ Record {i+1}: Found LCCN {lccn} (book)")
                return lccn
            
            # Try alternative patterns
            alt_patterns = [
                r'<identifier[^>]*type="lccn"[^>]*>([^<]+)</identifier>',
                r'<lccn>([^<]+)</lccn>',
                r'LCCN:\s*([^\s<]+)'
            ]
            
            for pattern in alt_patterns:
                match = re.search(pattern, record, re.IGNORECASE)
                if match:
                    lccn = match.group(1).strip()
                    if self.debug:
                        print(f"   ✅ Record {i+1}: Found LCCN {lccn} (alt pattern)")
                    return lccn
        
        if self.debug:
            print("❌ No book LCCN found in any record")
        return None
    
    def _extract_lccn(self, xml_text):
        """Extract LCCN from MODS XML response (single record)"""
        # Check if this is a book (not a movie/video)
        if '<typeOfResource>moving image</typeOfResource>' in xml_text:
            if self.debug:
                print("❌ Skipping movie/video result")
            return None
        
        if '<typeOfResource>sound recording</typeOfResource>' in xml_text:
            if self.debug:
                print("❌ Skipping audio recording result")
            return None
        
        # Look for LCCN in the response
        lccn_match = re.search(r'<mods:identifier[^>]*type="lccn"[^>]*>([^<]+)</mods:identifier>',
                               xml_text, re.IGNORECASE)
        if lccn_match:
            lccn = lccn_match.group(1).strip()
            if self.debug:
                print(f"✅ Found LCCN: {lccn}")
            return lccn
        
        # Try alternative patterns
        alt_patterns = [
            r'<identifier[^>]*type="lccn"[^>]*>([^<]+)</identifier>',
            r'<lccn>([^<]+)</lccn>',
            r'LCCN:\s*([^\s<]+)'
        ]
        
        for pattern in alt_patterns:
            match = re.search(pattern, xml_text, re.IGNORECASE)
            if match:
                lccn = match.group(1).strip()
                if self.debug:
                    print(f"✅ Found LCCN (alt pattern): {lccn}")
                return lccn
        
        if self.debug:
            print("❌ No LCCN found in response")
        return None

def test_isbn_to_lccn_workflow():
    """Test the ISBN to LCCN conversion workflow"""
    print("🔍 Testing ISBN to LCCN Conversion Workflow")
    print("=" * 60)
    
    converter = LOCConverter(debug=False)  # No debug for ISBN workflow
    
    # Test ISBNs with correct numbers
    test_isbns = [
        "9780446310789",  # To Kill a Mockingbird
        "9780141439518",  # Pride and Prejudice
        "9789992195239"   # مولانا (Mowlana)
    ]
    
    successful_conversions = 0
    
    for isbn in test_isbns:
        print(f"\n📖 --- Testing ISBN: {isbn} ---")
        
        # Convert ISBN to LCCN
        lccn = converter.isbn_to_lccn(isbn)
        
        if lccn:
            successful_conversions += 1
            print(f"✅ Successfully converted ISBN {isbn} to LCCN {lccn}")
        else:
            print(f"❌ Failed to convert ISBN {isbn} to LCCN")
        
        time.sleep(2)  # Rate limiting
    
    print(f"\n" + "=" * 60)
    print(f"📊 WORKFLOW TEST SUMMARY")
    print(f"Successful ISBN → LCCN conversions: {successful_conversions}/{len(test_isbns)}")
    
    if successful_conversions > 0:
        print(f"\n✅ SUCCESS: ISBN to LCCN conversion is working!")
    else:
        print(f"\n❌ Workflow needs refinement")

def test_title_author_to_lccn_workflow():
    """Test the Title/Author to LCCN conversion workflow"""
    print("\n🔍 Testing Title/Author to LCCN Conversion Workflow")
    print("=" * 60)
    
    converter = LOCConverter(debug=True)
    
    # Test title/author combinations
    test_cases = [
        ("To Kill a Mockingbird", "Harper Lee"),
        ("Pride and Prejudice", "Jane Austen"),
        ("The Great Gatsby", "F. Scott Fitzgerald"),
        ("1984", "George Orwell"),
        ("The Catcher in the Rye", "J.D. Salinger"),
        ("مولانا", None),  # Mowlana in Arabic (no specific author)
        ("Mowlana", None),  # Mowlana in English
        ("Rumi", "Jalal al-Din"),  # Rumi's English name
        ("Masnavi", "Rumi"),  # Rumi's most famous work
        ("Jalal al-Din Rumi", None),  # Full name
        ("The Rubaiyat", "Omar Khayyam"),  # Persian poetry (English translation)
        ("Shahnameh", "Ferdowsi"),  # Persian epic (English translation)
        ("One Thousand and One Nights", None)  # Arabic literature (English)
    ]
    
    successful_conversions = 0
    
    for title, author in test_cases:
        print(f"\n📚 --- Testing: '{title}' by {author} ---")
        
        # Test title + author search
        lccn = converter.title_author_to_lccn(title, author)
        
        if lccn:
            successful_conversions += 1
            print(f"✅ Successfully found LCCN {lccn} for '{title}' by {author}")
        else:
            print(f"❌ Failed to find LCCN for '{title}' by {author}")
            
            # Try title-only search as fallback
            print(f"🔄 Trying title-only search...")
            lccn_title_only = converter.title_author_to_lccn(title)
            if lccn_title_only:
                print(f"✅ Found LCCN {lccn_title_only} for title '{title}' (no author)")
                successful_conversions += 1
            else:
                print(f"❌ No LCCN found for title '{title}' either")
        
        time.sleep(2)  # Rate limiting
    
    print(f"\n" + "=" * 60)
    print(f"📊 TITLE/AUTHOR TEST SUMMARY")
    print(f"Successful Title/Author → LCCN conversions: {successful_conversions}/{len(test_cases)}")
    
    if successful_conversions > 0:
        print(f"\n✅ SUCCESS: Title/Author to LCCN conversion is working!")
    else:
        print(f"\n❌ Workflow needs refinement")

def test_comparison_workflow():
    """Compare ISBN vs Title/Author search results"""
    print("\n🔍 Testing ISBN vs Title/Author Comparison")
    print("=" * 60)
    
    converter = LOCConverter(debug=False)  # Less verbose for comparison
    
    # Test cases with both ISBN and title/author
    test_cases = [
        {
            "isbn": "9780446310789",
            "title": "To Kill a Mockingbird",
            "author": "Harper Lee"
        },
        {
            "isbn": "9780141439518", 
            "title": "Pride and Prejudice",
            "author": "Jane Austen"
        }
    ]
    
    print("📚 Testing US/English books (should work):")
    for case in test_cases:
        print(f"\n📖 Testing: {case['title']} by {case['author']}")
        print(f"   📖 ISBN: {case['isbn']}")
        
        # ISBN search
        lccn_isbn = converter.isbn_to_lccn(case['isbn'])
        print(f"   🔍 ISBN search result: {lccn_isbn}")
        
        time.sleep(1)
        
        # Title/Author search
        lccn_title = converter.title_author_to_lccn(case['title'], case['author'])
        print(f"   🔍 Title/Author search result: {lccn_title}")
        
        # Compare results
        if lccn_isbn and lccn_title:
            if lccn_isbn == lccn_title:
                print(f"   ✅ Both methods found same LCCN: {lccn_isbn}")
            else:
                print(f"   ⚠️  Different LCCNs found: ISBN={lccn_isbn}, Title={lccn_title}")
        elif lccn_isbn:
            print(f"   ✅ Only ISBN search found LCCN: {lccn_isbn}")
        elif lccn_title:
            print(f"   ✅ Only Title/Author search found LCCN: {lccn_title}")
        else:
            print(f"   ❌ No LCCN found by either method")
        
        time.sleep(2)
    
    print("\n📚 Testing International/Arabic books (should work!):")
    international_cases = [
        {
            "isbn": "9789992195239",
            "title": "مولانا",
            "author": None
        }
    ]
    
    for case in international_cases:
        print(f"\n📖 Testing: {case['title']} by {case['author']}")
        print(f"   📖 ISBN: {case['isbn']}")
        
        # ISBN search
        lccn_isbn = converter.isbn_to_lccn(case['isbn'])
        print(f"   🔍 ISBN search result: {lccn_isbn}")
        
        time.sleep(1)
        
        # Title/Author search
        lccn_title = converter.title_author_to_lccn(case['title'], case['author'])
        print(f"   🔍 Title/Author search result: {lccn_title}")
        
        if not lccn_isbn and not lccn_title:
            print(f"   ❌ LOC should have this book (LCCN: 2012342951)")
        else:
            print(f"   ✅ Great! LOC has this international book!")
        
        time.sleep(2)
    
    for case in test_cases:
        print(f"\n📖 Testing: {case['title']} by {case['author']}")
        print(f"   📖 ISBN: {case['isbn']}")
        
        # ISBN search
        lccn_isbn = converter.isbn_to_lccn(case['isbn'])
        print(f"   🔍 ISBN search result: {lccn_isbn}")
        
        time.sleep(1)
        
        # Title/Author search
        lccn_title = converter.title_author_to_lccn(case['title'], case['author'])
        print(f"   🔍 Title/Author search result: {lccn_title}")
        
        # Compare results
        if lccn_isbn and lccn_title:
            if lccn_isbn == lccn_title:
                print(f"   ✅ Both methods found same LCCN: {lccn_isbn}")
            else:
                print(f"   ⚠️  Different LCCNs found: ISBN={lccn_isbn}, Title={lccn_title}")
        elif lccn_isbn:
            print(f"   ✅ Only ISBN search found LCCN: {lccn_isbn}")
        elif lccn_title:
            print(f"   ✅ Only Title/Author search found LCCN: {lccn_title}")
        else:
            print(f"   ❌ No LCCN found by either method")
        
        time.sleep(2)

if __name__ == "__main__":
    test_isbn_to_lccn_workflow()
    test_title_author_to_lccn_workflow()
    test_comparison_workflow()
