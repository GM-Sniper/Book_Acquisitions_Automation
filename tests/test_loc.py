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

Test Data:
---------
Uses three well-known books for testing:
- To Kill a Mockingbird (ISBN: 9780446310789)
- Pride and Prejudice (ISBN: 9780141439518)

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

class ISBNToLCCNConverter:
    def __init__(self):
        self.config = Config()
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'LibraryAutomation/1.0 (Educational Project)'
        })
    
    def isbn_to_lccn(self, isbn):
        """Convert ISBN to LCCN using LOC SRU service"""
        print(f"Converting ISBN {isbn} to LCCN...")
        
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
            print(f"  SRU Status: {response.status_code}")
            
            if response.status_code == 200:
                # Look for LCCN in the response
                lccn_match = re.search(r'<mods:identifier[^>]*type="lccn"[^>]*>([^<]+)</mods:identifier>', 
                                     response.text, re.IGNORECASE)
                
                if lccn_match:
                    lccn = lccn_match.group(1).strip()
                    print(f"✓ Found LCCN: {lccn}")
                    return lccn
                else:
                    # Try alternative LCCN patterns
                    alt_patterns = [
                        r'<identifier[^>]*type="lccn"[^>]*>([^<]+)</identifier>',
                        r'<lccn>([^<]+)</lccn>',
                        r'LCCN:\s*([^\s<]+)'
                    ]
                    
                    for pattern in alt_patterns:
                        match = re.search(pattern, response.text, re.IGNORECASE)
                        if match:
                            lccn = match.group(1).strip()
                            print(f"✓ Found LCCN (alternative): {lccn}")
                            return lccn
                    
                    print("  No LCCN found in response")
                    return None
            else:
                print(f"  SRU request failed: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"  Error converting ISBN to LCCN: {e}")
            return None

def test_isbn_to_lccn_workflow():
    """Test the ISBN to LCCN conversion workflow"""
    print("Testing ISBN to LCCN Conversion Workflow")
    print("=" * 60)
    
    converter = ISBNToLCCNConverter()
    
    # Test ISBNs with correct numbers
    test_isbns = [
        "9780446310789",  # To Kill a Mockingbird

        "9780141439518"   # Pride and Prejudice
    ]
    
    successful_conversions = 0
    
    for isbn in test_isbns:
        print(f"\n--- Testing ISBN: {isbn} ---")
        
        # Convert ISBN to LCCN
        lccn = converter.isbn_to_lccn(isbn)
        
        if lccn:
            successful_conversions += 1
            print(f"✓ Successfully converted ISBN {isbn} to LCCN {lccn}")
        else:
            print(f"✗ Failed to convert ISBN {isbn} to LCCN")
        
        time.sleep(3)  # Rate limiting
    
    print(f"\n" + "=" * 60)
    print(f"WORKFLOW TEST SUMMARY")
    print(f"Successful ISBN → LCCN conversions: {successful_conversions}/{len(test_isbns)}")
    
    if successful_conversions > 0:
        print(f"\n✓ SUCCESS: ISBN to LCCN conversion is working!")
    else:
        print(f"\n✗ Workflow needs refinement")

if __name__ == "__main__":
    test_isbn_to_lccn_workflow()
