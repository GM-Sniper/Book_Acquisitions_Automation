import requests
import re
import time
from typing import Optional, List, Dict

class LOCConverter:
    """Library of Congress ISBN to LCCN Converter"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'LibraryAutomation/1.0 (Educational Project)'
        })
    
    def isbn_to_lccn(self, isbn: str) -> Optional[str]:
        """
        Convert ISBN to LCCN using LOC SRU service
        
        Args:
            isbn (str): The ISBN to convert
            
        Returns:
            Optional[str]: The LCCN if found, None otherwise
        """
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
                # Look for LCCN in the response
                lccn_match = re.search(r'<mods:identifier[^>]*type="lccn"[^>]*>([^<]+)</mods:identifier>', 
                                     response.text, re.IGNORECASE)
                
                if lccn_match:
                    lccn = lccn_match.group(1).strip()
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
                            return lccn
                    
                    return None
            else:
                return None
                
        except Exception as e:
            print(f"Error converting ISBN {isbn} to LCCN: {e}")
            return None
    
    def get_lccn_for_isbns(self, isbns: List[str]) -> Dict[str, Optional[str]]:
        """
        Get LCCN for multiple ISBNs
        
        Args:
            isbns (List[str]): List of ISBNs to convert
            
        Returns:
            Dict[str, Optional[str]]: Dictionary mapping ISBNs to their LCCNs
        """
        results = {}
        
        for isbn in isbns:
            lccn = self.isbn_to_lccn(isbn)
            results[isbn] = lccn
            
            # Rate limiting to be respectful to the API
            time.sleep(1)
        
        return results

# Convenience function for single ISBN conversion
def convert_isbn_to_lccn(isbn: str) -> Optional[str]:
    """
    Convenience function to convert a single ISBN to LCCN
    
    Args:
        isbn (str): The ISBN to convert
        
    Returns:
        Optional[str]: The LCCN if found, None otherwise
    """
    converter = LOCConverter()
    return converter.isbn_to_lccn(isbn)

# Test function
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python LOC.py <ISBN>")
        print("Example: python LOC.py 9780446310789")
        sys.exit(1)
    
    isbn = sys.argv[1]
    converter = LOCConverter()
    
    print(f"Converting ISBN {isbn} to LCCN...")
    lccn = converter.isbn_to_lccn(isbn)
    
    if lccn:
        print(f"✓ Found LCCN: {lccn}")
    else:
        print("✗ No LCCN found for this ISBN") 