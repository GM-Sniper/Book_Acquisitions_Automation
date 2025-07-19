import requests
import re
import time
from typing import Optional, List, Dict


class LOCConverter:
    """Library of Congress ISBN/Title/Author to LCCN Converter"""

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
            clean_isbn = re.sub(r'[-\s]', '', isbn)

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
                return self._extract_lccn(response.text)
            else:
                return None

        except Exception as e:
            print(f"Error converting ISBN {isbn} to LCCN: {e}")
            return None

    def title_author_to_lccn(self, title: str, author: Optional[str] = None) -> Optional[str]:
        """
        Convert Title (and optional Author) to LCCN using LOC SRU service

        Args:
            title (str): The book title to search
            author (Optional[str]): The author (optional)

        Returns:
            Optional[str]: The LCCN if found, None otherwise
        """
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
                'maximumRecords': '1',
                'recordSchema': 'mods'
            }

            response = self.session.get(url, params=params, timeout=15)

            if response.status_code == 200:
                return self._extract_lccn(response.text)
            else:
                return None

        except Exception as e:
            print(f"Error converting title '{title}' to LCCN: {e}")
            return None

    def _extract_lccn(self, xml_text: str) -> Optional[str]:
        """
        Extract LCCN from MODS XML response

        Args:
            xml_text (str): Raw XML string

        Returns:
            Optional[str]: Extracted LCCN or None
        """
        lccn_match = re.search(r'<mods:identifier[^>]*type="lccn"[^>]*>([^<]+)</mods:identifier>',
                               xml_text, re.IGNORECASE)
        if lccn_match:
            return lccn_match.group(1).strip()

        # Try alternative patterns
        alt_patterns = [
            r'<identifier[^>]*type="lccn"[^>]*>([^<]+)</identifier>',
            r'<lccn>([^<]+)</lccn>',
            r'LCCN:\s*([^\s<]+)'
        ]

        for pattern in alt_patterns:
            match = re.search(pattern, xml_text, re.IGNORECASE)
            if match:
                return match.group(1).strip()

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
            time.sleep(1)  # Respect API

        return results


# Test function
if __name__ == "__main__":
    import sys

    converter = LOCConverter()

    if len(sys.argv) == 2:
        isbn = sys.argv[1]
        print(f"Converting ISBN {isbn} to LCCN...")
        lccn = converter.isbn_to_lccn(isbn)
        print(f"✓ Found LCCN: {lccn}" if lccn else "✗ No LCCN found for this ISBN")

    elif len(sys.argv) >= 3:
        title = sys.argv[1]
        author = sys.argv[2] if len(sys.argv) >= 3 else None
        print(f"Searching LCCN for title: '{title}'" + (f", author: '{author}'" if author else ""))
        lccn = converter.title_author_to_lccn(title, author)
        print(f"✓ Found LCCN: {lccn}" if lccn else "✗ No LCCN found for this title/author")
    else:
        print("Usage:")
        print("  ISBN:  python LOC.py 9780446310789")
        print('  Title: python LOC.py "To Kill a Mockingbird" "Harper Lee"')
