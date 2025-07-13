"""
WorldCat API utility functions for book metadata retrieval
"""
import requests
import json
import time
import base64
from typing import Optional, List, Dict
from config.config import Config

class WorldCatAPI:
    """WorldCat API client for book metadata retrieval"""
    
    def __init__(self):
        self.config = Config()
        self.base_url = "https://www.worldcat.org/webservices/catalog"
        self.headers = {
            'User-Agent': 'BookAcquisitionsAutomation/1.0',
            'Accept': 'application/json'
        }
        # WorldCat credentials should be set in environment variables
        self.client_id = self.config.WORLDCAT_CLIENT_ID if hasattr(self.config, 'WORLDCAT_CLIENT_ID') else None
        self.client_secret = self.config.WORLDCAT_CLIENT_SECRET if hasattr(self.config, 'WORLDCAT_CLIENT_SECRET') else None
        
    def search_by_isbn(self, isbn: str) -> Optional[Dict]:
        """
        Search for a book by ISBN using WorldCat API
        Args:
            isbn (str): ISBN to search for
        Returns:
            dict: Book metadata or None if not found
        """
        try:
            # Clean ISBN (remove hyphens, spaces)
            clean_isbn = isbn.replace('-', '').replace(' ', '')
            
            # WorldCat SRU endpoint for ISBN search
            url = f"{self.base_url}/search/sru"
            params = {
                'query': f'srw.isbn="{clean_isbn}"',
                'version': '1.1',
                'operation': 'searchRetrieve',
                'maximumRecords': '1',
                'recordSchema': 'info:srw/schema/1/dc-v1.1',
                'wskey': self.client_id
            }
            
            # Add authentication headers
            auth_headers = self.headers.copy()
            if self.client_secret:
                # Use Basic Auth with Client ID and Secret
                credentials = f"{self.client_id}:{self.client_secret}"
                encoded_credentials = base64.b64encode(credentials.encode()).decode()
                auth_headers['Authorization'] = f'Basic {encoded_credentials}'
            
            response = requests.get(url, params=params, headers=auth_headers, timeout=15)
            
            if response.status_code == 200:
                return self._parse_sru_response(response.text, isbn)
            else:
                print(f"WorldCat API error: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"WorldCat ISBN search failed: {e}")
            return None
    
    def search_by_title_author(self, title: str, authors: List[str] = None) -> Optional[Dict]:
        """
        Search for a book by title and authors using WorldCat API
        Args:
            title (str): Book title
            authors (list): List of author names
        Returns:
            dict: Book metadata or None if not found
        """
        try:
            # Build search query
            query_parts = [f'srw.ti="{title}"']
            if authors:
                for author in authors:
                    if author and author.strip():
                        query_parts.append(f'srw.au="{author}"')
            
            query = ' AND '.join(query_parts)
            
            url = f"{self.base_url}/search/sru"
            params = {
                'query': query,
                'version': '1.1',
                'operation': 'searchRetrieve',
                'maximumRecords': '5',
                'recordSchema': 'info:srw/schema/1/dc-v1.1',
                'wskey': self.client_id
            }
            
            # Add authentication headers
            auth_headers = self.headers.copy()
            if self.client_secret:
                # Use Basic Auth with Client ID and Secret
                credentials = f"{self.client_id}:{self.client_secret}"
                encoded_credentials = base64.b64encode(credentials.encode()).decode()
                auth_headers['Authorization'] = f'Basic {encoded_credentials}'
            
            response = requests.get(url, params=params, headers=auth_headers, timeout=15)
            
            if response.status_code == 200:
                return self._parse_sru_response(response.text, title)
            else:
                print(f"WorldCat API error: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"WorldCat title/author search failed: {e}")
            return None
    
    def get_oclc_number(self, isbn: str) -> Optional[str]:
        """
        Get OCLC number for a given ISBN
        Args:
            isbn (str): ISBN to search for
        Returns:
            str: OCLC number or None if not found
        """
        try:
            metadata = self.search_by_isbn(isbn)
            if metadata and metadata.get('oclc_no'):
                return metadata['oclc_no']
            return None
        except Exception as e:
            print(f"Error getting OCLC number for ISBN {isbn}: {e}")
            return None
    
    def _parse_sru_response(self, response_text: str, search_term: str) -> Optional[Dict]:
        """
        Parse SRU response from WorldCat API
        Args:
            response_text (str): Raw response text
            search_term (str): Original search term for logging
        Returns:
            dict: Parsed metadata or None
        """
        try:
            # Parse XML response (simplified parsing)
            # Look for key metadata fields in the response
            
            # Extract title
            title_match = self._extract_xml_field(response_text, 'dc:title')
            title = title_match if title_match else ''
            
            # Extract authors
            author_matches = self._extract_xml_fields(response_text, 'dc:creator')
            authors = author_matches if author_matches else []
            
            # Extract publisher
            publisher_match = self._extract_xml_field(response_text, 'dc:publisher')
            publisher = publisher_match if publisher_match else ''
            
            # Extract date
            date_match = self._extract_xml_field(response_text, 'dc:date')
            published_date = date_match if date_match else ''
            
            # Extract OCLC number
            oclc_match = self._extract_xml_field(response_text, 'oclcterms:OCLCnumber')
            oclc_no = oclc_match if oclc_match else ''
            
            # Extract ISBNs
            isbn_matches = self._extract_xml_fields(response_text, 'dc:identifier')
            isbns = []
            isbn_10 = ''
            isbn_13 = ''
            
            for isbn in isbn_matches:
                if 'ISBN' in isbn:
                    clean_isbn = isbn.replace('ISBN ', '').replace('ISBN:', '').strip()
                    if len(clean_isbn) == 10:
                        isbn_10 = clean_isbn
                        isbns.append(clean_isbn)
                    elif len(clean_isbn) == 13:
                        isbn_13 = clean_isbn
                        isbns.append(clean_isbn)
            
            metadata = {
                'title': title,
                'author': ', '.join(authors),
                'publisher': publisher,
                'published_date': published_date,
                'd_o_pub': published_date,
                'oclc_no': oclc_no,
                'lc_no': '',
                'isbn_10': isbn_10,
                'isbn_13': isbn_13,
                'isbn': '; '.join(isbns) if isbns else '',
                'source': 'worldcat'
            }
            
            return metadata if title else None
            
        except Exception as e:
            print(f"Error parsing WorldCat response: {e}")
            return None
    
    def _extract_xml_field(self, xml_text: str, field_name: str) -> Optional[str]:
        """Extract a single field from XML response"""
        try:
            import re
            pattern = f'<{field_name}[^>]*>(.*?)</{field_name}>'
            match = re.search(pattern, xml_text, re.IGNORECASE | re.DOTALL)
            return match.group(1).strip() if match else None
        except Exception:
            return None
    
    def _extract_xml_fields(self, xml_text: str, field_name: str) -> List[str]:
        """Extract multiple fields from XML response"""
        try:
            import re
            pattern = f'<{field_name}[^>]*>(.*?)</{field_name}>'
            matches = re.findall(pattern, xml_text, re.IGNORECASE | re.DOTALL)
            return [match.strip() for match in matches if match.strip()]
        except Exception:
            return []
    
    def search_multiple_isbns(self, isbns: List[str]) -> Dict[str, Optional[Dict]]:
        """
        Search for multiple ISBNs and return results
        Args:
            isbns (List[str]): List of ISBNs to search
        Returns:
            Dict[str, Optional[Dict]]: Dictionary mapping ISBNs to their metadata
        """
        results = {}
        
        for isbn in isbns:
            metadata = self.search_by_isbn(isbn)
            results[isbn] = metadata
            
            # Rate limiting
            time.sleep(1)
        
        return results 