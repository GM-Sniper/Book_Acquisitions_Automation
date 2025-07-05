"""
OpenLibrary API utility functions
"""
import requests
import json
from typing import Optional, List, Dict

class OpenLibraryAPI:
    """OpenLibrary API client for book metadata retrieval"""
    
    def __init__(self):
        self.base_url = "https://openlibrary.org"
        self.headers = {
            'User-Agent': 'BookAcquisitionsAutomation/1.0'
        }
    
    def search_by_isbn(self, isbn: str) -> Optional[Dict]:
        """
        Search for a book by ISBN using OpenLibrary API
        Args:
            isbn (str): ISBN to search for
        Returns:
            dict: Book metadata or None if not found
        """
        try:
            # Use the Partner API for ISBN search
            url = f"{self.base_url}/api/books"
            params = {
                'bibkeys': f'ISBN:{isbn}',
                'format': 'json',
                'jscmd': 'data'
            }
            
            response = requests.get(url, params=params, headers=self.headers)
            response.raise_for_status() #raise an exception if the request is not successful
            
            data = response.json()
            isbn_key = f'ISBN:{isbn}'
            
            if isbn_key in data and data[isbn_key]:
                return self._extract_metadata(data[isbn_key])
            
            return None
            
        except Exception as e:
            print(f"OpenLibrary ISBN search failed: {e}")
            return None
    
    def search_by_title_author(self, title: str, authors: List[str] = None) -> Optional[Dict]:
        """
        Search for a book by title and authors using OpenLibrary Search API
        Args:
            title (str): Book title
            authors (list): List of author names
        Returns:
            dict: Book metadata or None if not found
        """
        try:
            # First try exact title match
            exact_result = self._search_exact_title(title, authors)
            if exact_result:
                return exact_result
            
            # If no exact match, try partial title match
            partial_result = self._search_partial_title(title, authors)
            if partial_result:
                return partial_result
            
            return None
            
        except Exception as e:
            print(f"OpenLibrary title/author search failed: {e}")
            return None
    
    def _search_exact_title(self, title: str, authors: List[str] = None) -> Optional[Dict]:
        """Search for exact title match"""
        try:
            # Build search query for exact match
            query_parts = [f'title:"{title}"']
            if authors:
                for author in authors:
                    if author and author.strip():
                        query_parts.append(f'author:"{author}"')
            
            query = ' '.join(query_parts)
            
            url = f"{self.base_url}/search.json"
            params = {
                'q': query,
                'limit': 10
            }
            
            response = requests.get(url, params=params, headers=self.headers)
            response.raise_for_status()
            
            data = response.json()
            
            if data.get('docs') and len(data['docs']) > 0:
                # Look for exact title match
                for book in data['docs']:
                    if book.get('title', '').lower() == title.lower():
                        return self._extract_metadata_from_search(book)
                
                # If no exact match, return first result
                return self._extract_metadata_from_search(data['docs'][0])
            
            return None
            
        except Exception as e:
            print(f"OpenLibrary exact title search failed: {e}")
            return None
    
    def _search_partial_title(self, title: str, authors: List[str] = None) -> Optional[Dict]:
        """Search for partial title match"""
        try:
            # Build search query for partial match
            query_parts = [title]  # No quotes for partial matching
            if authors:
                for author in authors:
                    if author and author.strip():
                        query_parts.append(f'author:"{author}"')
            
            query = ' '.join(query_parts)
            
            url = f"{self.base_url}/search.json"
            params = {
                'q': query,
                'limit': 10
            }
            
            response = requests.get(url, params=params, headers=self.headers)
            response.raise_for_status()
            
            data = response.json()
            
            if data.get('docs') and len(data['docs']) > 0:
                # Return first result (most relevant)
                return self._extract_metadata_from_search(data['docs'][0])
            
            return None
            
        except Exception as e:
            print(f"OpenLibrary partial title search failed: {e}")
            return None
    
    def search_arabic_book(self, title: str, authors: List[str] = None) -> Optional[Dict]:
        """
        Search for Arabic books using OpenLibrary API
        Args:
            title (str): Arabic book title
            authors (list): List of Arabic author names
        Returns:
            dict: Book metadata or None if not found
        """
        try:
            # First try exact title match
            exact_result = self._search_arabic_exact(title, authors)
            if exact_result:
                return exact_result
            
            # If no exact match, try partial title match
            partial_result = self._search_arabic_partial(title, authors)
            if partial_result:
                return partial_result
            
            return None
            
        except Exception as e:
            print(f"OpenLibrary Arabic search failed: {e}")
            return None
    
    def _search_arabic_exact(self, title: str, authors: List[str] = None) -> Optional[Dict]:
        """Search for exact Arabic title match"""
        try:
            # Build search query for exact Arabic match
            query_parts = [f'title:"{title}"']
            if authors:
                for author in authors:
                    if author and author.strip():
                        query_parts.append(f'author:"{author}"')
            
            query = ' '.join(query_parts)
            
            url = f"{self.base_url}/search.json"
            params = {
                'q': query,
                'limit': 10,
                'language': 'ara'  # Arabic language filter
            }
            
            response = requests.get(url, params=params, headers=self.headers)
            response.raise_for_status()
            
            data = response.json()
            
            if data.get('docs') and len(data['docs']) > 0:
                # Look for exact title match
                for book in data['docs']:
                    if book.get('title', '').lower() == title.lower():
                        return self._extract_metadata_from_search(book)
                
                # If no exact match, return first result
                return self._extract_metadata_from_search(data['docs'][0])
            
            return None
            
        except Exception as e:
            print(f"OpenLibrary Arabic exact search failed: {e}")
            return None
    
    def _search_arabic_partial(self, title: str, authors: List[str] = None) -> Optional[Dict]:
        """Search for partial Arabic title match"""
        try:
            # Build search query for partial Arabic match
            query_parts = [title]  # No quotes for partial matching
            if authors:
                for author in authors:
                    if author and author.strip():
                        query_parts.append(f'author:"{author}"')
            
            query = ' '.join(query_parts)
            
            url = f"{self.base_url}/search.json"
            params = {
                'q': query,
                'limit': 10,
                'language': 'ara'  # Arabic language filter
            }
            
            response = requests.get(url, params=params, headers=self.headers)
            response.raise_for_status()
            
            data = response.json()
            
            if data.get('docs') and len(data['docs']) > 0:
                # Return first result (most relevant)
                return self._extract_metadata_from_search(data['docs'][0])
            
            return None
            
        except Exception as e:
            print(f"OpenLibrary Arabic partial search failed: {e}")
            return None
    
    
    def _extract_metadata(self, book_data: Dict) -> Dict:
        """
        Extract metadata from OpenLibrary book data (ISBN search)
        """
        print(f"DEBUG: _extract_metadata called")
        print(f"DEBUG: book_data keys: {list(book_data.keys())}")
        print(f"DEBUG: book_data type: {type(book_data)}")
        
        # Handle authors - OpenLibrary ISBN search doesn't include authors field
        # We'll need to get authors from a different source or leave empty
        authors = []
        
        # Handle publishers - they can be strings or dictionaries
        publishers = []
        publishers_data = book_data.get('publishers', [])
        print(f"DEBUG: publishers_data: {publishers_data}")
        print(f"DEBUG: publishers_data type: {type(publishers_data)}")
        
        if publishers_data:
            for i, publisher in enumerate(publishers_data):
                print(f"DEBUG: publisher[{i}]: {publisher}")
                print(f"DEBUG: publisher[{i}] type: {type(publisher)}")
                if isinstance(publisher, dict):
                    publisher_name = publisher.get('name', '')
                    print(f"DEBUG: extracted name from dict: {publisher_name}")
                    publishers.append(publisher_name)
                else:
                    publisher_str = str(publisher)
                    print(f"DEBUG: converted to string: {publisher_str}")
                    publishers.append(publisher_str)
        
        print(f"DEBUG: final publishers list: {publishers}")
        
        metadata = {
            'title': book_data.get('title', ''),
            'author': ', '.join(authors),  # Empty for now since authors not in ISBN search
            'publisher': ', '.join(publishers),
            'published_date': book_data.get('publish_date', ''),
            'd_o_pub': book_data.get('publish_date', ''),
            'oclc_no': '',
            'lc_no': '',
            'isbn_10': '',
            'isbn_13': '',
            'isbn': '',
            'source': 'openlibrary_isbn'
        }
        
        print(f"DEBUG: metadata created: {metadata}")
        return metadata
        
        # Extract ISBNs
        isbns = []
        identifiers = book_data.get('identifiers', {})
        if 'isbn_10' in identifiers and identifiers['isbn_10']:
            metadata['isbn_10'] = identifiers['isbn_10'][0] if isinstance(identifiers['isbn_10'], list) else identifiers['isbn_10']
            isbns.append(metadata['isbn_10'])
        if 'isbn_13' in identifiers and identifiers['isbn_13']:
            metadata['isbn_13'] = identifiers['isbn_13'][0] if isinstance(identifiers['isbn_13'], list) else identifiers['isbn_13']
            isbns.append(metadata['isbn_13'])
        
        metadata['isbn'] = '; '.join(isbns) if isbns else ''
        
        return metadata
    
    def _extract_metadata_from_search(self, book_data: Dict) -> Dict:
        """
        Extract metadata from OpenLibrary search results
        """
        print(f"DEBUG: _extract_metadata_from_search called")
        print(f"DEBUG: book_data keys: {list(book_data.keys())}")
        print(f"DEBUG: book_data type: {type(book_data)}")
        
        # Handle authors - they can be strings or dictionaries
        authors = []
        author_data = book_data.get('author_name', [])
        print(f"DEBUG: author_data: {author_data}")
        print(f"DEBUG: author_data type: {type(author_data)}")
        
        if author_data:
            for i, author in enumerate(author_data):
                print(f"DEBUG: author[{i}]: {author}")
                print(f"DEBUG: author[{i}] type: {type(author)}")
                if isinstance(author, dict):
                    author_name = author.get('name', '')
                    print(f"DEBUG: extracted name from dict: {author_name}")
                    authors.append(author_name)
                else:
                    author_str = str(author)
                    print(f"DEBUG: converted to string: {author_str}")
                    authors.append(author_str)
        
        print(f"DEBUG: final authors list: {authors}")
        
        # Handle publishers
        publisher_data = book_data.get('publisher', [])
        print(f"DEBUG: publisher_data: {publisher_data}")
        print(f"DEBUG: publisher_data type: {type(publisher_data)}")
        
        metadata = {
            'title': book_data.get('title', ''),
            'author': ', '.join(authors),
            'publisher': ', '.join(publisher_data) if isinstance(publisher_data, list) else str(publisher_data),
            'published_date': str(book_data.get('first_publish_year', '')),
            'd_o_pub': str(book_data.get('first_publish_year', '')),
            'oclc_no': '',
            'lc_no': '',
            'isbn_10': '',
            'isbn_13': '',
            'isbn': '',
            'source': 'openlibrary_search'
        }
        
        print(f"DEBUG: metadata created: {metadata}")
        return metadata
        
        # Extract ISBNs from search results
        isbns = []
        if 'isbn' in book_data:
            for isbn in book_data['isbn']:
                if len(isbn) == 10:
                    metadata['isbn_10'] = isbn
                    isbns.append(isbn)
                elif len(isbn) == 13:
                    metadata['isbn_13'] = isbn
                    isbns.append(isbn)
        
        metadata['isbn'] = '; '.join(isbns) if isbns else ''
        
        return metadata
    