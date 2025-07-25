"""
Enhanced ISBN Service using isbnlib for reliable book metadata lookup (No LLM version)
"""
import isbnlib
import logging
import time
import json
from typing import Optional, Dict, List, Union
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Set up logging
logger = logging.getLogger(__name__)

class ISBNService:
    """Enhanced and reliable ISBN lookup service using isbnlib (No LLM)"""
    
    # Available services in order of preference
    SERVICES = ['openl', 'goob', 'wiki']
    
    def __init__(self, debug: bool = False, rate_limit: float = 1.0, timeout: int = 30):
        """
        Initialize ISBN service
        Args:
            debug (bool): Enable debug logging
            rate_limit (float): Minimum seconds between requests
            timeout (int): Request timeout in seconds
        """
        self.debug = debug
        self.rate_limit = rate_limit
        self.timeout = timeout
        self.last_request_time = 0
        
        # Setup session with retries
        self.session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
    
    def _rate_limit_wait(self):
        """Implement rate limiting between requests"""
        if self.rate_limit > 0:
            elapsed = time.time() - self.last_request_time
            if elapsed < self.rate_limit:
                time.sleep(self.rate_limit - elapsed)
        self.last_request_time = time.time()
    
    def search_by_isbn(self, isbn: str, merge_results: bool = True) -> Optional[Dict]:
        """
        Search for book metadata by ISBN using multiple services
        Args:
            isbn (str): ISBN to search for (10 or 13 digits)
            merge_results (bool): Whether to merge results from multiple services
        Returns:
            dict: Book metadata or None if not found
        """
        try:
            # Clean and validate the ISBN
            clean_isbn = isbnlib.clean(isbn)
            if not clean_isbn or not self.validate_isbn(clean_isbn):
                return None
            
            if merge_results:
                return self._search_with_merge(clean_isbn)
            else:
                return self._search_single_service(clean_isbn)
            
        except Exception as e:
            return None
    
    def _search_single_service(self, clean_isbn: str) -> Optional[Dict]:
        """Search using single service approach (original method)"""
        for service in self.SERVICES:
            try:
                self._rate_limit_wait()
                metadata = isbnlib.meta(clean_isbn, service=service)
                if metadata and self._is_valid_metadata(metadata):
                    result = self._parse_metadata(metadata, clean_isbn, service)
                    return result
            except Exception as e:
                continue
        return None
    
    def _search_with_merge(self, clean_isbn: str) -> Optional[Dict]:
        """Search using merge approach - combine results from multiple services"""
        all_results = []
        for service in self.SERVICES:
            try:
                self._rate_limit_wait()
                metadata = isbnlib.meta(clean_isbn, service=service)
                if metadata and self._is_valid_metadata(metadata):
                    result = self._parse_metadata(metadata, clean_isbn, service)
                    all_results.append(result)
            except Exception as e:
                continue
        if not all_results:
            return None
        merged_result = self._merge_results(all_results, clean_isbn)
        return merged_result
    
    def search_by_title_author(self, title: str, authors: Optional[List[str]] = None, 
                             limit: int = 5) -> Optional[Dict]:
        """
        Search for book metadata by title and authors
        Args:
            title (str): Book title
            authors (list): List of author names
            limit (int): Maximum number of ISBNs to try
        Returns:
            dict: Book metadata or None if not found
        """
        try:
            # Build search query
            query_parts = [title]
            if authors:
                query_parts.extend(authors)
            query = ' '.join(query_parts)
            self._rate_limit_wait()
            # Search for ISBNs
            isbns = isbnlib.isbn_from_words(query)
            if not isbns:
                return None
            # Try each ISBN found (up to limit)
            for isbn in isbns[:limit]:
                result = self.search_by_isbn(isbn)
                if result:
                    # Verify the result matches our search criteria
                    if self._matches_search_criteria(result, title, authors):
                        return result
            return None
        except Exception as e:
            return None
    
    def _is_valid_metadata(self, metadata: Dict) -> bool:
        """Check if metadata contains minimum required information"""
        return (
            metadata and 
            isinstance(metadata, dict) and
            (metadata.get('Title') or metadata.get('title')) and
            len(str(metadata.get('Title', metadata.get('title', ''))).strip()) > 0
        )
    
    def _matches_search_criteria(self, result: Dict, title: str, authors: Optional[List[str]]) -> bool:
        """Verify search result matches original search criteria"""
        if not result:
            return False
        # Check title similarity (basic check)
        result_title = result.get('title', '').lower()
        search_title = title.lower()
        # Allow partial matches for flexibility
        title_match = search_title in result_title or result_title in search_title
        # Check author match if provided
        author_match = True
        if authors:
            result_authors = result.get('author', '').lower()
            author_match = any(author.lower() in result_authors for author in authors)
        return title_match and author_match
    
    def _parse_metadata(self, metadata: Dict, isbn: str, source: str) -> Dict:
        """
        Parse metadata into standardized format with better field handling
        Args:
            metadata (dict): Raw metadata from service
            isbn (str): Original ISBN
            source (str): Source service name
        Returns:
            dict: Standardized metadata
        """
        # Handle different field name variations
        title = (metadata.get('Title') or metadata.get('title') or 
                metadata.get('TITLE') or '').strip()
        # Extract authors with multiple possible field names
        authors = (metadata.get('Authors') or metadata.get('authors') or 
                  metadata.get('Author') or metadata.get('author') or [])
        if isinstance(authors, str):
            # Handle comma-separated authors
            authors = [a.strip() for a in authors.split(',') if a.strip()]
        elif not isinstance(authors, list):
            authors = [str(authors)] if authors else []
        # Extract publishers
        publishers = (metadata.get('Publisher') or metadata.get('publisher') or 
                     metadata.get('publishers') or [])
        if isinstance(publishers, str):
            publishers = [publishers]
        elif not isinstance(publishers, list):
            publishers = [str(publishers)] if publishers else []
        # Extract publication year with multiple possible formats
        year = (metadata.get('Year') or metadata.get('year') or 
               metadata.get('published') or metadata.get('publication_date') or '')
        # Clean up year if it's a full date
        if isinstance(year, str) and len(year) > 4:
            import re
            year_match = re.search(r'\b(19|20)\d{2}\b', year)
            if year_match:
                year = year_match.group()
        # Extract ISBNs
        isbn_10 = metadata.get('ISBN-10', '')
        isbn_13 = metadata.get('ISBN-13', '')
        # Try to convert if one is missing
        if isbn_13 and not isbn_10:
            try:
                isbn_10 = isbnlib.to_isbn10(isbn_13) or ''
            except:
                pass
        elif isbn_10 and not isbn_13:
            try:
                isbn_13 = isbnlib.to_isbn13(isbn_10) or ''
            except:
                pass
        # Combine all ISBNs
        all_isbns = []
        for potential_isbn in [isbn_10, isbn_13, isbn]:
            clean = isbnlib.clean(str(potential_isbn))
            if clean and clean not in all_isbns:
                all_isbns.append(clean)
        result = {
            'title': title,
            'author': ', '.join(authors) if authors else '',
            'authors': authors,  # Keep original list too
            'publisher': ', '.join(publishers) if publishers else '',
            'publishers': publishers,  # Keep original list too
            'published_date': str(year),
            'd_o_pub': str(year),
            'year': str(year),
            'oclc_no': str(metadata.get('OCLC', metadata.get('oclc', ''))),
            'lc_no': str(metadata.get('LC', metadata.get('lc', ''))),
            'isbn_10': isbn_10,
            'isbn_13': isbn_13,
            'isbn': '; '.join(all_isbns),
            'all_isbns': all_isbns,
            'source': f'isbnlib_{source}',
            'raw_metadata': metadata if self.debug else None,
            'subjects': metadata.get('Subjects', metadata.get('subjects', ''))
        }
        return result
    
    def _merge_results(self, results: List[Dict], isbn: str) -> Dict:
        """
        Intelligently merge results from multiple services for better accuracy
        Args:
            results (list): List of parsed metadata results
            isbn (str): Original ISBN
        Returns:
            dict: Merged metadata with best available information
        """
        if not results:
            return {}
        if len(results) == 1:
            return results[0]
        merged = results[0].copy()
        enhanced_fields = set()
        for result in results[1:]:
            # Merge title - prefer longer/more complete titles
            current_title = merged.get('title', '')
            new_title = result.get('title', '')
            if len(new_title) > len(current_title) and new_title:
                merged['title'] = new_title
                enhanced_fields.add('title')
            # Merge author - combine unique authors
            current_authors = merged.get('author', '')
            new_authors = result.get('author', '')
            if new_authors and new_authors not in current_authors:
                if current_authors:
                    merged['author'] = f"{current_authors}; {new_authors}"
                else:
                    merged['author'] = new_authors
                enhanced_fields.add('author')
            # Merge publisher - prefer non-empty publishers
            if not merged.get('publisher') and result.get('publisher'):
                merged['publisher'] = result['publisher']
                enhanced_fields.add('publisher')
            # Merge publication date - prefer more specific dates
            current_year = merged.get('published_date', '')
            new_year = result.get('published_date', '')
            if new_year and (not current_year or len(new_year) > len(current_year)):
                merged['published_date'] = new_year
                merged['d_o_pub'] = new_year
                merged['year'] = new_year
                enhanced_fields.add('published_date')
            # Merge ISBNs - combine all unique ISBNs
            current_isbns = set(merged.get('isbn', '').split('; ') if merged.get('isbn') else [])
            new_isbns = set(result.get('isbn', '').split('; ') if result.get('isbn') else [])
            all_isbns = current_isbns.union(new_isbns)
            if all_isbns:
                merged['isbn'] = '; '.join(sorted(all_isbns))
                enhanced_fields.add('isbn')
            # Merge language - prefer more specific language info
            if not merged.get('language') and result.get('language'):
                merged['language'] = result['language']
                enhanced_fields.add('language')
            # Merge format - prefer more specific format info
            if not merged.get('format') and result.get('format'):
                merged['format'] = result['format']
                enhanced_fields.add('format')
            # Merge subjects - combine unique subjects
            current_subjects = merged.get('subjects', '')
            new_subjects = result.get('subjects', '')
            if new_subjects and new_subjects not in current_subjects:
                if current_subjects:
                    merged['subjects'] = f"{current_subjects}; {new_subjects}"
                else:
                    merged['subjects'] = new_subjects
                enhanced_fields.add('subjects')
            # Merge OCLC and LC numbers - prefer non-empty values
            if not merged.get('oclc_no') and result.get('oclc_no'):
                merged['oclc_no'] = result['oclc_no']
                enhanced_fields.add('oclc_no')
            if not merged.get('lc_no') and result.get('lc_no'):
                merged['lc_no'] = result['lc_no']
                enhanced_fields.add('lc_no')
        # Update source to indicate merged results
        sources = list(set([r.get('source', '') for r in results if r.get('source')]))
        merged['source'] = f"merged_{'_'.join(sources)}"
        return merged
    
    def validate_isbn(self, isbn: str) -> bool:
        """
        Validate ISBN format and checksum
        Args:
            isbn (str): ISBN to validate
        Returns:
            bool: True if valid, False otherwise
        """
        try:
            clean_isbn = isbnlib.clean(isbn)
            if not clean_isbn:
                return False
            return isbnlib.is_isbn10(clean_isbn) or isbnlib.is_isbn13(clean_isbn)
        except:
            return False

# Convenience functions

def quick_isbn_search(isbn: str, debug: bool = False, merge_results: bool = True, **kwargs) -> Optional[Dict]:
    """Quick ISBN search with enhanced service and result merging (No LLM)"""
    service = ISBNService(debug=debug, **kwargs)
    return service.search_by_isbn(isbn, merge_results=merge_results)

def quick_title_search(title: str, authors: Optional[List[str]] = None, 
                      debug: bool = False, merge_results: bool = True, **kwargs) -> Optional[Dict]:
    """Quick title/author search with enhanced service and result merging (No LLM)"""
    service = ISBNService(debug=debug, **kwargs)
    return service.search_by_title_author(title, authors)

def validate_isbn(isbn: str) -> bool:
    """Quick ISBN validation"""
    service = ISBNService()
    return service.validate_isbn(isbn)