"""
Google Books API utility functions
"""
import os
from googleapiclient.discovery import build
import re
from config.config import Config

def get_google_books_service(api_key=None):
    api_key = Config.GOOGLE_BOOKS_API_KEY
    if not api_key:
        raise RuntimeError("Google Books API key is missing or invalid!")
    return build('books', 'v1', developerKey=api_key)

def search_book_by_isbn(isbn, api_key=None):
    """Search for a book by ISBN"""
    service = get_google_books_service(api_key)
    request = service.volumes().list(q=f'isbn:{isbn}')
    response = request.execute()
    return response

def search_book_by_title_author(title, authors=None, api_key=None, max_results=5):
    """
    Search for a book by title and optionally authors
    Args:
        title (str): Book title
        authors (list): List of author names
        api_key (str): Google Books API key
        max_results (int): Maximum number of results to return
    Returns:
        dict: Google Books API response
    """
    service = get_google_books_service(api_key)
    
    # Build search query
    query_parts = []
    
    # Add title (in quotes for exact match)
    if title:
        # Clean title - remove extra spaces and special characters
        clean_title = re.sub(r'\s+', ' ', title.strip())
        query_parts.append(f'"{clean_title}"')
    
    # Add authors if provided
    if authors:
        for author in authors:
            if author and author.strip():
                clean_author = re.sub(r'\s+', ' ', author.strip())
                query_parts.append(f'author:"{clean_author}"')
    
    # Combine query parts
    query = ' '.join(query_parts)
    
    # Execute search
    request = service.volumes().list(
        q=query,
        maxResults=max_results,
        orderBy='relevance'
    )
    response = request.execute()
    return response

def search_arabic_book(title, authors=None, api_key=None):
    """
    Specialized search for Arabic books
    Args:
        title (str): Arabic book title
        authors (list): List of Arabic author names
        api_key (str): Google Books API key
    Returns:
        dict: Google Books API response
    """
    service = get_google_books_service(api_key)
    
    # Build search query for Arabic books
    query_parts = []
    
    if title:
        # For Arabic books, we might want to try both Arabic and transliterated versions
        clean_title = re.sub(r'\s+', ' ', title.strip())
        query_parts.append(f'"{clean_title}"')
    
    if authors:
        for author in authors:
            if author and author.strip():
                clean_author = re.sub(r'\s+', ' ', author.strip())
                query_parts.append(f'author:"{clean_author}"')
    
    # Add language filter for Arabic books
    query = ' '.join(query_parts)
    
    # Execute search with language preference
    request = service.volumes().list(
        q=query,
        maxResults=10,
        orderBy='relevance',
        langRestrict='ar'  # Restrict to Arabic language
    )
    response = request.execute()
    return response

def extract_book_metadata(google_books_response):
    """
    Extract useful metadata from Google Books API response
    Args:
        google_books_response (dict): Response from Google Books API
    Returns:
        dict: Extracted metadata
    """
    if not google_books_response.get('items'):
        return None
    
    # Get the first (most relevant) result
    book = google_books_response['items'][0]
    volume_info = book.get('volumeInfo', {}) 
    
    metadata = {
        'title': volume_info.get('title', ''),
        'author': ', '.join(volume_info.get('authors', [])) if volume_info.get('authors') else '',
        'publisher': volume_info.get('publisher', ''),
        'published_date': volume_info.get('publishedDate', ''),
        'd_o_pub': volume_info.get('publishedDate', ''),  # Date of Publication
        'oclc_no': '',  # OCLC number (not available in Google Books API)
        'lc_no': '',    # Library of Congress number (not available in Google Books API)
        'isbn_10': '',
        'isbn_13': '',
        'isbn': ''  # Combined ISBN field
    }
    
    # Extract ISBNs from industry identifiers
    industry_identifiers = volume_info.get('industryIdentifiers', [])
    isbns = []
    for identifier in industry_identifiers:
        if identifier.get('type') == 'ISBN_10':
            metadata['isbn_10'] = identifier.get('identifier', '')
            isbns.append(identifier.get('identifier', ''))
        elif identifier.get('type') == 'ISBN_13':
            metadata['isbn_13'] = identifier.get('identifier', '')
            isbns.append(identifier.get('identifier', ''))
    
    # Set combined ISBN field
    metadata['isbn'] = '; '.join(isbns) if isbns else ''
    
    return metadata
