from isbntools.app import get_isbnlike
import requests
import re


def extract_isbns(text):
    """
    Extracts ISBN-10 and ISBN-13 numbers from a given text using isbntools.
    Args:
        text (str): The text to search for ISBNs.
    Returns:
        dict: A dictionary with keys 'isbn10' and 'isbn13', each containing a list of found ISBNs.
    """
    isbns = get_isbnlike(text)
    isbn10 = [isbn for isbn in isbns if len(isbn.replace('-', '')) == 10]
    isbn13 = [isbn for isbn in isbns if len(isbn.replace('-', '')) == 13]
    return {'isbn10': isbn10, 'isbn13': isbn13}

def validate_isbn_online(isbn):
    """
    Validate ISBN against Google Books API.
    Args:
        isbn (str): ISBN to validate.
    Returns:
        dict: Validation result with book information if found.
    """
    try:
        # Try Google Books API
        google_url = f"https://www.googleapis.com/books/v1/volumes?q=isbn:{isbn}"
        response = requests.get(google_url, timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            if data.get('totalItems', 0) > 0:
                book_info = data['items'][0]['volumeInfo']
                return {
                    'valid': True,
                    'title': book_info.get('title', ''),
                    'authors': book_info.get('authors', []),
                    'publisher': book_info.get('publisher', ''),
                    'published_date': book_info.get('publishedDate', ''),
                    'source': 'Google Books'
                }
        
        return {'valid': False, 'error': 'Not found in online databases'}
        
    except Exception as e:
        return {'valid': False, 'error': f'Online validation failed: {str(e)}'}

def extract_and_validate_isbns(text):
    """
    Extract ISBNs and validate them online.
    Args:
        text (str): The text to search for ISBNs.
    Returns:
        dict: Enhanced ISBN information with validation results.
    """
    # Use existing extraction
    isbn_results = extract_isbns(text)
    
    # Validate each ISBN found
    validated_isbns = []
    
    for isbn_type, isbn_list in isbn_results.items():
        for isbn in isbn_list:
            # Clean the ISBN
            clean_isbn = isbn.replace('-', '')
            validation = validate_isbn_online(clean_isbn)
            
            validated_isbns.append({
                'isbn': clean_isbn,
                'original': isbn,
                'type': isbn_type,
                'validation': validation
            })
    
    return {
        'isbns': validated_isbns,
        'total_found': len(validated_isbns),
        'validated_count': len([isbn for isbn in validated_isbns if isbn['validation']['valid']])
    }
