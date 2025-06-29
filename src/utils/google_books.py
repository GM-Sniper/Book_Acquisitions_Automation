"""
Google Books API utility functions
"""
import os
from googleapiclient.discovery import build

def get_google_books_service(api_key=None):
    """Initialize Google Books API service"""
    if api_key is None:
        api_key = os.getenv('GOOGLE_BOOKS_API_KEY')
    return build('books', 'v1', developerKey=api_key)

# Example function to search for a book by ISBN
def search_book_by_isbn(isbn, api_key=None):
    service = get_google_books_service(api_key)
    request = service.volumes().list(q=f'isbn:{isbn}')
    response = request.execute()
    return response 