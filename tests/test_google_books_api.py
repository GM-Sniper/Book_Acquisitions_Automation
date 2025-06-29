import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.google_books import search_book_by_isbn

def test_google_books():
    isbn = "9780140328721"  # Example: Matilda by Roald Dahl
    result = search_book_by_isbn(isbn)
    print("Google Books API result:", result)

if __name__ == "__main__":
    test_google_books() 