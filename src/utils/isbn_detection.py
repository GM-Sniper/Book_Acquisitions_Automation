from isbntools.app import get_isbnlike


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
