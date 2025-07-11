from src.utils.fuzzy import fuzzy_match
from src.utils.google_books import search_book_by_isbn, search_book_by_title_author, extract_book_metadata
from src.utils.openlibrary import OpenLibraryAPI


def get_unified_metadata(title, authors, isbns, lccns=None):
    """
    Query Google Books, OpenLibrary, and LOC and return unified metadata fields.
    Uses fuzzy matching for fallback on title/author if ISBN not found.
    lccns: optional, a list or string of LCCNs to use for the 'LC no.' field.
    Returns a dict with keys: TITLE, AUTHOR, PUBLISHED, D.O Pub., OCLC no., LC no., ISBN
    """
    gb_data = None
    ol_data = None
    api = OpenLibraryAPI()
    found_by_isbn = False
    if isbns:
        for isbn in isbns:
            # Google Books
            try:
                gb_result = search_book_by_isbn(isbn)
                if gb_result and gb_result.get('items'):
                    gb_data = extract_book_metadata(gb_result)
                    if gb_data.get('isbn') and isbn in gb_data['isbn']:
                        found_by_isbn = True
                        break
                    else:
                        gb_data = None
            except Exception:
                pass
            # OpenLibrary
            try:
                ol_data = api.search_by_isbn(isbn)
                if ol_data and ol_data.get('isbn') and isbn in ol_data['isbn']:
                    found_by_isbn = True
                    break
                else:
                    ol_data = None
            except Exception:
                pass
    # Fallback: search by title/author if no ISBN match
    found_by_fallback = False
    fallback_gb = None
    fallback_ol = None
    if not found_by_isbn and title:
        try:
            gb_result = search_book_by_title_author(title, authors)
            if gb_result and gb_result.get('items'):
                candidate = extract_book_metadata(gb_result)
                title_ratio = fuzzy_match(candidate.get('title', ''), title)
                author_ratio = fuzzy_match(candidate.get('author', ''), ', '.join(authors) if authors else '')
                if title_ratio >= 0.9 and author_ratio >= 0.9:
                    fallback_gb = candidate
                    found_by_fallback = True
        except Exception:
            pass
    if not found_by_isbn and not found_by_fallback and title:
        try:
            ol_candidate = api.search_by_title_author(title, authors)
            if ol_candidate:
                title_ratio = fuzzy_match(ol_candidate.get('title', ''), title)
                author_ratio = fuzzy_match(ol_candidate.get('author', ''), ', '.join(authors) if authors else '')
                if title_ratio >= 0.9 and author_ratio >= 0.9:
                    fallback_ol = ol_candidate
                    found_by_fallback = True
        except Exception:
            pass
    if isinstance(lccns, list):
        lccn_str = '; '.join([l for l in lccns if l])
    elif isinstance(lccns, str):
        lccn_str = lccns
    else:
        lccn_str = ''
    if found_by_isbn:
        use_gb = gb_data if gb_data else None
        use_ol = ol_data if ol_data else None
    elif found_by_fallback:
        use_gb = fallback_gb if fallback_gb else None
        use_ol = fallback_ol if fallback_ol else None
    else:
        use_gb = None
        use_ol = None
    unified = {
        'TITLE': use_gb['title'] if use_gb and use_gb.get('title') else (use_ol['title'] if use_ol else title),
        'AUTHOR': use_gb['author'] if use_gb and use_gb.get('author') else (use_ol['author'] if use_ol else ', '.join(authors) if authors else ''),
        'PUBLISHED': use_gb['publisher'] if use_gb and use_gb.get('publisher') else (use_ol['publisher'] if use_ol else ''),
        'D.O Pub.': use_gb['published_date'] if use_gb and use_gb.get('published_date') else (use_ol['published_date'] if use_ol else ''),
        'OCLC no.': use_ol['oclc_no'] if use_ol and 'oclc_no' in use_ol else '',
        'LC no.': lccn_str,
        'ISBN': use_gb['isbn'] if use_gb and use_gb.get('isbn') else (use_ol['isbn'] if use_ol else '; '.join(isbns) if isbns else ''),
    }
    return unified 