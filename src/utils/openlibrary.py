import requests
import time
from typing import List, Optional, Dict, Any

class OpenLibraryAPI:
    BASE_URL = "https://openlibrary.org"
    HEADERS = {"User-Agent": "BookLookup/1.0"}

    def __init__(self, debug: bool = False, rate_limit: float = 1.0):
        self.debug = debug
        self.rate_limit = rate_limit
        self.last_request_time = 0

    def _rate_limit_wait(self):
        """Simple rate limiting to be respectful"""
        if self.rate_limit > 0:
            elapsed = time.time() - self.last_request_time
            if elapsed < self.rate_limit:
                time.sleep(self.rate_limit - elapsed)
        self.last_request_time = time.time()

    def search_by_isbn(self, isbn: str) -> Optional[Dict]:
        """Search by ISBN - handles both formats OpenLibrary expects"""
        clean_isbn = ''.join(c for c in isbn if c.isdigit() or c.upper() == 'X')
        
        url = f"{self.BASE_URL}/api/books"
        params = {
            "bibkeys": f"ISBN:{clean_isbn}",
            "format": "json",
            "jscmd": "data"
        }
        data = self._get(url, params)
        book_data = data.get(f"ISBN:{clean_isbn}")
        return self._parse_book_data(book_data, source="isbn") if book_data else None

    def fetch_work_details(self, work_key: str) -> Optional[Dict]:
        """Fetch work details from OpenLibrary using the work key (e.g., '/works/OL45804W')"""
        if not work_key:
            return None
        url = f"{self.BASE_URL}{work_key}.json"
        data = self._get(url, {})
        if not data:
            return None
        # Optionally fetch editions
        editions_url = f"{self.BASE_URL}{work_key}/editions.json"
        editions_data = self._get(editions_url, {"limit": 1, "sort": "last_modified"})
        latest_edition = None
        if editions_data and editions_data.get("entries"):
            latest_edition = editions_data["entries"][0]
        return {"work": data, "latest_edition": latest_edition}

    def search_by_title_author(self, title: str, authors: Optional[List[str]] = None, lang: Optional[str] = None, expand_work: bool = True) -> Optional[Dict]:
        query = self._build_query(title, authors)
        results = self._search(query, lang=lang)
        if not results:
            return None

        print(f"🔍 OPENLIBRARY SEARCH RESULTS-----------------------------: {results}")
        book_data = self._find_best_match(results, title, authors)
        parsed = self._parse_book_data(book_data, source="search") if book_data else None
        # Expand with work details if requested and available
        if expand_work and book_data and book_data.get("key", "").startswith("/works/"):
            work_details = self.fetch_work_details(book_data["key"])
            if work_details:
                parsed["work_details"] = work_details["work"]
                parsed["latest_edition"] = work_details["latest_edition"]
        return parsed

    def _build_query(self, title: str, authors: Optional[List[str]]) -> str:
        """Build query with proper escaping"""
        clean_title = title.replace('"', '\\"')
        query = f'title:"{clean_title}"'
        if authors:
            for author in authors:
                if author and author.strip():
                    clean_author = author.replace('"', '\\"')
                    query += f' author:"{clean_author}"'
        return query

    def _search(self, query: str, lang: Optional[str] = None) -> List[Dict[str, Any]]:
        url = f"{self.BASE_URL}/search.json"
        params = {"q": query, "limit": 10}
        if lang:
            params["language"] = lang
        data = self._get(url, params)
        return data.get("docs", [])

    def _find_best_match(self, books: List[Dict], title: str, authors: Optional[List[str]] = None) -> Optional[Dict]:
        """Better matching logic"""
        if not books:
            return None
            
        # Look for exact title match first
        title_lower = title.lower()
        for book in books:
            if book.get("title", "").lower() == title_lower:
                return book
        
        # If authors provided, try to match on author too
        if authors:
            author_lower = [a.lower() for a in authors]
            for book in books:
                book_title = book.get("title", "").lower()
                book_authors = [a.lower() for a in book.get("author_name", [])]
                
                if (title_lower in book_title or book_title in title_lower) and \
                   any(auth in " ".join(book_authors) for auth in author_lower):
                    return book
        
        return books[0] if books else None

    def _parse_book_data(self, data: Dict, source: str = "search") -> Dict:
        if self.debug:
            print(f"\n📚 RAW OPENLIBRARY DATA for {source}:")
            print(f"Title: {data.get('title', 'N/A')}")
            print(f"Authors field: {data.get('authors', 'N/A')}")
            print(f"Author_name field: {data.get('author_name', 'N/A')}")
            print(f"Publishers field: {data.get('publishers', 'N/A')}")
            print(f"Publisher field: {data.get('publisher', 'N/A')}")
            print(f"Publish date: {data.get('publish_date', 'N/A')}")
            print(f"First publish year: {data.get('first_publish_year', 'N/A')}")
            print(f"ISBNs: {data.get('isbn', 'N/A')}")
            print(f"Identifiers: {data.get('identifiers', 'N/A')}")
            print()
        
        # Handle authors - different formats for search vs isbn
        if source == "search":
            authors = data.get("author_name", [])
        else:  # isbn source
            authors_data = data.get("authors", [])
            if isinstance(authors_data, list):
                authors = []
                for author in authors_data:
                    if isinstance(author, dict):
                        authors.append(author.get('name', ''))
                    else:
                        authors.append(str(author))
            else:
                authors = []
        
        # Handle publishers - different for search vs isbn
        if source == "search":
            publishers = data.get("publisher", [])
        else:  # isbn source
            publishers_data = data.get("publishers", [])
            if isinstance(publishers_data, list):
                publishers = []
                for pub in publishers_data:
                    if isinstance(pub, dict):
                        publishers.append(pub.get('name', ''))
                    else:
                        publishers.append(str(pub))
            else:
                publishers = publishers_data if publishers_data else []
        
        # Handle ISBNs - different sources
        if source == "search":
            isbns = data.get("isbn", [])
            isbn_10 = next((i for i in isbns if len(i) == 10), "")
            isbn_13 = next((i for i in isbns if len(i) == 13), "")
        else:  # isbn source
            identifiers = data.get("identifiers", {})
            isbn_10_list = identifiers.get("isbn_10", [])
            isbn_13_list = identifiers.get("isbn_13", [])
            isbn_10 = isbn_10_list[0] if isbn_10_list else ""
            isbn_13 = isbn_13_list[0] if isbn_13_list else ""
        
        # Handle publish date
        if source == "search":
            publish_years = data.get("publish_year", [])
            first_year = data.get("first_publish_year")
            if publish_years:
                publish_date = str(max(publish_years))
            elif first_year:
                publish_date = str(first_year)
            else:
                publish_date = ""
        else:  # isbn source
            publish_date = data.get("publish_date", "")
        
        # Handle other identifiers for isbn source
        oclc_no = ""
        lc_no = ""
        if source == "isbn":
            identifiers = data.get("identifiers", {})
            oclc_list = identifiers.get("oclc", [])
            lccn_list = identifiers.get("lccn", [])
            oclc_no = ", ".join(oclc_list) if oclc_list else ""
            lc_no = ", ".join(lccn_list) if lccn_list else ""

        result = {
            "title": data.get("title", ""),
            "author": ", ".join(authors) if authors else "",
            "publisher": ", ".join(publishers) if publishers else "",
            "published_date": str(publish_date),
            "d_o_pub": str(publish_date),
            "oclc_no": oclc_no,
            "lc_no": lc_no,
            "isbn_10": isbn_10,
            "isbn_13": isbn_13,
            "isbn": "; ".join(filter(None, [isbn_10, isbn_13])),
            "source": f"openlibrary_{source}"
        }
        
        if self.debug:
            print(f"🔍 PARSED RESULT:")
            print(f"   Title: {result.get('title', 'N/A')}")
            print(f"   Author: {result.get('author', 'N/A')}")
            print(f"   Publisher: {result.get('publisher', 'N/A')}")
            print(f"   Published: {result.get('published_date', 'N/A')}")
            print(f"   ISBN-10: {result.get('isbn_10', 'N/A')}")
            print(f"   ISBN-13: {result.get('isbn_13', 'N/A')}")
            print(f"   OCLC: {result.get('oclc_no', 'N/A')}")
            print(f"   LC: {result.get('lc_no', 'N/A')}")
            print()
        
        return result

    def _get(self, url: str, params: Dict) -> Dict:
        try:
            self._rate_limit_wait()
            
            if self.debug:
                print(f"🌐 Requesting: {url} with params: {params}")
            
            response = requests.get(url, params=params, headers=self.HEADERS, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            if self.debug:
                print(f"ERROR: Request failed for {url}: {e}")
            return {}