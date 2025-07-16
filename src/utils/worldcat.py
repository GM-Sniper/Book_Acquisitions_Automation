import requests
import time
from typing import Optional, List, Dict
from config.config import Config

class WorldCatAPIv2:
    """
    WorldCat Search API v2 client.
    Uses OAuth2 Client Credentials grant to authenticate.
    """
    def __init__(self):
        self.config = Config()
        self.client_id = self.config.WORLDCAT_CLIENT_ID
        self.client_secret = self.config.WORLDCAT_CLIENT_SECRET
        self.region = "americas"  # adjust if needed
        self.token_url = "https://oauth.oclc.org/token"
        self.api_base_url = f"https://{self.region}.discovery.api.oclc.org/worldcat/search/v2"
        self.access_token = None
        self.token_expires_at = 0

    def get_access_token(self):
        """Get (or refresh) OAuth2 access token"""
        if self.access_token and time.time() < self.token_expires_at - 60:
            return self.access_token  # still valid

        print("[INFO] Requesting new access token...")
        data = {
            'grant_type': 'client_credentials',
            'client_id': self.client_id,
            'client_secret': self.client_secret,
            'scope': 'WorldCatSearchAPI'
        }
        try:
            resp = requests.post(self.token_url, data=data, timeout=10)
            resp.raise_for_status()
            token_data = resp.json()
            self.access_token = token_data['access_token']
            self.token_expires_at = time.time() + token_data['expires_in']
            return self.access_token
        except requests.RequestException as e:
            print(f"[ERROR] Token request failed: {e}")
            return None

    def search_by_isbn(self, isbn: str) -> Optional[Dict]:
        """Search for a book by ISBN"""
        token = self.get_access_token()
        url = f"{self.api_base_url}/bibs"
        params = {"q": f"isbn:{isbn}", "count": "5"}
        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/json"
        }
        resp = requests.get(url, headers=headers, params=params, timeout=10)
        if resp.status_code == 200:
            return self._parse_v2_response(resp.json())
        else:
            print(f"[ERROR] Search failed: {resp.status_code} {resp.text}")
            return None

    def search_by_title_author(self, title: str, authors: List[str] = None) -> Optional[Dict]:
        """Search by title and (optionally) author"""
        token = self.get_access_token()
        url = f"{self.api_base_url}/bibs"
        query = f'ti:"{title}"'
        if authors:
            for author in authors:
                if author.strip():
                    query += f' AND au:"{author}"'
        params = {"q": query, "count": "5"}
        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/json"
        }
        resp = requests.get(url, headers=headers, params=params, timeout=10)
        if resp.status_code == 200:
            return self._parse_v2_response(resp.json())
        else:
            print(f"[ERROR] Search failed: {resp.status_code} {resp.text}")
            return None

    def _parse_v2_response(self, data: Dict) -> Optional[Dict]:
        """Extract basic book metadata from v2 JSON response"""
        try:
            if not data.get('entries'):
                return None
            entry = data['entries'][0]  # take the first result

            metadata = {
                'title': entry.get('title', ''),
                'author': ', '.join(entry.get('contributors', [])),
                'publisher': entry.get('publisher', ''),
                'published_date': entry.get('publishedDate', ''),
                'oclc_no': entry.get('oclcNumber', ''),
                'isbn': '; '.join(entry.get('isbn', [])),
                'source': 'worldcat'
            }
            return metadata
        except Exception as e:
            print(f"[ERROR] Failed to parse response: {e}")
            return None

    def search_multiple_isbns(self, isbns: List[str]) -> Dict[str, Optional[Dict]]:
        """Batch search multiple ISBNs (with simple throttling)"""
        results = {}
        for isbn in isbns:
            results[isbn] = self.search_by_isbn(isbn)
            time.sleep(1)  # be nice to the API
        return results
