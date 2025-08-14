import requests
import time
from typing import Optional, List, Dict
from config.config import Config


class WorldCatAPIv2:
    """
    WorldCat Search API v2 client (brief-bibs endpoint).
    Uses OAuth2 Client Credentials grant to authenticate.
    """
    def __init__(self):
        self.config = Config()
        self.client_id = self.config.WORLDCAT_CLIENT_ID
        self.client_secret = self.config.WORLDCAT_CLIENT_SECRET
        self.region = "americas"  # or 'europe', 'apac' depending on location
        self.token_url = "https://oauth.oclc.org/token"
        self.api_base_url = f"https://{self.region}.discovery.api.oclc.org/worldcat/search/v2"
        self.access_token = None
        self.token_expires_at = 0

    def get_access_token(self) -> Optional[str]:
        """Retrieve or refresh OAuth2 access token."""
        if self.access_token and time.time() < self.token_expires_at - 60:
            return self.access_token

        print("[INFO] Requesting new access token...")
        data = {
            'grant_type': 'client_credentials',
            'scope': 'wcapi:view_bib wcapi:view_holdings wcapi:view_my_holdings wcapi:view_retained_holdings'
        }
        try:
            resp = requests.post(self.token_url, data=data, auth=(self.client_id, self.client_secret), timeout=10)
            resp.raise_for_status()
            token_data = resp.json()
            self.access_token = token_data['access_token']
            self.token_expires_at = time.time() + token_data['expires_in']
            print(f"[DEBUG] Token obtained successfully, expires in {token_data['expires_in']} seconds")
            return self.access_token
        except requests.RequestException as e:
            print(f"[ERROR] Token request failed: {e}")
            return None

    def search_by_isbn(self, isbn: str) -> Optional[Dict]:
        """Search for a book using ISBN."""
        token = self.get_access_token()
        if not token:
            return None

        params = {"q": f"isbn:{isbn}"}
        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/json"
        }

        try:
            print(f"[DEBUG] Making request to: {self.api_base_url}/brief-bibs")
            print(f"[DEBUG] Headers: {headers}")
            print(f"[DEBUG] Params: {params}")
            resp = requests.get(f"{self.api_base_url}/brief-bibs", headers=headers, params=params, timeout=10)
            resp.raise_for_status()
            return self._parse_v2_response(resp.json())
        except requests.RequestException as e:
            print(f"[ERROR] ISBN search failed: {e}")
            return None

    def search_by_title_author(self, title: str, authors: Optional[List[str]] = None) -> Optional[Dict]:
        """Search for a book using title and optional authors."""
        token = self.get_access_token()
        if not token:
            return None

        query = f'ti:"{title}"'
        if authors:
            for author in authors:
                if author.strip():
                    query += f' AND au:"{author.strip()}"'

        params = {"q": query, "limit": "1"}
        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/json"
        }

        try:
            resp = requests.get(f"{self.api_base_url}/brief-bibs", headers=headers, params=params, timeout=10)
            resp.raise_for_status()
            return self._parse_v2_response(resp.json())
        except requests.RequestException as e:
            print(f"[ERROR] Title/author search failed: {e}")
            return None

    def _parse_v2_response(self, data: Dict) -> Optional[Dict]:
        """Parse metadata from brief-bibs response."""
        try:
            entries = data.get('briefBib', [])
            if not entries and 'briefBibs' in data:
                entries = data['briefBibs']  # fallback for plural key

            if not entries:
                return None

            entry = entries[0]

            metadata = {
                'title': entry.get('title', ''),
                'author': ', '.join(
                    [c.get('name', '') for c in entry.get('contributors', [])]
                ),
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
        """Batch search multiple ISBNs with delay."""
        results = {}
        for isbn in isbns:
            results[isbn] = self.search_by_isbn(isbn)
            time.sleep(1)
        return results
