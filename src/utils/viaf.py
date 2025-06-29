"""
VIAF API utility functions for author standardization
"""
import requests

VIAF_SEARCH_URL = 'https://viaf.org/viaf/search'

def search_author_viaf(author_name):
    """Search VIAF for an author name using the /viaf/search endpoint with httpAccept=application/json."""
    cql_query = f'local.names all "{author_name}"'
    params = {
        'query': cql_query,
        'httpAccept': 'application/json'
    }
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        response = requests.get(VIAF_SEARCH_URL, params=params, headers=headers, timeout=10)
        content_type = response.headers.get('Content-Type', '')
        if response.status_code == 200 and 'json' in content_type:
            return response.json()
        else:
            print(f"VIAF error: status {response.status_code}, content-type: {content_type}, content: {response.text[:200]}")
            return {}
    except Exception as e:
        print(f"VIAF request failed: {e}")
        return {} 