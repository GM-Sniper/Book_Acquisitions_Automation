import os
import json
from config.config import Config
from google import genai

def llm_metadata_combiner(gemini_data, google_books_data, openlibrary_data, loc_data, isbnlib_data, debug=False):
    """
    Use Gemini LLM to merge all metadata fields from all sources.
    Args:
        gemini_data (dict): Metadata from Gemini Vision
        google_books_data (dict): Metadata from Google Books
        openlibrary_data (dict): Metadata from OpenLibrary
        loc_data (dict): Metadata from LOC (e.g., LCCN)
        isbnlib_data (dict): Metadata from isbnlib
        debug (bool): If True, also return provenance for each field
    Returns:
        dict: Merged metadata (and provenance if debug=True)
    """
    if not Config.GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY not found in environment variables.")

    client = genai.Client(api_key=Config.GEMINI_API_KEY)

    # Compose the prompt for Gemini
    prompt = f'''
You are a book metadata expert. You are given metadata for the same book from multiple sources. Your job is to merge them into the most accurate, complete, and consistent record possible.

Here are the metadata dicts from each source (in JSON):
- gemini: {json.dumps(gemini_data, ensure_ascii=False, indent=2)}
- google_books: {json.dumps(google_books_data, ensure_ascii=False, indent=2)}
- openlibrary: {json.dumps(openlibrary_data, ensure_ascii=False, indent=2)}
- loc: {json.dumps(loc_data, ensure_ascii=False, indent=2)}
- isbnlib: {json.dumps(isbnlib_data, ensure_ascii=False, indent=2)}

Rules:
- Always use the ISBN(s) from the gemini source as the primary ISBN(s) for the final output.
- For other fields (title, author, publisher, etc.), prefer the value that is most complete, accurate, and consistent across sources.
- If sources disagree, prefer the value that matches the gemini source, or the value that is most complete.
- If a field is missing in all sources, leave it blank or null.
- For the LCCN, use the value from the loc source if available.
- Return the final merged metadata as a JSON object with keys: title, authors, publisher, published_date, edition, series, genre, language, isbn, isbn10, isbn13, lccn, oclc_no, additional_text.
- If debug mode is enabled, also return a provenance object mapping each field to the source(s) used.

Respond ONLY with a JSON object with the following structure:
{{
  "merged_metadata": {{ ... }},
  "provenance": {{ ... }}  // Only include this if debug mode is enabled
}}
'''
    if not debug:
        prompt += "\nDo NOT include the provenance object if debug mode is not enabled."

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        # Extract the JSON from the response
        result = None
        try:
            result = json.loads(response.text)
        except Exception:
            # Try to extract the first JSON object
            import re
            match = re.search(r'\{.*\}', response.text, re.DOTALL)
            if match:
                result = json.loads(match.group(0))
        if not result:
            raise ValueError("Gemini LLM did not return valid JSON.")
        if debug:
            return result.get("merged_metadata", {}), result.get("provenance", {})
        else:
            return result.get("merged_metadata", {})
    except Exception as e:
        print(f"[LLM Combiner] Gemini LLM failed: {e}")
        # Fallback: simple rule-based merge
        merged = {}
        # Always use Gemini ISBNs
        merged['isbn'] = gemini_data.get('isbn') or gemini_data.get('isbn13') or gemini_data.get('isbn10')
        merged['isbn10'] = gemini_data.get('isbn10') or google_books_data.get('isbn_10') or openlibrary_data.get('isbn_10') or isbnlib_data.get('isbn_10')
        merged['isbn13'] = gemini_data.get('isbn13') or google_books_data.get('isbn_13') or openlibrary_data.get('isbn_13') or isbnlib_data.get('isbn_13')
        merged['title'] = gemini_data.get('title') or google_books_data.get('title') or openlibrary_data.get('title') or isbnlib_data.get('title')
        merged['authors'] = gemini_data.get('authors') or google_books_data.get('author') or openlibrary_data.get('author') or isbnlib_data.get('author')
        merged['publisher'] = gemini_data.get('publisher') or google_books_data.get('publisher') or openlibrary_data.get('publisher') or isbnlib_data.get('publisher')
        merged['published_date'] = gemini_data.get('year') or google_books_data.get('published_date') or openlibrary_data.get('published_date') or isbnlib_data.get('year')
        merged['edition'] = gemini_data.get('edition') or google_books_data.get('edition') or openlibrary_data.get('edition') or isbnlib_data.get('edition')
        merged['series'] = gemini_data.get('series') or google_books_data.get('series') or openlibrary_data.get('series') or isbnlib_data.get('series')
        merged['genre'] = gemini_data.get('genre') or google_books_data.get('genre') or openlibrary_data.get('genre') or isbnlib_data.get('genre')
        merged['language'] = gemini_data.get('language') or google_books_data.get('language') or openlibrary_data.get('language') or isbnlib_data.get('language')
        merged['lccn'] = loc_data.get('lccn') if loc_data else ''
        merged['oclc_no'] = openlibrary_data.get('oclc_no') or ''
        merged['additional_text'] = gemini_data.get('additional_text') or ''
        if debug:
            provenance = {k: 'fallback' for k in merged.keys()}
            return merged, provenance
        else:
            return merged 