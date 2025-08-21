# src/utils/call_number_generator.py
"""
Robust LC-style call number generator for AUC-like practice.

- Tries Gemini (model from env CALL_NUMBER_MODEL or defaults to gemini-1.5-pro)
- Falls back to gemini-1.5-flash if quota/rate limited
- If Gemini isn't available or still rate-limited, returns a provisional LC-style
  number computed offline so your workflow never blocks.

Returns a string call number. When a fallback is used, also sets
metadata['call_number_note'] so the UI can display an informative note.
"""

import os
import time
import re
from typing import Dict, Optional

try:
    import google.generativeai as genai  # pip install google-generativeai
except Exception:
    genai = None  # allow offline fallback when SDK isn't present


# ---------- helpers ----------

def _trim(s: Optional[str], n: int = 160) -> str:
    if not s:
        return ""
    s = re.sub(r"\s+", " ", str(s)).strip()
    return (s[: n - 1] + "…") if len(s) > n else s


def _lc_stub_from_metadata(md: Dict) -> str:
    """
    Offline, *provisional* LC-style call number when Gemini is unavailable.
    This is intentionally simple—good enough to shelve temporarily.
    """
    genre = (md.get("genre") or "").lower()

    cls = "Z"  # default “books/bibliographies”
    if any(k in genre for k in ["history", "civilization"]):
        cls = "D"
    if any(k in genre for k in ["philosophy", "ethics", "logic"]):
        cls = "B"
    if any(k in genre for k in ["religion", "islam", "christian", "jewish"]):
        cls = "BL"
    if any(k in genre for k in ["econom", "finance", "business"]):
        cls = "HB"
    if any(k in genre for k in ["law", "legal"]):
        cls = "K"
    if any(k in genre for k in ["politic", "government", "intl relations"]):
        cls = "J"
    if any(k in genre for k in ["sociol", "anthrop", "social"]):
        cls = "HM"
    if any(k in genre for k in ["psychol"]):
        cls = "BF"
    if any(k in genre for k in ["science", "physics", "chemistry", "math"]):
        cls = "Q"
    if any(k in genre for k in ["computer", "ai", "programming", "data"]):
        cls = "QA"
    if any(k in genre for k in ["literature", "novel", "poetry", "fiction", "mystery", "thriller"]):
        cls = "PN"

    # Cutter-ish code from author family name
    authors = md.get("authors") or []
    family = ""
    if isinstance(authors, list) and authors:
        family = re.sub(r"[^A-Za-z\u0600-\u06FF]+", "", authors[0].split()[-1]).upper()
    elif isinstance(authors, str) and authors:
        family = re.sub(r"[^A-Za-z\u0600-\u06FF]+", "", authors.split()[-1]).upper()
    cutter_letters = (family[:3] or "AAA")

    # second cutter from title’s first significant word
    title = (md.get("title") or "").strip()
    word = re.sub(r"^(a|an|the)\s+", "", title, flags=re.I).split()[0] if title else ""
    word = re.sub(r"[^A-Za-z\u0600-\u06FF]+", "", word).upper()[:3]
    word = word or "TIT"

    year = (md.get("year") or md.get("published_date") or "")
    year = re.findall(r"\d{4}", str(year))
    year = year[0] if year else ""

    # Example: PN .ABC TIT 2023   (very rough but usable)
    return f"{cls} .{cutter_letters} {word} {year}".strip()


def _build_prompt(md: Dict) -> str:
    # keep inputs short to reduce token use → fewer quota hits
    title = _trim(md.get("title"), 120)
    if isinstance(md.get("authors"), list):
        authors = ", ".join(md.get("authors"))
    else:
        authors = _trim(md.get("authors"), 120)
    pub = _trim(md.get("publisher"), 80)
    year = _trim(md.get("year") or md.get("published_date"), 10)
    lang = _trim(md.get("language"), 30)
    genre = _trim(md.get("genre"), 80)
    series = _trim(md.get("series"), 80)
    isbn_combined = _trim(md.get("isbn") or "; ".join(filter(None, [md.get("isbn10", ""), md.get("isbn13", "")])))

    return (
        "You are a cataloger at AUC using Library of Congress Classification (LCC). "
        "Return only the final call number (no notes), in typical AUC style, e.g.: "
        "HB3722 .S65 2021, PN1995.9.F54 B76 2018, DS36.7 .A33 2009.\n\n"
        f"Title: {title}\n"
        f"Authors: {authors}\n"
        f"Publisher: {pub}\n"
        f"Year: {year}\n"
        f"Language: {lang}\n"
        f"Genre/Subject: {genre}\n"
        f"Series: {series}\n"
        f"ISBN(s): {isbn_combined}\n"
    )


def _clean_call_number(s: str) -> str:
    s = s.strip()
    s = s.splitlines()[0]
    s = re.sub(r"^Call\s*Number\s*[:\-]?\s*", "", s, flags=re.I)
    s = re.sub(r"[`\"'<>]", "", s)
    return s


def _try_gemini(prompt: str, model_name: str, max_retries: int = 3) -> Optional[str]:
    if genai is None:
        return None
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return None

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)

    delay = 1.5
    for _ in range(max_retries):
        try:
            resp = model.generate_content(
                prompt,
                generation_config={"temperature": 0.2, "max_output_tokens": 64},
            )
            if hasattr(resp, "text") and resp.text:
                return _clean_call_number(resp.text)
            return None
        except Exception as e:
            msg = str(e).lower()
            if "429" in msg or "quota" in msg or "rate" in msg:
                time.sleep(delay)
                delay *= 1.8
            else:
                time.sleep(min(delay, 2.0))
                delay *= 1.5
    return None


# ---------- public API ----------

def generate_call_number_with_gemini(metadata: Dict) -> Optional[str]:
    """
    Returns a call number string. Also sets metadata['call_number_note']
    when a fallback is used.
    """
    if metadata.get("call_number"):
        return metadata["call_number"]

    prompt = _build_prompt(metadata)

    # 1) preferred model
    primary = os.getenv("CALL_NUMBER_MODEL", "gemini-1.5-pro")
    cn = _try_gemini(prompt, primary)
    if cn:
        return cn

    # 2) fallback model
    cn = _try_gemini(prompt, "gemini-1.5-flash")
    if cn:
        metadata["call_number_note"] = "Generated via Gemini fallback model (flash) due to quota on pro."
        return cn

    # 3) offline provisional
    provisional = _lc_stub_from_metadata(metadata)
    metadata["call_number_note"] = "Provisional LC call number generated offline (Gemini quota/rate limit)."
    return provisional
