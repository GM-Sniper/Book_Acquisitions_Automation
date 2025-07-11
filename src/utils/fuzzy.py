import difflib

def fuzzy_match(a: str, b: str) -> float:
    """
    Return a similarity ratio between two strings (0-1) using difflib.
    Args:
        a (str): First string
        b (str): Second string
    Returns:
        float: Similarity ratio (0.0 to 1.0)
    """
    return difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()

# Example usage
if __name__ == "__main__":
    s1 = "The Great Gatsby"
    s2 = "the great gatsby"
    print(f"Similarity: {fuzzy_match(s1, s2):.2f}") 