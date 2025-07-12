import streamlit as st
import sys
import os
import cv2
import time
import tempfile
import difflib

# Add the project root folder to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.vision.gemini_processing import process_book_images
from src.vision.preprocessing import preprocess_image
from src.metadata.metadata_extraction import metadata_combiner
from src.utils.isbn_detection import extract_isbns, extract_and_validate_isbns
from src.utils.google_books import search_book_by_isbn, search_book_by_title_author, extract_book_metadata
from src.utils.openlibrary import OpenLibraryAPI
from src.utils.LOC import LOCConverter

# Helper function to unify metadata from Google Books and OpenLibrary

def fuzzy_match(a, b):
    """Return a similarity ratio between two strings (0-1) using difflib."""
    return difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()

def get_unified_metadata(title, authors, isbns, lccns=None, edition=None):
    """
    Query Google Books and OpenLibrary and return unified metadata fields.
    lccns: optional, a list or string of LCCNs to use for the 'LC no.' field.
    edition: optional, edition information from Gemini metadata.
    Returns a dict with keys: TITLE, AUTHOR, PUBLISHED, D.O Pub., OCLC no., LC no., ISBN
    """
    gb_data = None
    ol_data = None
    api = OpenLibraryAPI()
    # Prefer ISBN search if available
    found_by_isbn = False
    if isbns:
        for isbn in isbns:
            # Google Books
            try:
                gb_result = search_book_by_isbn(isbn)
                if gb_result and gb_result.get('items'):
                    gb_data = extract_book_metadata(gb_result)
                    # Only use if ISBN matches exactly
                    if gb_data and gb_data.get('isbn') and isbn in gb_data['isbn']:
                        found_by_isbn = True
                        break
                    else:
                        gb_data = None
            except Exception as e:
                pass
            # OpenLibrary
            try:
                ol_data = api.search_by_isbn(isbn)
                if ol_data and ol_data.get('isbn') and isbn in ol_data['isbn']:
                    found_by_isbn = True
                    break
                else:
                    ol_data = None
            except Exception as e:
                pass
    # Fallback: search by title/author if no ISBN match
    found_by_fallback = False
    fallback_gb = None
    fallback_ol = None
    if not found_by_isbn and title:
        try:
            gb_result = search_book_by_title_author(title, authors, edition)
            if gb_result and gb_result.get('items'):
                candidate = extract_book_metadata(gb_result)
                # Fuzzy match title and author
                title_ratio = fuzzy_match(candidate.get('title', ''), title)
                author_ratio = fuzzy_match(candidate.get('author', ''), ', '.join(authors) if authors else '')
                if title_ratio >= 0.9 and author_ratio >= 0.9:
                    fallback_gb = candidate
                    found_by_fallback = True
        except Exception as e:
            pass
    if not found_by_isbn and not found_by_fallback and title:
        try:
            ol_candidate = api.search_by_title_author(title, authors, edition)
            if ol_candidate:
                title_ratio = fuzzy_match(ol_candidate.get('title', ''), title)
                author_ratio = fuzzy_match(ol_candidate.get('author', ''), ', '.join(authors) if authors else '')
                if title_ratio >= 0.9 and author_ratio >= 0.9:
                    fallback_ol = ol_candidate
                    found_by_fallback = True
        except Exception as e:
            pass
    # Compose unified result
    if isinstance(lccns, list):
        lccn_str = '; '.join([l for l in lccns if l])
    elif isinstance(lccns, str):
        lccn_str = lccns
    else:
        lccn_str = ''
    # Decide which data to use
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

# Utility function to save preprocessed image
def save_preprocessed_image(image_np, original_filename, output_dir='data/processed'):
    os.makedirs(output_dir, exist_ok=True)
    base, ext = os.path.splitext(original_filename)
    out_path = os.path.join(output_dir, f"{base}_processed.png")
    cv2.imwrite(out_path, image_np)
    return out_path

# Function to process a single image with Gemini
def process_image_with_gemini(image_input, image_name, image_type="cover"):
    """Process a single image through Gemini Vision"""
    st.header(f'{image_type.title()} Processing')
    
    # Display original image
    col1, col2 = st.columns(2)
    with col1:
        st.subheader('Original Image')
        st.image(image_input, caption=f'{image_type.title()} (Original)')
    
    # Preprocess and display
    with st.spinner(f'Preprocessing {image_type} with CLAHE enhancement...'):
        # Read image content
        image_content = image_input.read()
        preprocessed_image = preprocess_image(image_content)
    
    with col2:
        st.subheader('Preprocessed Image (CLAHE Enhanced)')
        st.image(preprocessed_image, caption=f'{image_type.title()} (Preprocessed)', channels='GRAY')
    
    # Save preprocessed image
    saved_path = save_preprocessed_image(preprocessed_image, f"{image_name}_{image_type}_{int(time.time())}.jpg")
    st.success(f'✅ Preprocessed {image_type} saved to: `{saved_path}`')
    
    # Process with Gemini
    with st.spinner(f'Extracting metadata with Gemini Vision...'):
        try:
            # Convert preprocessed image to bytes for Gemini
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                cv2.imwrite(tmp_file.name, preprocessed_image)
                tmp_file_path = tmp_file.name
            
            # Process with Gemini
            metadata = process_book_images([tmp_file_path], prompt_type="comprehensive")
            
            # Clean up temporary file
            os.unlink(tmp_file_path)
            
            return {
                'metadata': metadata,
                'preprocessed_path': saved_path
            }
            
        except Exception as e:
            st.error(f"Error processing with Gemini: {e}")
            return {
                'metadata': None,
                'preprocessed_path': saved_path
            }

# Title of the app
st.title('Enhanced Book Cover Processing with Gemini Vision')

# Sidebar for configuration
st.sidebar.header("Configuration")
use_camera = st.sidebar.checkbox("Use Camera Capture (instead of upload)", value=False)
use_preprocessing = st.sidebar.checkbox("Use Image Preprocessing", value=True)
use_online_validation = st.sidebar.checkbox("Use Online ISBN Validation", value=True)
show_processing_details = st.sidebar.checkbox("Show Processing Details", value=True)

# --- Section: Image Input ---
if use_camera:
    st.header('Step 1: Camera Capture')
    st.write('Use your camera to capture the **front cover** and **back cover** images of the book.')
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Front Cover Capture")
        front_input = st.camera_input("Take a picture of the front cover", key="front_camera")
    
    with col2:
        st.subheader("Back Cover Capture")
        back_input = st.camera_input("Take a picture of the back cover", key="back_camera")

else:
    st.header('Step 1: Upload Book Cover Images')
    st.write('Please upload the **front cover** and **back cover** images of the book.')
    
    col1, col2 = st.columns(2)
    
    with col1:
        front_input = st.file_uploader('Front Cover Image', type=['jpg', 'jpeg', 'png'], key='front')
    
    with col2:
        back_input = st.file_uploader('Back Cover Image', type=['jpg', 'jpeg', 'png'], key='back')

# Check if both images are ready for processing
both_images_ready = front_input is not None and back_input is not None

if front_input and not back_input:
    st.info("📷 Front cover ready! Please capture/upload the back cover to continue.")
elif back_input and not front_input:
    st.info("📷 Back cover ready! Please capture/upload the front cover to continue.")
elif not front_input and not back_input:
    st.info("📷 Please capture/upload both front and back cover images to begin processing.")

# Only process if both images are ready
if both_images_ready:
    st.success("✅ Both images ready! Starting processing...")
    
    # Preprocess both covers
    with st.spinner('Preprocessing front and back covers with CLAHE enhancement...'):
        front_content = front_input.read()
        back_content = back_input.read()
        front_preprocessed = preprocess_image(front_content)
        back_preprocessed = preprocess_image(back_content)
    
    # Save preprocessed images
    front_saved_path = save_preprocessed_image(front_preprocessed, f"book_front_{int(time.time())}.jpg")
    back_saved_path = save_preprocessed_image(back_preprocessed, f"book_back_{int(time.time())}.jpg")
    st.success(f'✅ Preprocessed front cover saved to: `{front_saved_path}`')
    st.success(f'✅ Preprocessed back cover saved to: `{back_saved_path}`')
    
    # Process both covers together with Gemini
    with st.spinner('Extracting metadata with Gemini Vision (both covers)...'):
        import tempfile
        import cv2
        # Save both preprocessed images as temp files for Gemini
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_front:
            cv2.imwrite(tmp_front.name, front_preprocessed)
            tmp_front_path = tmp_front.name
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_back:
            cv2.imwrite(tmp_back.name, back_preprocessed)
            tmp_back_path = tmp_back.name
        try:
            combined_metadata = process_book_images([tmp_front_path, tmp_back_path], prompt_type="comprehensive")
        finally:
            os.unlink(tmp_front_path)
            os.unlink(tmp_back_path)
    
    # Display combined results
    if combined_metadata:
        st.header('📚 Combined Book Metadata (Gemini, both covers)')
        st.json(combined_metadata)
        
        # Extract ISBNs for further processing
        isbns = []
        if combined_metadata.get('isbn'):
            if isinstance(combined_metadata['isbn'], list):
                isbns = combined_metadata['isbn']
            elif isinstance(combined_metadata['isbn'], str):
                isbns = [combined_metadata['isbn']]
        
        # Get LOC LCCN data
        loc_results = {}
        if isbns:
            with st.spinner('Querying Library of Congress for LCCN numbers...'):
                loc_converter = LOCConverter()
                loc_results = loc_converter.get_lccn_for_isbns(isbns)
        
        # Get all found LCCNs as a list
        lccn_list = [lccn for lccn in loc_results.values() if lccn] if loc_results else []
        
        # Get unified metadata from external sources
        unified = get_unified_metadata(
            combined_metadata.get('title', ''),
            combined_metadata.get('authors', []),
            isbns,
            lccns=lccn_list,
            edition=combined_metadata.get('edition')
        )
        
        st.header('📚 Unified Book Metadata (Gemini + Google Books + OpenLibrary + LOC)')
        st.table([unified])
    else:
        st.warning('❌ No metadata could be extracted from both covers.')
    
    # Performance metrics
    st.header('📈 Performance Metrics')
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Front Cover Preprocessed", "✅ Done")
    with col2:
        st.metric("Back Cover Preprocessed", "✅ Done")

