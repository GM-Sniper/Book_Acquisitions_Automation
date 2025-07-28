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
from src.utils.isbnlib_service import ISBNService
from src.utils.LOC import LOCConverter
from src.metadata.llm_metadata_combiner import llm_metadata_combiner

# Helper function to unify metadata from Google Books and OpenLibrary

def fuzzy_match(a, b):
    """Return a similarity ratio between two strings (0-1) using difflib."""
    return difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()

def extract_all_isbns(metadata):
    """
    Extract all unique ISBNs (10 and 13) from a metadata dict.
    Handles fields: isbn, isbn10, isbn_10, isbn13, isbn_13.
    """
    isbn_fields = ['isbn', 'isbn10', 'isbn_10', 'isbn13', 'isbn_13']
    isbns = set()
    for field in isbn_fields:
        value = metadata.get(field)
        if not value:
            continue
        if isinstance(value, list):
            for v in value:
                if v and isinstance(v, str):
                    isbns.add(v.strip())
        elif isinstance(value, str):
            for v in value.replace(';', ',').split(','):
                v = v.strip()
                if v:
                    isbns.add(v)
    return [isbn for isbn in isbns if isbn]

def get_unified_metadata(title, authors, isbns, lccns=None, edition=None, gemini_data=None, google_books_data=None, openlibrary_data=None, loc_data=None, isbnlib_data=None, debug=False):
    """
    Enhanced: Use Gemini's ISBN(s) for all lookups, and merge all metadata using the LLM combiner.
    Args:
        title, authors, isbns, lccns, edition: as before
        gemini_data, google_books_data, openlibrary_data, loc_data, isbnlib_data: dicts from each source
        debug: if True, return provenance for each field
    Returns:
        dict: merged metadata (and provenance if debug=True)
    """
    # Always use Gemini ISBNs for all lookups and as final output
    primary_isbns = []
    if gemini_data:
        if isinstance(gemini_data.get('isbn'), list):
            primary_isbns = gemini_data['isbn']
        elif gemini_data.get('isbn'):
            primary_isbns = [gemini_data['isbn']]
        elif gemini_data.get('isbn13'):
            primary_isbns = [gemini_data['isbn13']]
        elif gemini_data.get('isbn10'):
            primary_isbns = [gemini_data['isbn10']]
    if not primary_isbns:
        primary_isbns = isbns or []

    # Call the LLM combiner
    if debug:
        merged, provenance = llm_metadata_combiner(
            gemini_data or {},
            google_books_data or {},
            openlibrary_data or {},
            loc_data or {},
            isbnlib_data or {},
            debug=True
        )
        return merged, provenance
    else:
        merged = llm_metadata_combiner(
            gemini_data or {},
            google_books_data or {},
            openlibrary_data or {},
            loc_data or {},
            isbnlib_data or {},
            debug=False
        )
        return merged

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
        isbns = extract_all_isbns(combined_metadata)
        print(f"[DEBUG] ISBNs sent to LOC: {isbns}")

        # Get LOC LCCN data
        loc_results_raw = {}
        loc_results = {}
        lccn_list = []
        if isbns:
            with st.spinner('Querying Library of Congress for LCCN numbers...'):
                loc_converter = LOCConverter()
                loc_results_raw = loc_converter.get_lccn_for_isbns(isbns)
                # Map first non-null LCCN to 'lccn' key for the combiner
                lccn_value = next((lccn for lccn in loc_results_raw.values() if lccn), None)
                loc_results = {'lccn': lccn_value} if lccn_value else {}
                lccn_list = [lccn for lccn in loc_results_raw.values() if lccn] if loc_results_raw else []

        # Fetch Google Books metadata
        gb_data = None
        for isbn in isbns:
            try:
                gb_result = search_book_by_isbn(isbn)
                if gb_result and gb_result.get('items'):
                    gb_data = extract_book_metadata(gb_result)
                    break
            except Exception as e:
                print(f"Google Books error: {e}")

        # Fetch OpenLibrary metadata
        from src.utils.openlibrary import OpenLibraryAPI
        ol_api = OpenLibraryAPI()
        ol_data = None
        for isbn in isbns:
            try:
                ol_data = ol_api.search_by_isbn(isbn)
                if ol_data and ol_data.get('title'):
                    break
            except Exception as e:
                print(f"OpenLibrary error: {e}")

        # Fetch isbnlib metadata
        api = ISBNService(debug=True)
        isbnlib_data = None
        for isbn in isbns:
            try:
                isbnlib_data = api.search_by_isbn(isbn)
                if isbnlib_data and isbnlib_data.get('title'):
                    break
            except Exception as e:
                print(f"isbnlib error: {e}")

        # Get unified metadata from external sources
        unified, provenance = get_unified_metadata(
            combined_metadata.get('title', ''),
            combined_metadata.get('authors', []),
            isbns,
            lccns=lccn_list,
            edition=combined_metadata.get('edition'),
            gemini_data=combined_metadata,
            google_books_data=gb_data,
            openlibrary_data=ol_data,
            loc_data=loc_results,
            isbnlib_data=isbnlib_data,
            debug=show_processing_details
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

