import streamlit as st
import sys
import os
import cv2
import time
import tempfile

# Add the project root folder to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.vision.preprocessing import preprocess_image
from src.vision.OCR_Processing import extract_text_from_image, extract_text_with_confidence
from src.metadata.metadata_extraction import extract_metadata_with_gemini, metadata_combiner
from src.utils.isbn_detection import extract_isbns, extract_and_validate_isbns
from src.utils.google_books import search_book_by_isbn, search_book_by_title_author, extract_book_metadata
from src.utils.openlibrary import OpenLibraryAPI

# Helper function to unify metadata from Google Books and OpenLibrary

def get_unified_metadata(title, authors, isbns):
    """
    Query Google Books and OpenLibrary and return unified metadata fields.
    Returns a dict with keys: TITLE, AUTHOR, PUBLISHED, D.O Pub., OCLC no., LC no., ISBN
    """
    gb_data = None
    ol_data = None
    api = OpenLibraryAPI()
    # Prefer ISBN search if available
    if isbns:
        for isbn in isbns:
            # Google Books
            try:
                gb_result = search_book_by_isbn(isbn)
                if gb_result and gb_result.get('items'):
                    gb_data = extract_book_metadata(gb_result)
                    break
            except Exception as e:
                pass
            # OpenLibrary
            try:
                ol_data = api.search_by_isbn(isbn)
                if ol_data:
                    break
            except Exception as e:
                pass
    # If no ISBN or not found, use title/author
    if not gb_data and title:
        try:
            gb_result = search_book_by_title_author(title, authors)
            if gb_result and gb_result.get('items'):
                gb_data = extract_book_metadata(gb_result)
        except Exception as e:
            pass
    if not ol_data and title:
        try:
            ol_data = api.search_by_title_author(title, authors)
        except Exception as e:
            pass
    # Compose unified result
    unified = {
        'TITLE': gb_data['title'] if gb_data and gb_data.get('title') else (ol_data['title'] if ol_data else title),
        'AUTHOR': gb_data['author'] if gb_data and gb_data.get('author') else (ol_data['author'] if ol_data else ', '.join(authors) if authors else ''),
        'PUBLISHED': gb_data['publisher'] if gb_data and gb_data.get('publisher') else (ol_data['publisher'] if ol_data else ''),
        'D.O Pub.': gb_data['published_date'] if gb_data and gb_data.get('published_date') else (ol_data['published_date'] if ol_data else ''),
        'OCLC no.': ol_data['oclc_no'] if ol_data and 'oclc_no' in ol_data else '',
        'LC no.': ol_data['lc_no'] if ol_data and 'lc_no' in ol_data else '',
        'ISBN': gb_data['isbn'] if gb_data and gb_data.get('isbn') else (ol_data['isbn'] if ol_data else '; '.join(isbns) if isbns else ''),
    }
    return unified

# Utility function to save preprocessed image
def save_preprocessed_image(image_np, original_filename, output_dir='data/processed'):
    os.makedirs(output_dir, exist_ok=True)
    base, ext = os.path.splitext(original_filename)
    out_path = os.path.join(output_dir, f"{base}_processed.png")
    cv2.imwrite(out_path, image_np)
    return out_path

# Function to process a single image (works for both camera and file uploads)
def process_image(image_input, image_name, image_type="cover"):
    """Process a single image through the entire pipeline"""
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
    
    # OCR with enhanced features
    with st.spinner('Extracting text with confidence scoring...'):
        if use_enhanced_ocr:
            # Use enhanced OCR with confidence
            # Save preprocessed image temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                cv2.imwrite(tmp_file.name, preprocessed_image)  # Use preprocessed image
                tmp_file_path = tmp_file.name
            
            try:
                ocr_result = extract_text_with_confidence(tmp_file_path)
                extracted_text = ocr_result['text']
                confidence = ocr_result['confidence']
                word_count = ocr_result.get('word_count', 0)
            finally:
                # Clean up temporary file
                os.unlink(tmp_file_path)
        else:
            # Use basic OCR
            extracted_text = extract_text_from_image(preprocessed_image)
            confidence = None
            word_count = len(extracted_text.split()) if extracted_text else 0
    
    # Display text extraction results
    st.subheader(f'📝 Extracted Text ({image_type.title()})')
    st.text_area(f'{image_type.title()} OCR Text', extracted_text, height=150)
    
    if use_enhanced_ocr and confidence is not None:
        # Create confidence indicator
        if confidence >= 0.8:
            confidence_color = "🟢"
            confidence_status = "Excellent"
        elif confidence >= 0.6:
            confidence_color = "🟡"
            confidence_status = "Good"
        elif confidence >= 0.4:
            confidence_color = "🟠"
            confidence_status = "Fair"
        else:
            confidence_color = "🔴"
            confidence_status = "Poor"
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Confidence Score", f"{confidence:.2f}", f"{confidence_color} {confidence_status}")
        with col2:
            st.metric("Word Count", word_count)
        with col3:
            st.metric("Text Quality", confidence_status)
    
    return {
        'text': extracted_text,
        'confidence': confidence,
        'word_count': word_count,
        'preprocessed_path': saved_path
    }

# Title of the app
st.title('Enhanced Book Cover OCR System')

# Sidebar for configuration
st.sidebar.header("Configuration")
use_camera = st.sidebar.checkbox("Use Camera Capture (instead of upload)", value=False)
use_enhanced_ocr = st.sidebar.checkbox("Use Enhanced OCR (Confidence Scoring)", value=True)
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
    
    # Process front cover
    front_result = process_image(front_input, "book", "front")
    
    # Process back cover
    back_result = process_image(back_input, "book", "back")
    
    # Extract metadata from front cover
    metadata = None
    if front_result['text']:
        with st.spinner('Extracting metadata with Gemini AI...'):
            metadata = extract_metadata_with_gemini(front_result['text'])
        
        if metadata:
            st.subheader('📋 Extracted Metadata (Front Cover)')
            col1, col2 = st.columns(2)
            with col1:
                if metadata.get('title'):
                    st.success(f"**Title:** {metadata['title']}")
                else:
                    st.warning("**Title:** Not found")
            with col2:
                if metadata.get('authors'):
                    st.success(f"**Authors:** {', '.join(metadata['authors'])}")
                else:
                    st.warning("**Authors:** Not found")
            
            if show_processing_details:
                st.json(metadata)
        else:
            st.warning('❌ No metadata (title/author) found in front cover text.')
    
    # Extract ISBNs from back cover
    isbns = None
    if back_result['text']:
        with st.spinner('Extracting and validating ISBNs...'):
            if use_online_validation:
                # Use enhanced ISBN detection with online validation
                isbn_results = extract_and_validate_isbns(back_result['text'])
                isbns = isbn_results
            else:
                # Use basic ISBN detection
                isbns = extract_isbns(back_result['text'])
        
        if isbns:
            if use_online_validation:
                st.subheader('📚 Extracted ISBNs with Online Validation (Back Cover)')
                
                if isbn_results['total_found'] > 0:
                    st.success(f"Found {isbn_results['total_found']} ISBN(s), {isbn_results['validated_count']} validated online")
                    
                    for i, isbn_data in enumerate(isbn_results['isbns']):
                        with st.expander(f"ISBN {i+1}: {isbn_data['isbn']}"):
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write(f"**Original:** {isbn_data['original']}")
                                st.write(f"**Type:** {isbn_data['type']}")
                            with col2:
                                if isbn_data['validation']['valid']:
                                    st.success("✅ **Validated Online**")
                                    st.write(f"**Title:** {isbn_data['validation']['title']}")
                                    st.write(f"**Authors:** {', '.join(isbn_data['validation']['authors'])}")
                                    st.write(f"**Publisher:** {isbn_data['validation']['publisher']}")
                                else:
                                    st.error("❌ **Validation Failed**")
                                    st.write(f"**Error:** {isbn_data['validation']['error']}")
                else:
                    st.warning('❌ No ISBNs found in back cover text.')
            else:
                st.subheader('📚 Extracted ISBNs (Back Cover)')
                if isbns and (isbns['isbn10'] or isbns['isbn13']):
                    st.json(isbns)
                else:
                    st.warning('❌ No ISBNs found in back cover text.')
        else:
            st.warning('❌ No ISBNs found in back cover text.')

    # Combine and display final result if both are available
    if metadata:
        # Prepare ISBN list for combination
        isbn_list = []
        if isbns:
            if use_online_validation and isinstance(isbns, dict) and 'isbns' in isbns:
                isbn_list = [isbn_data['isbn'] for isbn_data in isbns['isbns']]
            elif isinstance(isbns, dict):
                if isbns.get('isbn10'):
                    isbn_list.extend(isbns['isbn10'])
                if isbns.get('isbn13'):
                    isbn_list.extend(isbns['isbn13'])
            elif isinstance(isbns, list):
                isbn_list = isbns
        # Use extracted metadata fields
        title = metadata.get('title', '')
        authors = metadata.get('authors', [])
        unified = get_unified_metadata(title, authors, isbn_list)
        st.header('📚 Unified Book Metadata (Google Books + OpenLibrary)')
        st.table([unified])

    # Performance metrics
    st.header('📈 Performance Metrics')
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if front_result['confidence']:
            st.metric("Front Cover Quality", f"{front_result['confidence']:.2f}")
        if back_result['confidence']:
            st.metric("Back Cover Quality", f"{back_result['confidence']:.2f}")
    
    with col2:
        st.metric("Front Words", front_result['word_count'])
        st.metric("Back Words", back_result['word_count'])
    
    with col3:
        if use_online_validation and isinstance(isbns, dict):
            st.metric("ISBNs Found", isbns.get('total_found', 0))
            st.metric("ISBNs Validated", isbns.get('validated_count', 0))

