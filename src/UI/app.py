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

# Utility function to save preprocessed image
def save_preprocessed_image(image_np, original_filename, output_dir='data/processed'):
    os.makedirs(output_dir, exist_ok=True)
    base, ext = os.path.splitext(original_filename)
    out_path = os.path.join(output_dir, f"{base}_processed.png")
    cv2.imwrite(out_path, image_np)
    return out_path

# Title of the app
st.title('Enhanced Book Cover OCR System')

# Sidebar for configuration
st.sidebar.header("Configuration")
use_enhanced_ocr = st.sidebar.checkbox("Use Enhanced OCR (Confidence Scoring)", value=True)
use_online_validation = st.sidebar.checkbox("Use Online ISBN Validation", value=True)
show_processing_details = st.sidebar.checkbox("Show Processing Details", value=True)

# --- Section: Upload image(s) ---
st.header('Step 1: Upload Book Cover Images')
st.write('Please upload the **front cover** and **back cover** images of the book.')

col1, col2 = st.columns(2)

with col1:
    front_file = st.file_uploader('Front Cover Image', type=['jpg', 'jpeg', 'png'], key='front')

with col2:
    back_file = st.file_uploader('Back Cover Image', type=['jpg', 'jpeg', 'png'], key='back')

# Initialize variables
front_text = None
back_text = None
front_confidence = None
back_confidence = None
metadata = None
isbns = None
combined_metadata = None

# Check if both images are ready for processing
both_images_ready = False

# Check if both files are uploaded
front_ready = front_file is not None
back_ready = back_file is not None
both_images_ready = front_ready and back_ready

if front_ready and not back_ready:
    st.info("📁 Front cover uploaded! Please upload the back cover to continue.")
elif back_ready and not front_ready:
    st.info("📁 Back cover uploaded! Please upload the front cover to continue.")
elif not front_ready and not back_ready:
    st.info("📁 Please upload both front and back cover images to begin processing.")

# Only process if both images are ready
if both_images_ready:
    st.success("✅ Both images ready! Starting processing...")
    
    # Process front cover
    st.header(' Front Cover Processing')
    
    # Display original image
    col1, col2 = st.columns(2)
    with col1:
        st.subheader('Original Image')
        st.image(front_file, caption='Front Cover (Original)')
    
    # Preprocess and display
    with st.spinner('Preprocessing front cover with CLAHE enhancement...'):
        # Read file content once
        front_file_content = front_file.read()
        preprocessed_front = preprocess_image(front_file_content)
    
    with col2:
        st.subheader('Preprocessed Image (CLAHE Enhanced)')
        st.image(preprocessed_front, caption='Front Cover (Preprocessed)', channels='GRAY')
    
    # Save preprocessed image
    saved_path = save_preprocessed_image(preprocessed_front, front_file.name)
    st.success(f'✅ Preprocessed front cover saved to: `{saved_path}`')
    
    # OCR with enhanced features
    with st.spinner('Extracting text with confidence scoring...'):
        if use_enhanced_ocr:
            # Use enhanced OCR with confidence
            # For uploaded files, we need to save them temporarily first
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                tmp_file.write(front_file_content)  # Use the content we already read
                tmp_file_path = tmp_file.name
            
            try:
                front_result = extract_text_with_confidence(tmp_file_path)
                front_text = front_result['text']
                front_confidence = front_result['confidence']
                front_word_count = front_result.get('word_count', 0)
            finally:
                # Clean up temporary file
                os.unlink(tmp_file_path)
        else:
            # Use basic OCR
            front_text = extract_text_from_image(preprocessed_front)
            front_confidence = None
            front_word_count = len(front_text.split()) if front_text else 0
    
    # Display text extraction results
    st.subheader('📝 Extracted Text (Front Cover)')
    st.text_area('Front Cover OCR Text', front_text, height=150)
    
    if use_enhanced_ocr and front_confidence is not None:
        # Create confidence indicator
        if front_confidence >= 0.8:
            confidence_color = "🟢"
            confidence_status = "Excellent"
        elif front_confidence >= 0.6:
            confidence_color = "🟡"
            confidence_status = "Good"
        elif front_confidence >= 0.4:
            confidence_color = "🟠"
            confidence_status = "Fair"
        else:
            confidence_color = "🔴"
            confidence_status = "Poor"
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Confidence Score", f"{front_confidence:.2f}", f"{confidence_color} {confidence_status}")
        with col2:
            st.metric("Word Count", front_word_count)
        with col3:
            st.metric("Text Quality", confidence_status)
    
    # Metadata extraction
    if front_text:
        with st.spinner('Extracting metadata with Gemini AI...'):
            metadata = extract_metadata_with_gemini(front_text)
        
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

    # Process back cover
    st.header('🖼️ Back Cover Processing')
    
    # Display original image
    col1, col2 = st.columns(2)
    with col1:
        st.subheader('Original Image')
        st.image(back_file, caption='Back Cover (Original)')
    
    # Preprocess and display
    with st.spinner('Preprocessing back cover with CLAHE enhancement...'):
        # Read file content once
        back_file_content = back_file.read()
        preprocessed_back = preprocess_image(back_file_content)
    
    with col2:
        st.subheader('Preprocessed Image (CLAHE Enhanced)')
        st.image(preprocessed_back, caption='Back Cover (Preprocessed)', channels='GRAY')
    
    # Save preprocessed image
    saved_path = save_preprocessed_image(preprocessed_back, back_file.name)
    st.success(f'✅ Preprocessed back cover saved to: `{saved_path}`')
    
    # OCR with enhanced features
    with st.spinner('Extracting text with confidence scoring...'):
        if use_enhanced_ocr:
            # Use enhanced OCR with confidence
            # For uploaded files, we need to save them temporarily first
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                tmp_file.write(back_file_content)  # Use the content we already read
                tmp_file_path = tmp_file.name
            
            try:
                back_result = extract_text_with_confidence(tmp_file_path)
                back_text = back_result['text']
                back_confidence = back_result['confidence']
                back_word_count = back_result.get('word_count', 0)
            finally:
                # Clean up temporary file
                os.unlink(tmp_file_path)
        else:
            # Use basic OCR
            back_text = extract_text_from_image(preprocessed_back)
            back_confidence = None
            back_word_count = len(back_text.split()) if back_text else 0
    
    # Display text extraction results
    st.subheader('📝 Extracted Text (Back Cover)')
    st.text_area('Back Cover OCR Text', back_text, height=150)
    
    if use_enhanced_ocr and back_confidence is not None:
        # Create confidence indicator
        if back_confidence >= 0.8:
            confidence_color = "🟢"
            confidence_status = "Excellent"
        elif back_confidence >= 0.6:
            confidence_color = "🟡"
            confidence_status = "Good"
        elif back_confidence >= 0.4:
            confidence_color = "🟠"
            confidence_status = "Fair"
        else:
            confidence_color = "🔴"
            confidence_status = "Poor"
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Confidence Score", f"{back_confidence:.2f}", f"{confidence_color} {confidence_status}")
        with col2:
            st.metric("Word Count", back_word_count)
        with col3:
            st.metric("Text Quality", confidence_status)
    
    # ISBN extraction with enhanced features
    if back_text:
        with st.spinner('Extracting and validating ISBNs...'):
            if use_online_validation:
                # Use enhanced ISBN detection with online validation
                isbn_results = extract_and_validate_isbns(back_text)
                isbns = isbn_results
            else:
                # Use basic ISBN detection
                isbns = extract_isbns(back_text)
        
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
    if metadata and isbns:
        st.header('🎯 Final Combined Results')
        
        # Prepare ISBN list for combination
        isbn_list = []
        if use_online_validation and isinstance(isbns, dict) and 'isbns' in isbns:
            # Enhanced ISBN results
            isbn_list = [isbn_data['isbn'] for isbn_data in isbns['isbns']]
        elif isinstance(isbns, dict):
            # Basic ISBN results
            if isbns.get('isbn10'):
                isbn_list.extend(isbns['isbn10'])
            if isbns.get('isbn13'):
                isbn_list.extend(isbns['isbn13'])
        elif isinstance(isbns, list):
            isbn_list = isbns
        
        combined_metadata = metadata_combiner(metadata, isbn_list)
        
        # Display combined results
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📋 Combined Metadata")
            if combined_metadata.get('title'):
                st.success(f"**Title:** {combined_metadata['title']}")
            if combined_metadata.get('authors'):
                st.success(f"**Authors:** {', '.join(combined_metadata['authors'])}")
            if combined_metadata.get('isbns'):
                st.success(f"**ISBNs:** {', '.join(combined_metadata['isbns'])}")
        
        with col2:
            st.subheader("📊 Processing Summary")
            if front_confidence:
                st.metric("Front Cover Confidence", f"{front_confidence:.2f}")
            if back_confidence:
                st.metric("Back Cover Confidence", f"{back_confidence:.2f}")
            if use_online_validation and isinstance(isbns, dict) and 'validated_count' in isbns:
                st.metric("Validated ISBNs", isbns['validated_count'])
        
        if show_processing_details:
            st.subheader("🔍 Detailed Results")
            st.json(combined_metadata)

    # Performance metrics
    st.header('📈 Performance Metrics')
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if front_confidence:
            st.metric("Front Cover Quality", f"{front_confidence:.2f}")
        if back_confidence:
            st.metric("Back Cover Quality", f"{back_confidence:.2f}")
    
    with col2:
        if 'front_word_count' in locals():
            st.metric("Front Words", front_word_count)
        if 'back_word_count' in locals():
            st.metric("Back Words", back_word_count)
    
    with col3:
        if use_online_validation and isinstance(isbns, dict):
            st.metric("ISBNs Found", isbns.get('total_found', 0))
            st.metric("ISBNs Validated", isbns.get('validated_count', 0))

