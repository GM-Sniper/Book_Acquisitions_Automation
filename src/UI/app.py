import streamlit as st
import sys
import os
import cv2

# Add the project root folder to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.vision.preprocessing import preprocess_image
from src.vision.OCR_Processing import extract_text_from_image
from src.metadata.metadata_extraction import extract_metadata_with_gemini, metadata_combiner
from src.utils.isbn_detection import extract_isbns

# Utility function to save preprocessed image
def save_preprocessed_image(image_np, original_filename, output_dir='data/processed'):
    os.makedirs(output_dir, exist_ok=True)
    base, ext = os.path.splitext(original_filename)
    out_path = os.path.join(output_dir, f"{base}_processed.png")
    cv2.imwrite(out_path, image_np)
    return out_path

# Title of the app
st.title('Book Cover OCR Proof of Concept')

# --- Section: Upload image(s) ---
st.header('Step 1: Upload Book Cover Images')
st.write('Please upload the **front cover** image of the book.')
front_file = st.file_uploader('Front Cover Image', type=['jpg', 'jpeg', 'png'], key='front')

st.write('Now upload the **back cover** image of the book.')
back_file = st.file_uploader('Back Cover Image', type=['jpg', 'jpeg', 'png'], key='back')

front_text = None
back_text = None
metadata = None
isbns = None
combined_metadata = None

if front_file:
    st.subheader('Front Cover: Original')
    st.image(front_file, caption='Front Cover (Original)')
    # Preprocess and display
    preprocessed_front = preprocess_image(front_file.read())
    st.subheader('Front Cover: Preprocessed for OCR')
    st.image(preprocessed_front, caption='Front Cover (Preprocessed)', channels='GRAY')
    # Save preprocessed image
    saved_path = save_preprocessed_image(preprocessed_front, front_file.name)
    st.success(f'Preprocessed front cover saved to: {saved_path}')
    # OCR
    front_text = extract_text_from_image(preprocessed_front)
    st.subheader('Extracted Text (Front Cover)')
    st.text_area('Front Cover OCR Text', front_text, height=150)
    # Metadata extraction
    metadata = extract_metadata_with_gemini(front_text)
    if metadata:
        st.subheader('Extracted Metadata (Front Cover)')
        st.json(metadata)
    else:
        st.warning('No metadata (title/author) found in front cover text.')

if back_file:
    st.subheader('Back Cover: Original')
    st.image(back_file, caption='Back Cover (Original)')
    # Preprocess and display
    preprocessed_back = preprocess_image(back_file.read())
    st.subheader('Back Cover: Preprocessed for OCR')
    st.image(preprocessed_back, caption='Back Cover (Preprocessed)', channels='GRAY')
    # Save preprocessed image
    saved_path = save_preprocessed_image(preprocessed_back, back_file.name)
    st.success(f'Preprocessed back cover saved to: {saved_path}')
    # OCR
    back_text = extract_text_from_image(preprocessed_back)
    st.subheader('Extracted Text (Back Cover)')
    st.text_area('Back Cover OCR Text', back_text, height=150)
    # ISBN extraction
    isbns = extract_isbns(back_text)
    if isbns and (isbns['isbn10'] or isbns['isbn13']):
        st.subheader('Extracted ISBNs (Back Cover)')
        st.json(isbns)
    else:
        st.warning('No ISBNs found in back cover text.')

# Combine and display final result if both are available
if metadata and isbns:
    # Flatten ISBNs to a list (if extract_isbns returns dict)
    isbn_list = []
    if isinstance(isbns, dict):
        if isbns.get('isbn10'):
            isbn_list.append(isbns['isbn10'])
        if isbns.get('isbn13'):
            isbn_list.append(isbns['isbn13'])
    elif isinstance(isbns, list):
        isbn_list = isbns
    combined_metadata = metadata_combiner(metadata, isbn_list)
    st.header('Final Combined Metadata (Front + Back)')
    st.json(combined_metadata)

st.header('Proof of Concept Complete')