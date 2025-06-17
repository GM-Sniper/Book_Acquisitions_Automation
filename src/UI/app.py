import streamlit as st
import sys
import os
import cv2

# Add the project root folder to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.vision.preprocessing import preprocess_image

# Utility function to save preprocessed image
def save_preprocessed_image(image_np, original_filename, output_dir='data/processed'):
    os.makedirs(output_dir, exist_ok=True)
    base, ext = os.path.splitext(original_filename)
    out_path = os.path.join(output_dir, f"{base}_processed.png")
    cv2.imwrite(out_path, image_np)
    return out_path

# Title of the app
st.title('Book Cover OCR Preprocessing Demo')

# --- Section: Upload image(s) ---
st.header('Step 1: Upload Book Cover Images')
st.write('Please upload the **front cover** image of the book.')
front_file = st.file_uploader('Front Cover Image', type=['jpg', 'jpeg', 'png'], key='front')

st.write('Now upload the **back cover** image of the book.')
back_file = st.file_uploader('Back Cover Image', type=['jpg', 'jpeg', 'png'], key='back')

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

st.header('Next Steps')

# --- Section: Capture image from webcam ---
st.header('Or Capture Image from Webcam')
captured_image = st.camera_input('Take a picture with your webcam')
if captured_image is not None:
    st.write("Captured Image:")
    st.image(captured_image, caption="Captured Image")