import streamlit as st

# Title of the app
st.title('Book Image Upload & Capture')

# --- Section: Upload image(s) ---
st.header('Upload Image(s)')
uploaded_files = st.file_uploader(
    'Upload one or more images (e.g., book cover, barcode)',
    type=['jpg', 'jpeg', 'png'],
    accept_multiple_files=True
)
if uploaded_files:
    for uploaded_file in uploaded_files:
        st.write(f"Filename: {uploaded_file.name}")
        st.image(uploaded_file, caption=uploaded_file.name)

# --- Section: Capture image from webcam ---
st.header('Or Capture Image from Webcam')
captured_image = st.camera_input('Take a picture with your webcam')
if captured_image is not None:
    st.write("Captured Image:")
    st.image(captured_image, caption="Captured Image")