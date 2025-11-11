import streamlit as st
from PIL import Image
import os
from pathlib import Path
import sys

# --- Fix for Python Imports ---
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))
# -----------------------------

from src.extraction_logic.extractor import extract_receipt_info

# --- Page Configuration ---
st.set_page_config(
    page_title="Receipt Text Extractor",
    page_icon="🧾",
    layout="wide"
)

# --- Main Application ---
st.title("🧾 Retail Bill - Full Text Extractor")
st.write("Upload an image of a retail receipt, and the AI will find and read all the text.")

uploaded_file = st.file_uploader("Choose a receipt image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    st.image(Image.open(uploaded_file), caption='Original Uploaded Receipt', use_container_width=True)
    
    st.write("") # Spacer
    
    temp_dir = project_root / "temp"
    temp_dir.mkdir(exist_ok=True)
    input_image_path = temp_dir / uploaded_file.name
    annotated_image_path = temp_dir / f"annotated_{uploaded_file.name}"
    
    with open(input_image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
        
    with st.spinner('analyzing the receipt...'):
        # --- FIX IS HERE ---
        # We now unpack all THREE return values from the function
        parsed_data, annotated_path, reconstructed_lines = extract_receipt_info(str(input_image_path), str(annotated_image_path))

    st.success("Analysis complete!")
    
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Key Information")
        # Display the parsed data (company, date, total, items)
        st.json(parsed_data)
        
        st.subheader(" Detected Fields")
        if annotated_path and Path(annotated_path).exists():
            st.image(Image.open(annotated_path), caption='All detected text fields')
        
    with col2:
        # Display the full reconstructed text
        st.subheader("Reconstructed Bill Text")
        if reconstructed_lines:
            # Display in a code block to preserve formatting
            st.code("\n".join(reconstructed_lines))
        else:
            st.warning("Could not reconstruct text lines.")

    
    # Clean up temporary files
    os.remove(input_image_path)
    if annotated_path and Path(annotated_path).exists():
        os.remove(annotated_path)