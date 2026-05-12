import cv2
import time
import numpy as np
import os
from PIL import Image
import streamlit as st


from processing import ImageProcessor
from extraction import DataExtractor

# --- Directory Management ---
BLURRED_FOLDER = "blurred_plates"
EXTRACTED_FOLDER = "extracted_data"

if not os.path.exists(BLURRED_FOLDER):
    os.makedirs(BLURRED_FOLDER)
if not os.path.exists(EXTRACTED_FOLDER):
    os.makedirs(EXTRACTED_FOLDER)

# --- Page Configuration ---
st.set_page_config(page_title="Deep Learning ALPR", page_icon="🚗", layout="wide")

# --- Load AI Models (Cached for Speed) ---
@st.cache_resource
def load_models():
    processor = ImageProcessor()
    extractor = DataExtractor()
    return processor, extractor

processor, extractor = load_models()

# --- UI Header ---
st.title("🚗 Deep Learning ALPR System")
st.markdown("Upload a vehicle image below to isolate, enhance, and extract the license plate using our custom YOLOv8 and DIP pipeline.")

# --- Sidebar Controls ---
st.sidebar.header("Control Panel")
action = st.sidebar.radio("Choose Action:", ["Extract Plate Text", "Blur Number Plate"])
st.sidebar.markdown("---")
st.sidebar.markdown("**Powered by:**\n- YOLOv8 (Custom Trained)\n- Spatial DIP Filtering\n- PaddleOCR")

# --- Main App Logic ---
uploaded_file = st.file_uploader("Upload a Car Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    original_img = cv2.imdecode(file_bytes, 1)
    
    st.subheader("Raw Input Image")
    st.image(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB), use_container_width=True)

    if st.button("Process Image", type="primary"):
        start_time = time.time()
        
        with st.spinner("Running AI Pipeline..."):
            plate_pts = processor.find_plate_direct(original_img)
            
            if plate_pts is not None:
                base_name = os.path.splitext(uploaded_file.name)[0]

                if action == "Extract Plate Text":
                    st.markdown("---")
                    st.subheader("Pipeline Visualization")
                    col1, col2 = st.columns(2)
                    
                    warped = processor.four_point_transform(original_img, plate_pts)
                    with col1:
                        st.markdown("**1. Cropped & Warped (Raw)**")
                        st.image(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB), use_container_width=True)
                    
                    enhanced_plate = processor.enhance_for_ocr(warped)
                    with col2:
                        st.markdown("**2. DIP Enhanced (Deep Learning Ready)**")
                        st.image(cv2.cvtColor(enhanced_plate, cv2.COLOR_BGR2RGB), use_container_width=True)
                    
                    text = extractor.extract_plate_text(enhanced_plate)
                    
                    csv_path = os.path.join(EXTRACTED_FOLDER, "extracted_plates.csv")
                    # Note: Ensure extraction.py accepts a second argument for path if needed
                    try:
                        extractor.save_to_csv(text, csv_path)
                    except TypeError:
                        # Fallback just in case your extraction.py hasn't been updated to take the path yet
                        extractor.save_to_csv(text)
                    
                    st.success(f"**Extracted Number Plate:** {text}")
                    st.info(f"💾 Extracted data saved to `{csv_path}`")

                elif action == "Blur Number Plate":
                    blurred_img = processor.blur_plate_on_original(original_img, plate_pts)
                    st.markdown("---")
                    st.subheader("Anonymized Output")
                    st.image(cv2.cvtColor(blurred_img, cv2.COLOR_BGR2RGB), use_container_width=True)
                    
                    blurred_image_path = os.path.join(BLURRED_FOLDER, f"blurred_{base_name}.jpg")
                    cv2.imwrite(blurred_image_path, blurred_img)
                    st.success("Successfully blurred the license plate.")
                    st.info(f"💾 Blurred image saved to `{blurred_image_path}`")

                time_taken = time.time() - start_time
                st.info(f"⏱️ **Pipeline completed in {time_taken:.3f} seconds** (O(1) Direct YOLOv8 Inference)")

            else:
                st.error("No license plate detected. Please try an image with a clearer view of the plate.")