import cv2
import time
import numpy as np
import os
from PIL import Image
import streamlit as st
import matplotlib.pyplot as plt

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
input_source = st.sidebar.selectbox(
    "Image Source",
    ["Upload", "Load from Disk"]
)
# FFT is now the default mode
enhancement_mode = st.sidebar.selectbox(
    "DIP Enhancement Mode",
    ["fft", "clahe", "he", "sharpen", "adaptive", "none"] 
)
show_segments = st.sidebar.checkbox("Show Character Segments", value=False)
st.sidebar.markdown("---")
st.sidebar.markdown("**Powered by:**\n- YOLOv8 (Custom Trained)\n- Spatial & Frequency DIP Filtering\n- PaddleOCR")

# --- Main App Logic ---
uploaded_file = None
disk_path = None

if input_source == "Upload":
    uploaded_file = st.file_uploader("Upload a Car Image", type=["jpg", "jpeg", "png"])
else:
    disk_path = st.text_input("Enter full image path (e.g., C:\\path\\to\\image.jpg)")

original_img = None
image_name = "uploaded_image"

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    original_img = cv2.imdecode(file_bytes, 1)
    image_name = os.path.splitext(uploaded_file.name)[0]
elif disk_path:
    if os.path.isfile(disk_path):
        original_img = cv2.imread(disk_path)
        image_name = os.path.splitext(os.path.basename(disk_path))[0]
    else:
        st.error("Invalid path. Please provide a valid image file path.")

if original_img is not None:
    
    st.subheader("Raw Input Image")
    st.image(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB), width="stretch")

    if st.button("Process Image", type="primary"):
        start_time = time.time()
        
        with st.spinner("Running AI Pipeline..."):
            plate_pts = processor.find_plate_direct(original_img)
            detection_source = "YOLO"
            
            if plate_pts is not None:
                base_name = image_name
                st.info(f"✅ Detection source: **{detection_source}**")

                if action == "Extract Plate Text":
                    # --- Process the DIP Math First ---
                    warped = processor.four_point_transform(original_img, plate_pts)
                    enhanced_plate = processor.enhance_for_ocr(warped, enhancement_mode)
                    edge_map = processor.generate_edge_map(warped)
                    mse, psnr = processor.calculate_metrics(warped, enhanced_plate)
                    
                    st.markdown("---")
                    
                    # --- MANDATORY REQUIREMENT: QUANTITATIVE EVALUATION ---
                    st.subheader("Statistical Quality Metrics")
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Mean Squared Error (MSE)", f"{mse:.2f}")
                    if psnr == float('inf'):
                        m2.metric("Peak Signal-to-Noise Ratio (PSNR)", "Infinity")
                    else:
                        m2.metric("Peak Signal-to-Noise Ratio (PSNR)", f"{psnr:.2f} dB")
                    m3.metric("Active DIP Filter", enhancement_mode.upper())
                    
                    st.markdown("---")
                    st.subheader("DIP Pipeline Visualization")
                    
                    # --- MANDATORY REQUIREMENT: SPATIAL EDGE DETECTION ---
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.markdown("**1. Raw Cropped Plate**")
                        st.image(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB), width="stretch")
                    with col2:
                        st.markdown("**2. Spatial Edge Map (Sobel)**")
                        st.image(edge_map, width="stretch", channels="GRAY")
                    with col3:
                        st.markdown(f"**3. Enhanced Output ({enhancement_mode.upper()})**")
                        st.image(cv2.cvtColor(enhanced_plate, cv2.COLOR_BGR2RGB), width="stretch")

                    # --- CRASH-PROOF MATPLOTLIB HISTOGRAMS ---
                    st.markdown("---")
                    st.subheader("Grayscale Intensity Histograms")
                    
                    # Create a safe, static Matplotlib figure for the Raw Image
                    fig1, ax1 = plt.subplots(figsize=(6, 3))
                    fig1.patch.set_facecolor('#0B0F19') # Matches dark theme
                    ax1.set_facecolor('#1A2235')
                    ax1.hist(cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY).ravel(), 256, [0, 256], color='#FF0000', alpha=0.7)
                    ax1.tick_params(colors='white')
                    ax1.set_title("Raw Plate", color='white')
                    
                    # Create a safe, static Matplotlib figure for the Enhanced Image
                    fig2, ax2 = plt.subplots(figsize=(6, 3))
                    fig2.patch.set_facecolor('#0B0F19')
                    ax2.set_facecolor('#1A2235')
                    ax2.hist(cv2.cvtColor(enhanced_plate, cv2.COLOR_BGR2GRAY).ravel(), 256, [0, 256], color='#00FFAA', alpha=0.7)
                    ax2.tick_params(colors='white')
                    ax2.set_title(f"Enhanced Plate ({enhancement_mode.upper()})", color='white')

                    # Display them side-by-side using Streamlit's static pyplot renderer
                    h1, h2 = st.columns(2)
                    with h1:
                        st.pyplot(fig1)
                    with h2:
                        st.pyplot(fig2)
                    
                    # Close the figures to completely free up memory
                    plt.close(fig1)
                    plt.close(fig2)

                    if show_segments:
                        st.markdown("---")
                        chars, bw, boxes = processor.segment_characters(enhanced_plate)
                        st.markdown("**4. Character Segmentation (Morphological DIP)**")
                        st.image(bw, caption="Adaptive Threshold + Morphological Opening Mask", width="stretch")
                        if chars:
                            st.markdown("**Segments (Left → Right)**")
                            st.image(
                                [cv2.cvtColor(c, cv2.COLOR_BGR2RGB) for c in chars],
                                width="stretch"
                            )
                        else:
                            st.info("No character segments found. Try another enhancement mode or a clearer image.")
                    
                    st.markdown("---")
                    # --- OCR Extraction ---
                    text = extractor.extract_plate_text(enhanced_plate)
                    csv_path = os.path.join(EXTRACTED_FOLDER, "extracted_plates.csv")
                    
                    try:
                        extractor.save_to_csv(text, csv_path)
                    except TypeError:
                        extractor.save_to_csv(text)
                    
                    st.success(f"**Extracted Number Plate:** {text}")
                    st.info(f"💾 Extracted data saved to `{csv_path}`")

                elif action == "Blur Number Plate":
                    blurred_img = processor.blur_plate_on_original(original_img, plate_pts)
                    st.markdown("---")
                    st.subheader("Anonymized Output")
                    st.image(cv2.cvtColor(blurred_img, cv2.COLOR_BGR2RGB), width="stretch")
                    
                    blurred_image_path = os.path.join(BLURRED_FOLDER, f"blurred_{base_name}.jpg")
                    cv2.imwrite(blurred_image_path, blurred_img)
                    st.success("Successfully blurred the license plate.")
                    st.info(f"💾 Blurred image saved to `{blurred_image_path}`")

                time_taken = time.time() - start_time
                st.info(f"⏱️ **Pipeline completed in {time_taken:.3f} seconds**")

            else:
                st.error("No license plate detected. Please try an image with a clearer view of the plate.")