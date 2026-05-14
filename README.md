# Deep Learning ALPR System

A Streamlit-based Automatic License Plate Recognition (ALPR) application that detects license plates using a custom YOLOv8 model, enhances the plate region with a Digital Image Processing (DIP) pipeline, and extracts text via PaddleOCR. It also supports anonymization by blurring detected plates on the original image.

## Highlights
- **YOLOv8 plate localization (AI-only detection)** for robust plate finding.
- **Extensive DIP pipeline**: grayscale, CLAHE/HE, sharpening, adaptive thresholding, spatial filtering.
- **Character segmentation (DIP)** for visual verification of extracted glyphs.
- **PaddleOCR-based text extraction** with timestamped CSV logging.
- **Privacy mode** to blur license plates in the original image.
- **Streamlit UI** with selectable DIP modes.

## Project Structure
- [app.py](app.py): Streamlit application entry point and UI logic.
- [processing.py](processing.py): Plate detection, perspective transform, enhancement, and blurring.
- [extraction.py](extraction.py): OCR extraction and CSV persistence.
- [best.pt](best.pt): Custom-trained YOLOv8 model weights.
- [requirements.txt](requirements.txt): Python dependencies.
- pics/: Sample images (optional).

## Architecture Overview
1. **Upload image** in the Streamlit UI.
2. **Plate localization** via YOLOv8 (AI-only detection).
3. **Perspective warp** isolates the plate region.
4. **DIP enhancement** (CLAHE/HE/sharpen/adaptive/none).
5. **Character segmentation** (DIP) for visual verification.
6. **PaddleOCR extracts** the plate text.
7. **Output saved** to CSV; optional **blurring** for anonymization.

## Features
### 1) Plate Text Extraction
- Detects license plate location (YOLOv8).
- Warps and enhances the plate region (selectable DIP mode).
- Optionally segments characters for visualization.
- Extracts text using PaddleOCR.
- Saves results to a timestamped CSV file in the extracted_data folder.

### 2) Plate Blurring (Anonymization)
- Detects license plate location.
- Applies Gaussian blur to the plate area.
- Saves the anonymized image in the blurred_plates folder.

## Requirements
- Python 3.9+ recommended
- See [requirements.txt](requirements.txt) for full dependency list.

## Setup
1. **Create and activate a virtual environment**.
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Ensure model weights are present**:
   - Place [best.pt](best.pt) in the project root.

## Run the Application
```bash
streamlit run app.py
```

## Output Files
- **Extracted text logs**: extracted_data/extracted_plates.csv
- **Blurred images**: blurred_plates/blurred_<image_name>.jpg

## Notes on Accuracy
- Detection quality depends on the **custom YOLOv8 model** and image quality.
- OCR results improve with clear, well-lit plates.
- The enhancement pipeline is designed to preserve natural gradients for OCR stability.

## Extending the Project
- Replace or fine-tune the YOLO model for different regions or plate formats.
- Add post-processing for plate normalization (e.g., regex validation).
- Store results in a database instead of CSV.
- Add batch processing for multiple images.
- Add quantitative evaluation (precision/recall, OCR accuracy) for YOLO vs DIP.

## Acknowledgements
- **Ultralytics YOLOv8** for detection
- **PaddleOCR** for text recognition
- **OpenCV** and **NumPy** for image processing
- **Streamlit** for the interactive UI

## License
This project is for academic and research use. Add a license file if you plan to publish publicly.
