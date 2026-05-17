# Deep Learning ALPR System

![DIP](https://img.shields.io/badge/DIP-Academic%20Project-4c8bf5)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Custom%20Model-8a2be2)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-ff4b4b)
![License](https://img.shields.io/badge/License-Academic-lightgrey)

## Overview
**Deep Learning ALPR System** is a 6th‑semester Digital Image Processing (DIP) project that bridges modern AI with rigorous mathematical image processing. It uses a custom YOLOv8 detector to localize the plate, then applies a DIP‑first enhancement bridge (spatial + frequency domain) to maximize character clarity before OCR extraction with PaddleOCR.

## Key Features
- **AI‑only localization** using a custom YOLOv8 model for reliable plate detection.
- **DIP enhancement bridge** with selectable modes: CLAHE, HE, sharpening, adaptive thresholding, and FFT high‑pass filtering.
- **Mandatory DIP concepts satisfied**:
  - **Frequency domain filtering** (FFT high‑pass sharpening).
  - **Spatial domain operations** (bilateral filtering, Gaussian blur, morphological opening, adaptive thresholding).
  - **Histogram analysis** (256‑bin grayscale histograms: raw vs. enhanced).
  - **Edge detection** (Sobel gradient magnitude map).
  - **Quantitative evaluation** (MSE and PSNR).
- **Dual application modes**: text extraction (CSV logging) and privacy‑preserving plate blurring.
- **Streamlit UI** with a cyberpunk dark theme and a 4‑step visual pipeline.

## System Architecture / Pipeline Flow
1. **Input acquisition**: image upload or disk path load.
2. **Localization**: YOLOv8 detects plate coordinates.
3. **Perspective correction**: geometric warping to frontal view.
4. **DIP enhancement bridge**: spatial/frequency filters to optimize contrast.
5. **Edge analysis**: Sobel X/Y gradient magnitude map.
6. **Histogram analysis**: 256‑bin grayscale histograms for raw vs. enhanced.
7. **Character segmentation**: adaptive thresholding + morphological opening.
8. **OCR extraction**: PaddleOCR reads multi‑line text.
9. **Persistence**: results logged to CSV; optional plate blurring saved to disk.

## Installation & Setup
### Prerequisites
- Python 3.9+

### Environment Setup
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### Critical Dependencies (Stable Builds)
To avoid Windows OCR engine crashes, use the pinned versions below:
- paddlepaddle==2.6.2
- paddleocr==2.8.1
- streamlit>=1.30.0
- opencv-python>=4.8.0
- ultralytics>=8.1.0
- matplotlib
- pandas

### Model Weights
Place the custom YOLOv8 weights file in the project root:
- best.pt

## Usage
```bash
streamlit run app.py
```

### UI Workflow
1. Select **Image Source**: Upload or Load from Disk.
2. Choose **DIP Enhancement Mode** (default: FFT).
3. Click **Process Image**.
4. Review the 4‑step DIP visualization and metrics.

## Directory Structure
```
.
├── .streamlit/
├── blurred_plates/
├── extracted_data/
├── pics/
├── app.py
├── processing.py
├── extraction.py
├── best.pt
├── requirements.txt
└── README.md
```

## Outputs
- Extracted text logs: extracted_data/extracted_plates.csv
- Blurred images: blurred_plates/blurred_<image_name>.jpg

## Academic Alignment (DIP Requirements)
- **Image I/O**: load from disk + save processed outputs.
- **Preprocessing**: grayscale, denoising, normalization/contrast enhancement.
- **Core DIP algorithms**: spatial filters, morphology, segmentation, FFT filtering.
- **Post‑processing**: metrics (MSE, PSNR), histogram analysis.
- **Integration**: end‑to‑end pipeline from input to final output.

## Acknowledgements
- Ultralytics YOLOv8
- PaddleOCR
- OpenCV, NumPy, Matplotlib
- Streamlit

## License
Academic use only. Add a license file if publishing publicly.
