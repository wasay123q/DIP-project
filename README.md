# Deep Learning ALPR System

![DIP](https://img.shields.io/badge/DIP-Academic%20Project-4c8bf5)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Custom%20Model-8a2be2)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-ff4b4b)
![License](https://img.shields.io/badge/License-Academic-lightgrey)

## Abstract
**Deep Learning ALPR System** is a 6th‑semester Digital Image Processing (DIP) project that integrates a custom YOLOv8 detector with a mathematically rigorous DIP enhancement bridge. The system localizes a license plate, applies spatial and frequency‑domain enhancements to maximize character contrast, and extracts text using PaddleOCR. It also provides privacy‑preserving plate anonymization through blurring. The design explicitly demonstrates **core DIP techniques** with visual outputs and metrics to satisfy academic evaluation criteria.

## Why This Project Matters (DIP Focus)
This project is designed to **showcase core DIP concepts**, not just AI inference. The AI model performs localization, while the **DIP pipeline performs the signal conditioning** that makes OCR reliable. Each processing step is visualized, measured, and justified with quantitative metrics.

## Key Features
- **AI localization** with a custom YOLOv8 model for robust plate detection.
- **DIP enhancement bridge** with selectable modes:
  - **FFT High‑Pass Filter** (frequency domain sharpening)
  - **CLAHE** (local contrast equalization)
  - **HE** (global histogram equalization)
  - **Sharpening** (unsharp masking)
  - **Adaptive thresholding** (text/background separation)
- **Spatial operations**: bilateral filtering, Gaussian blur, morphological opening.
- **Edge detection**: Sobel X/Y gradient magnitude map.
- **Histogram analysis**: 256‑bin grayscale histograms (raw vs enhanced) rendered with matplotlib.
- **Quantitative evaluation**: real‑time MSE and PSNR between raw and enhanced plates.
- **Two user modes**:
  - Extract plate text → logs to extracted_data/extracted_plates.csv
  - Blur plate → saves anonymized images to blurred_plates/
- **Streamlit UI** with a 4‑step visual pipeline and optional character segmentation.

## Technical Stack
- **Python** for orchestration and algorithms
- **OpenCV** for image processing and geometric transforms
- **Ultralytics YOLOv8** for plate localization
- **PaddleOCR** for text extraction
- **Streamlit** for the interactive UI
- **Matplotlib** for histogram visualization

## System Architecture / Pipeline Flow
1. **Input acquisition**: upload image or load from disk path.
2. **YOLOv8 localization**: plate coordinates are detected.
3. **Perspective correction**: rectifies the plate for consistent processing.
4. **DIP enhancement**: spatial/frequency filtering to improve contrast.
5. **Edge analysis**: Sobel gradient map for feature visualization.
6. **Histogram analysis**: grayscale intensity distribution (raw vs enhanced).
7. **Character segmentation**: adaptive thresholding + morphological opening.
8. **OCR extraction**: PaddleOCR reads the enhanced plate.
9. **Persistence**: results saved to CSV; optional blurring saved to disk.

## DIP Concepts Implemented (Explicit)
- **Grayscale conversion** for intensity‑domain processing.
- **Spatial filtering**: bilateral filtering and Gaussian blur.
- **Morphology**: morphological opening for segmentation cleanup.
- **Histogram operations**: HE/CLAHE plus explicit histogram visualization.
- **Edge detection**: Sobel gradient magnitude.
- **Frequency‑domain filtering**: FFT high‑pass mask.
- **Quantitative metrics**: MSE and PSNR between raw and enhanced plates.
- **Geometric transform**: perspective warping for plate rectification.

## DIP Enhancement Modes (Detailed)
- **FFT (High‑Pass)**: Applies DFT, masks low frequencies, and reconstructs edges to sharpen character strokes. Best for plates with soft edges or low contrast.
- **CLAHE**: Local contrast enhancement to recover text in uneven illumination.
- **HE (Histogram Equalization)**: Global contrast normalization, useful for evenly lit plates.
- **Sharpen**: Unsharp masking to enhance edges while preserving overall intensity.
- **Adaptive Threshold**: Converts to a high‑contrast binary image for clear foreground separation.
- **None**: No enhancement, useful for debugging and baseline comparison.

## Edge and Segmentation Details
- **Sobel Edge Map**: Computes gradient magnitude from X and Y derivatives to visualize character boundaries.
- **Character Segmentation**: Adaptive thresholding followed by morphological opening to remove noise and isolate character blobs for visualization.

## Quantitative Metrics (Definitions)
- **MSE (Mean Squared Error)** measures average squared difference between raw and enhanced plates.
- **PSNR (Peak Signal‑to‑Noise Ratio)** indicates enhancement quality in dB; higher is better.

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
To avoid OCR engine instability on Windows, use these versions:
- paddlepaddle==2.6.2
- paddleocr==2.8.1
- streamlit>=1.30.0
- opencv-python>=4.8.0
- ultralytics>=8.1.0
- matplotlib>=3.8.0
- pandas>=2.0.0

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
4. Review the visual pipeline and metrics panel.

## Parameters and Defaults
- **YOLO confidence threshold**: 0.1
- **YOLO image size**: 640
- **FFT low‑frequency radius**: 15 (high‑pass mask)
- **Sobel kernel size**: 3
- **Adaptive threshold**: block size 31, C = 10
- **Morphological opening**: 3×3 structuring element

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

## Reporting Guidance (For Academic Submission)
- Include sample input/output images for at least 5 test cases.
- Add a small table of MSE/PSNR values across multiple images.
- Briefly justify each DIP step and its effect on OCR quality.
- Add a pipeline diagram and a data‑flow diagram.

## Limitations
- YOLO detection may fail in extreme occlusions or very low‑resolution images.
- OCR accuracy depends on plate quality and illumination.
- FFT sharpening can amplify noise if the input is extremely grainy.

## Troubleshooting
- If OCR crashes on Windows, verify paddlepaddle==2.6.2 and paddleocr==2.8.1.
- If no plate is detected, try a clearer image or adjust lighting conditions.
- If the UI stops, reduce image size or disable heavy modes (e.g., FFT) for testing.

## Academic Rubric Alignment
- **Image I/O**: upload and disk‑load inputs; saved CSV and blurred images.
- **Preprocessing**: grayscale, denoising, enhancement, normalization.
- **Core DIP algorithms**: filtering, morphology, segmentation, FFT.
- **Post‑processing**: histogram analysis + MSE/PSNR metrics.
- **Integration**: complete pipeline from input to final output.

## Acknowledgements
- Ultralytics YOLOv8
- PaddleOCR
- OpenCV, NumPy, Matplotlib
- Streamlit

## License
Academic use only. Add a license file if publishing publicly.
