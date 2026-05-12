import cv2
import numpy as np
from ultralytics import YOLO

class ImageProcessor:
    def __init__(self, model_path="best.pt"):
        # Initialize your newly trained custom YOLOv8 model
        self.model = YOLO(model_path)

    @staticmethod
    def order_points(pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    @staticmethod
    def four_point_transform(image, pts):
        rect = ImageProcessor.order_points(pts)
        (tl, tr, br, bl) = rect
        widthA = np.linalg.norm(br - bl)
        widthB = np.linalg.norm(tr - tl)
        maxWidth = max(int(widthA), int(widthB))
        heightA = np.linalg.norm(tr - br)
        heightB = np.linalg.norm(tl - bl)
        maxHeight = max(int(heightA), int(heightB))
        dst = np.array([[0, 0], [maxWidth - 1, 0],
                        [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype="float32")
        M = cv2.getPerspectiveTransform(rect, dst)
        return cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    @staticmethod
    def enhance_for_ocr(warped_plate):
        """
        Refined DIP pipeline for Deep Learning OCRs.
        Fixes lighting and noise, but preserves natural character gradients.
        """
        # 1. Grayscale Conversion
        gray = cv2.cvtColor(warped_plate, cv2.COLOR_BGR2GRAY)

        # 2. CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_gray = clahe.apply(gray)

        # 3. Spatial Filtering (Bilateral Filter)
        # Melts away background grain but keeps the text edges sharp and natural
        smoothed = cv2.bilateralFilter(enhanced_gray, 11, 17, 17)

        # 4. Convert back to 3-channel BGR format
        # Deep Learning OCRs expect a standard 3-channel image array
        final_plate = cv2.cvtColor(smoothed, cv2.COLOR_GRAY2BGR)

        return final_plate

    def find_plate_direct(self, original_img):
        """
        Direct Pipeline: Custom YOLOv8 model directly finds the license plate.
        """
        # Run the custom model with a 10% confidence threshold and strict sizing
        results = self.model(original_img, conf=0.1, imgsz=640) 
        
        for result in results:
            boxes = result.boxes
            if len(boxes) > 0:
                # Grab the highest confidence box (The License Plate)
                x1, y1, x2, y2 = map(int, boxes[0].xyxy[0])
                
                # Convert the bounding box into the 4 points needed for the DIP pipeline
                pts = np.array([
                    [x1, y1], # Top-left
                    [x2, y1], # Top-right
                    [x2, y2], # Bottom-right
                    [x1, y2]  # Bottom-left
                ], dtype=np.int32)
                
                return pts
                
        return None

    @staticmethod
    def blur_plate_on_original(image, pts):
        mask = np.zeros_like(image)
        cv2.fillPoly(mask, [pts], (255, 255, 255))
        blurred = cv2.GaussianBlur(image, (51, 51), 0)
        out = np.where(mask == np.array([255, 255, 255]), blurred, image)
        return out