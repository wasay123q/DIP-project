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
    def enhance_for_ocr(warped_plate, mode="clahe"):
        """
        DIP enhancement pipeline for OCR readiness.
        mode: clahe | he | sharpen | adaptive | fft | none
        """
        if mode == "none":
            return warped_plate

        gray = cv2.cvtColor(warped_plate, cv2.COLOR_BGR2GRAY)

        if mode == "fft":
            # Apply Discrete Fourier Transform
            dft = cv2.dft(np.float32(gray), flags=cv2.DFT_COMPLEX_OUTPUT)
            dft_shift = np.fft.fftshift(dft)
            
            rows, cols = gray.shape
            crow, ccol = rows // 2, cols // 2
            
            # Create a High Pass Filter (HPF) mask to sharpen edges
            mask = np.ones((rows, cols, 2), np.uint8)
            r = 15 # radius for low frequencies to block
            x, y = np.ogrid[:rows, :cols]
            mask_area = (x - crow) ** 2 + (y - ccol) ** 2 <= r*r
            mask[mask_area] = 0
            
            # Apply mask and inverse DFT
            fshift = dft_shift * mask
            f_ishift = np.fft.ifftshift(fshift)
            img_back = cv2.idft(f_ishift)
            img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
            
            # Normalize to 0-255
            cv2.normalize(img_back, img_back, 0, 255, cv2.NORM_MINMAX)
            img_back = np.uint8(img_back)
            
            # Combine original with extracted high-frequency edges
            sharp = cv2.addWeighted(gray, 1.2, img_back, 0.5, 0)
            return cv2.cvtColor(sharp, cv2.COLOR_GRAY2BGR)

        if mode == "he":
            enhanced_gray = cv2.equalizeHist(gray)
            smoothed = cv2.bilateralFilter(enhanced_gray, 9, 17, 17)
            return cv2.cvtColor(smoothed, cv2.COLOR_GRAY2BGR)

        if mode == "sharpen":
            blurred = cv2.GaussianBlur(gray, (0, 0), 1.2)
            sharp = cv2.addWeighted(gray, 1.6, blurred, -0.6, 0)
            smoothed = cv2.bilateralFilter(sharp, 7, 17, 17)
            return cv2.cvtColor(smoothed, cv2.COLOR_GRAY2BGR)

        if mode == "adaptive":
            denoised = cv2.medianBlur(gray, 3)
            bw = cv2.adaptiveThreshold(
                denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 21, 10
            )
            return cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)

        # Default: CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_gray = clahe.apply(gray)
        smoothed = cv2.bilateralFilter(enhanced_gray, 11, 17, 17)
        return cv2.cvtColor(smoothed, cv2.COLOR_GRAY2BGR)

    # --- MANDATORY REQUIREMENT: EDGE DETECTION ---
    @staticmethod
    def generate_edge_map(warped_plate):
        """
        Generates a Sobel edge map to demonstrate spatial gradient analysis.
        """
        gray = cv2.cvtColor(warped_plate, cv2.COLOR_BGR2GRAY)
        # Apply Sobel on X and Y axes
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # Calculate combined gradient magnitude
        magnitude = cv2.magnitude(sobelx, sobely)
        cv2.normalize(magnitude, magnitude, 0, 255, cv2.NORM_MINMAX)
        return np.uint8(magnitude)

    # --- MANDATORY REQUIREMENT: QUANTITATIVE EVALUATION ---
    @staticmethod
    def calculate_metrics(img1, img2):
        """
        Calculates MSE and PSNR between the raw plate and enhanced plate.
        """
        # Ensure grayscale for accurate structural comparison
        if len(img1.shape) == 3:
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        if len(img2.shape) == 3:
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            
        # Calculate Mean Squared Error (MSE)
        err = np.sum((img1.astype("float") - img2.astype("float")) ** 2)
        err /= float(img1.shape[0] * img1.shape[1])
        mse = err
        
        # Calculate Peak Signal-to-Noise Ratio (PSNR)
        if mse == 0:
            psnr = float('inf') # Images are perfectly identical
        else:
            max_pixel = 255.0
            psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
            
        return mse, psnr

    # --- MANDATORY REQUIREMENT: HISTOGRAM ANALYSIS ---
    @staticmethod
    def compute_histogram(image):
        """
        Computes a grayscale histogram (256 bins).
        """
        if image is None or image.size == 0:
            return np.zeros(256, dtype=float)

        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        gray = gray.astype(np.uint8)
        hist = np.bincount(gray.ravel(), minlength=256).astype(float)
        return hist

    def find_plate_direct(self, original_img):
        """
        Direct Pipeline: Custom YOLOv8 model directly finds the license plate.
        """
        results = self.model(original_img, conf=0.1, imgsz=640) 
        
        for result in results:
            boxes = result.boxes
            if len(boxes) > 0:
                x1, y1, x2, y2 = map(int, boxes[0].xyxy[0])
                pts = np.array([
                    [x1, y1], [x2, y1], [x2, y2], [x1, y2]
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

    @staticmethod
    def segment_characters(plate_img):
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        bw = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 31, 10
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel, iterations=1)

        contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        h, w = bw.shape[:2]

        boxes = []
        for cnt in contours:
            x, y, cw, ch = cv2.boundingRect(cnt)
            if ch < 0.35 * h or ch > 0.95 * h:
                continue
            aspect = cw / float(ch + 1e-6)
            if aspect < 0.2 or aspect > 1.2:
                continue
            if cw < 0.02 * w:
                continue
            boxes.append((x, y, cw, ch))

        boxes = sorted(boxes, key=lambda b: b[0])
        characters = []
        for (x, y, cw, ch) in boxes:
            char = plate_img[y:y + ch, x:x + cw]
            if char.size > 0:
                characters.append(char)

        return characters, bw, boxes