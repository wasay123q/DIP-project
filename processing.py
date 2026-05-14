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
        mode: clahe | he | sharpen | adaptive | none
        """
        if mode == "none":
            return warped_plate

        gray = cv2.cvtColor(warped_plate, cv2.COLOR_BGR2GRAY)

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
    def validate_plate_candidate(image, pts):
        if pts is None or len(pts) != 4:
            return False
        warped = ImageProcessor.four_point_transform(image, pts)
        h, w = warped.shape[:2]
        if h == 0 or w == 0:
            return False
        aspect = w / float(h)
        if aspect < 2.0 or aspect > 6.5:
            return False
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_ratio = np.mean(edges > 0)
        return edge_ratio > 0.03

    @staticmethod
    def _contour_score(rect_w, rect_h, area, edge_density):
        aspect = rect_w / float(rect_h + 1e-6)
        aspect_score = 1.0 - min(abs(aspect - 4.0) / 4.0, 1.0)
        size_score = min(area / (250 * 80), 1.0)
        return 0.5 * aspect_score + 0.3 * size_score + 0.2 * min(edge_density / 0.1, 1.0)

    def find_plate_dip(self, original_img):
        gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 60, 180)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        h_img, w_img = gray.shape[:2]

        best_score = 0
        best_pts = None

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < (w_img * h_img) * 0.002:
                continue
            rect = cv2.minAreaRect(cnt)
            (cx, cy), (w, h), angle = rect
            if w == 0 or h == 0:
                continue
            rect_w, rect_h = max(w, h), min(w, h)
            aspect = rect_w / float(rect_h + 1e-6)
            if aspect < 2.0 or aspect > 6.5:
                continue

            box = cv2.boxPoints(rect)
            box = np.int32(box)
            x, y, bw, bh = cv2.boundingRect(box)
            x2, y2 = x + bw, y + bh
            x, y = max(0, x), max(0, y)
            x2, y2 = min(w_img, x2), min(h_img, y2)
            crop = gray[y:y2, x:x2]
            if crop.size == 0:
                continue
            edge_density = np.mean(cv2.Canny(crop, 50, 150) > 0)
            score = self._contour_score(rect_w, rect_h, area, edge_density)

            if score > best_score:
                best_score = score
                best_pts = box

        if best_pts is None:
            return None

        return best_pts

    def find_plate_hybrid(self, original_img):
        pts = self.find_plate_direct(original_img)
        if pts is not None and self.validate_plate_candidate(original_img, pts):
            return pts, "YOLO"

        dip_pts = self.find_plate_dip(original_img)
        if dip_pts is not None and self.validate_plate_candidate(original_img, dip_pts):
            return dip_pts, "DIP"

        return None, None

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