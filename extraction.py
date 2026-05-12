from paddleocr import PaddleOCR
import csv
import os
from datetime import datetime

class DataExtractor:
    def __init__(self):
        # Initialize PaddleOCR
        # use_angle_cls=True helps if the text is slightly tilted
        # show_log=False stops PaddleOCR from spamming your terminal
        self.ocr = PaddleOCR(use_angle_cls=True, lang='en')

    def extract_plate_text(self, img):
        """
        Unpacks PaddleOCR's complex nested list to extract just the text.
        """
        # Run the OCR on the enhanced image
        results = self.ocr.ocr(img)
        
        # If OCR completely fails to find anything, return a default message
        if not results or not results[0]:
            return ""
            
        extracted_texts = []
        
        # PaddleOCR returns data in this format: 
        # [ [[(x1,y1), (x2,y2)...], ('Detected Text', confidence_score)], ... ]
        for line in results[0]:
            # Navigate the nested list to grab just the string
            text = line[1][0] 
            extracted_texts.append(text)
            
        # Join all found text with a space (e.g., "PUNJAB" + " " + "AGQ 6360")
        final_text = " ".join(extracted_texts).strip()
        
        return final_text

    def save_to_csv(self, text, file_path="extracted_plates.csv"):
        """
        Saves the text and a timestamp to the exact folder specified by app.py.
        """
        # Don't save empty reads to the database
        if not text or text.strip() == "":
            return 
            
        # Ensure the directory exists (a safety net)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Check if the file already exists so we know whether to write headers
        file_exists = os.path.isfile(file_path)
        
        # Open the file in 'append' mode ('a')
        with open(file_path, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # If it's a brand new file, write the column headers first
            if not file_exists:
                writer.writerow(["Timestamp", "Extracted_Text"])
                
            # Write the current time and the plate text
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            writer.writerow([timestamp, text])