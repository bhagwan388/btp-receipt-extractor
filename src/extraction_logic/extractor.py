# from ultralytics import YOLO
# import easyocr
# import cv2
# from pathlib import Path
# from collections import defaultdict
# import re
# import json

# # --- Initialization ---
# try:
#     reader = easyocr.Reader(['en'], gpu=True)
#     print("EasyOCR is using GPU.")
# except Exception:
#     print("GPU not available for EasyOCR, falling back to CPU.")
#     reader = easyocr.Reader(['en'], gpu=False)

# # --- IMPORTANT: Point this path to your text detection model ---
# MODEL_PATH = Path("results/models/best_text_detector.pt")

# # --- Post-Processing Helper Functions ---

# def clean_price(text: str) -> str:
#     """
#     Cleans raw OCR text for a price, finding the most likely monetary value.
#     - Replaces common OCR errors (O/o -> 0, S/s -> 5, B -> 8).
#     - Uses a robust regex to find number sequences that look like prices.
#     """
#     if not isinstance(text, str):
#         return ""
        
#     # Replace common OCR errors for numbers
#     text = text.replace('o', '0').replace('O', '0')
#     text = text.replace('s', '5').replace('S', '5')
#     text = text.replace('B', '8')
    
#     # Regex to find number-like strings (e.g., "4.20", "250.00", "3.00")
#     matches = re.findall(r'(\d+[\s.,]\d+)', text)
    
#     if not matches:
#         # If no decimal match, find any number
#         matches = re.findall(r'(\d+)', text)
#         if not matches:
#             return ""

#     # Find the most number-like string (usually the longest or last one)
#     target_string = matches[-1]
    
#     # Remove all non-digit characters to get the final number
#     just_digits = re.sub(r'\D', '', target_string)
    
#     return just_digits

# def clean_text(text: str) -> str:
#     """Cleans general text by normalizing whitespace."""
#     if not isinstance(text, str):
#         return ""
#     text = re.sub(r'[^A-Za-z0-9\s.,%-]+', '', text)
#     return " ".join(text.split()).strip()

# # --- NEW: Function to reconstruct lines ---
# def reconstruct_lines(ocr_results):
#     """
#     Groups OCR results into lines based on their vertical position.
#     ocr_results is a list of tuples: (box, text)
#     box = [[x1,y1], [x2,y1], [x2,y2], [x1,y2]]
#     """
#     if not ocr_results:
#         return []

#     # Sort boxes primarily by their top-Y coordinate (top-to-bottom)
#     ocr_results.sort(key=lambda res: res[0][0][1])

#     lines = []
#     current_line = []
    
#     for res in ocr_results:
#         box, text = res
#         y_center = (box[0][1] + box[2][1]) / 2 # Y-center of the box

#         if not current_line:
#             current_line.append(res)
#         else:
#             # Check if current box belongs to the current line
#             # Get the Y-center of the last box in the current line
#             last_box, _ = current_line[-1]
#             last_y_center = (last_box[0][1] + last_box[2][1]) / 2
            
#             # Estimate the height of the current box
#             box_height = abs(box[2][1] - box[0][1])
            
#             # If Y-centers are close (e.g., within half a box height), consider it the same line
#             if abs(y_center - last_y_center) < (box_height * 0.7):
#                 current_line.append(res)
#             else:
#                 # New line detected, save the old line
#                 lines.append(current_line)
#                 current_line = [res]
    
#     # Add the last line
#     if current_line:
#         lines.append(current_line)

#     # Now, sort each line by X coordinate and join the text
#     reconstructed_text_lines = []
#     for line in lines:
#         # Sort boxes in the line by their X coordinate (left-to-right)
#         line.sort(key=lambda res: res[0][0][0])
#         # Join the text of all boxes in this line with appropriate spacing
#         line_text = "  ".join([text for _, text in line])
#         reconstructed_text_lines.append(line_text)
        
#     return reconstructed_text_lines

# # --- Main Extraction Function ---

# def extract_receipt_info(image_path: str, annotated_image_save_path: str) -> tuple:
#     """
#     Detects all text, extracts it, parses it, and saves an annotated image.
#     Returns: (parsed_data_dict, annotated_image_path, reconstructed_lines_list)
#     """
#     if not MODEL_PATH.exists():
#         return ({"error": f"Model not found at {MODEL_PATH}"}, None, None)

#     model = YOLO(MODEL_PATH)
#     results = model.predict(image_path, conf=0.25)
    
#     image = cv2.imread(image_path)
#     if image is None:
#         return ({"error": f"Image not found at {image_path}"}, None, None)

#     all_boxes = []
#     annotated_image = image.copy()
    
#     for result in results:
#         for box in result.boxes:
#             coords = box.xyxy[0].cpu().numpy().astype(int)
#             all_boxes.append(coords)
#             x1, y1, x2, y2 = coords
#             cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2) # Green box

#     cv2.imwrite(annotated_image_save_path, annotated_image)
    
#     # --- 3. Apply OCR and get text for each box ---
#     # We need the box and text together for line reconstruction
#     ocr_results_with_boxes = []
#     for box in all_boxes:
#         x1, y1, x2, y2 = box
#         # Use a small padding for better OCR
#         y1_pad, y2_pad = max(0, y1-5), min(image.shape[0], y2+5)
#         x1_pad, x2_pad = max(0, x1-5), min(image.shape[1], x2+5)
#         cropped_image = image[y1_pad:y2_pad, x1_pad:x2_pad]
        
#         # We need the full detail from EasyOCR
#         ocr_results = reader.readtext(cropped_image, detail=1)
        
#         if ocr_results:
#             for (box_coords, text, conf) in ocr_results:
#                 # Map the relative box coordinates back to the original image coordinates
#                 original_box_coords = [
#                     [int(box_coords[0][0] + x1_pad), int(box_coords[0][1] + y1_pad)], # top-left
#                     [int(box_coords[1][0] + x1_pad), int(box_coords[1][1] + y1_pad)], # top-right
#                     [int(box_coords[2][0] + x1_pad), int(box_coords[2][1] + y1_pad)], # bottom-right
#                     [int(box_coords[3][0] + x1_pad), int(box_coords[3][1] + y1_pad)]  # bottom-left
#                 ]
#                 ocr_results_with_boxes.append((original_box_coords, text))
    
#     # --- 4. Reconstruct the bill layout ---
#     reconstructed_lines = reconstruct_lines(ocr_results_with_boxes)
    
#     # --- 5. Parse the reconstructed text ---
#     final_structured_data = parse_extracted_text(reconstructed_lines)

#     return (final_structured_data, annotated_image_save_path, reconstructed_lines)


# # --- SMARTER PARSING FUNCTION V2.0 ---
# def parse_extracted_text(text_list: list) -> dict:
#     """
#     Parses a *reconstructed list* of text to find key information.
#     """
#     parsed_data = {
#         "company": "Not Found",
#         "date": "Not Found",
#         "items": [],
#         "total": "Not Found"
#     }
    
#     # --- Regex Patterns ---
#     date_pattern = re.compile(r'(\d{2}[-/]\d{2}[-/]\d{2,4})|(\d{1,2}\s+[A-Za-z]{3,}\s+\d{4})')
#     # This pattern looks for a line that starts with text and ends with a price
#     item_pattern = re.compile(r'^(.*?)\s+([\d,.]+)$')
#     # This pattern looks for a quantity, a name, and a price
#     qty_item_pattern = re.compile(r'^\d+\s+[Xx]?\s+(.*?)\s+([\d,.]+)$')

#     # --- Company Logic ---
#     if text_list:
#         for line in text_list:
#             # Find the first "important" looking line that isn't just noise
#             line_upper = line.upper()
#             if len(line_upper) > 3 and "TEL" not in line_upper and "JALAN" not in line_upper and "INVOICE" not in line_upper and "GST" not in line_upper:
#                 parsed_data["company"] = clean_text(line)
#                 break
    
#     # --- Line-by-Line Parsing ---
#     found_total = False
#     for i, text in enumerate(text_list):
#         text_upper = text.upper()
        
#         # Find Date
#         if parsed_data["date"] == "Not Found":
#             match = date_pattern.search(text)
#             if match:
#                 parsed_data["date"] = match.group(0)
        
#         # Find Total (Prioritize this)
#         if "TOTAL" in text_upper:
#             price = clean_price(text)
#             if price and price != "0":
#                 parsed_data["total"] = price
#                 found_total = True
#             elif (i + 1) < len(text_list) and not found_total: # Check next line
#                 next_line_text = text_list[i+1]
#                 price = clean_price(next_line_text)
#                 if price and price != "0":
#                     parsed_data["total"] = price
#                     found_total = True
        
#         # Find Subtotal (as a fallback for total)
#         if "SUBTOTAL" in text_upper and not found_total:
#             price = clean_price(text)
#             if price and price != "0":
#                 parsed_data["total"] = price # Use subtotal if no final total is found

#         # Find Line Items
#         item_match = item_pattern.search(text)
#         qty_item_match = qty_item_pattern.search(text)

#         if qty_item_match:
#             try:
#                 name = qty_item_match.group(1)
#                 price = qty_item_match.group(2)
#                 parsed_data["items"].append({
#                     "name": clean_text(name),
#                     "price": clean_price(price)
#                 })
#             except Exception: pass
#         elif item_match:
#             try:
#                 name = item_match.group(1)
#                 price = item_match.group(2)
#                 # Avoid capturing lines that are part of the summary
#                 if "TOTAL" not in name.upper() and "CASH" not in name.upper() and "CHANGE" not in name.upper():
#                     parsed_data["items"].append({
#                         "name": clean_text(name),
#                         "price": clean_price(price)
#                     })
#             except Exception: pass

#     return parsed_data


# from ultralytics import YOLO
# import easyocr
# import cv2
# from pathlib import Path
# from collections import defaultdict
# import re
# import json

# # --- Initialization ---
# try:
#     # Attempt to use the GPU; fallback to CPU if it fails
#     reader = easyocr.Reader(['en'], gpu=True)
#     print("EasyOCR is using GPU.")
# except Exception:
#     print("GPU not available for EasyOCR, falling back to CPU.")
#     reader = easyocr.Reader(['en'], gpu=False)

# # --- IMPORTANT: Point this path to your TEXT detection model ---
# MODEL_PATH = Path("results/models/best_text_detector.pt")

# # --- Post-Processing Helper Functions ---

# def clean_price(text: str) -> str:
#     """
#     Cleans raw OCR text for a price, finding the most likely monetary value.
#     - Replaces common OCR errors (O/o -> 0, S/s -> 5, B -> 8).
#     - Uses a robust regex to find number sequences that look like prices.
#     """
#     if not isinstance(text, str):
#         return ""
        
#     # --- NEW: More robust cleaning rules ---
#     text = text.replace('o', '0').replace('O', '0')
#     text = text.replace('s', '5').replace('S', '5')
#     text = text.replace('B', '8')
    
#     # Regex to find number-like strings (e.g., "4.20", "250.00", "3.00", "1,45")
#     # This pattern looks for digits separated by dots, commas, or spaces, ending in digits
#     matches = re.findall(r'(\d+[\s.,]+\d{2}\b)', text)
    
#     if not matches:
#         # If no decimal/comma match, find the largest number-like string
#         matches = re.findall(r'([\d,.]+)', text)
#         if not matches:
#             return ""

#     # Find the most likely price (e.g., the one with the most digits)
#     target_string = max(matches, key=lambda m: len(re.sub(r'\D', '', m)))
    
#     # Remove all non-digit characters to get the final number
#     just_digits = re.sub(r'\D', '', target_string)
    
#     return just_digits

# def clean_text(text: str) -> str:
#     """Cleans general text by normalizing whitespace."""
#     if not isinstance(text, str):
#         return ""
#     text = re.sub(r'[^A-Za-z0-9\s.,%-]+', '', text)
#     return " ".join(text.split()).strip()

# # --- NEW: More Robust Line Reconstruction Function ---
# def reconstruct_lines(ocr_results):
#     """
#     Groups OCR results into lines based on their vertical position.
#     ocr_results is a list of tuples: (box, text)
#     box = [[x1,y1], [x2,y1], [x2,y2], [x1,y2]]
#     """
#     if not ocr_results:
#         return []

#     # Sort boxes primarily by their top-Y coordinate (top-to-bottom)
#     ocr_results.sort(key=lambda res: res[0][0][1])

#     lines = []
#     current_line = []
    
#     for res in ocr_results:
#         box, text = res
#         y_center = (box[0][1] + box[2][1]) / 2 # Y-center of the box

#         if not current_line:
#             current_line.append(res)
#         else:
#             # Get the Y-center of the last box in the current line
#             last_box, _ = current_line[-1]
#             last_y_center = (last_box[0][1] + last_box[2][1]) / 2
            
#             # Estimate the height of the current box
#             box_height = abs(box[2][1] - box[0][1])
            
#             # If Y-centers are close (e.g., within 70% of a box height), consider it the same line
#             if abs(y_center - last_y_center) < (box_height * 0.7):
#                 current_line.append(res)
#             else:
#                 # New line detected, save the old line
#                 lines.append(current_line)
#                 current_line = [res]
    
#     # Add the last line
#     if current_line:
#         lines.append(current_line)

#     # Now, sort each line by X coordinate and join the text
#     reconstructed_text_lines = []
#     for line in lines:
#         # Sort boxes in the line by their X coordinate (left-to-right)
#         line.sort(key=lambda res: res[0][0][0])
#         # Join the text of all boxes in this line with appropriate spacing
#         line_text = "  ".join([text for _, text in line])
#         reconstructed_text_lines.append(line_text)
        
#     return reconstructed_text_lines

# # --- Main Extraction Function ---
# def extract_receipt_info(image_path: str, annotated_image_save_path: str) -> tuple:
#     """
#     Detects all text, extracts it, parses it, and saves an annotated image.
#     Returns: (parsed_data_dict, annotated_image_path, reconstructed_lines_list)
#     """
#     if not MODEL_PATH.exists():
#         return ({"error": f"Model not found at {MODEL_PATH}"}, None, None)

#     model = YOLO(MODEL_PATH)
#     results = model.predict(image_path, conf=0.25)
    
#     image = cv2.imread(image_path)
#     if image is None:
#         return ({"error": f"Image not found at {image_path}"}, None, None)

#     all_boxes = []
#     annotated_image = image.copy()
    
#     # --- NEW: Get full OCR results (boxes + text) for line reconstruction ---
#     ocr_results_with_boxes = [] 

#     for result in results:
#         for box in result.boxes:
#             coords = box.xyxy[0].cpu().numpy().astype(int)
#             x1, y1, x2, y2 = coords
#             all_boxes.append(coords)
            
#             # Draw the box on the annotated image
#             cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2) # Green box
            
#             # --- Perform OCR on this specific box ---
#             # Add padding for better OCR
#             y1_pad, y2_pad = max(0, y1-5), min(image.shape[0], y2+5)
#             x1_pad, x2_pad = max(0, x1-5), min(image.shape[1], x2+5)
#             cropped_image = image[y1_pad:y2_pad, x1_pad:x2_pad]
            
#             # We need the full detail from EasyOCR (detail=1)
#             ocr_results = reader.readtext(cropped_image, detail=1)
            
#             if ocr_results:
#                 for (box_coords, text, conf) in ocr_results:
#                     # Map the relative box coordinates back to the original image coordinates
#                     original_box_coords = [
#                         [int(box_coords[0][0] + x1_pad), int(box_coords[0][1] + y1_pad)], # top-left
#                         [int(box_coords[1][0] + x1_pad), int(box_coords[1][1] + y1_pad)], # top-right
#                         [int(box_coords[2][0] + x1_pad), int(box_coords[2][1] + y1_pad)], # bottom-right
#                         [int(box_coords[3][0] + x1_pad), int(box_coords[3][1] + y1_pad)]  # bottom-left
#                     ]
#                     ocr_results_with_boxes.append((original_box_coords, text))

#     # Save the image with all detected boxes
#     cv2.imwrite(annotated_image_save_path, annotated_image)
    
#     # --- 4. Reconstruct the bill layout ---
#     reconstructed_lines = reconstruct_lines(ocr_results_with_boxes)
    
#     # --- 5. Parse the reconstructed text ---
#     final_structured_data = parse_extracted_text(reconstructed_lines)

#     return (final_structured_data, annotated_image_save_path, reconstructed_lines)


# # --- SMARTER PARSING FUNCTION V3.0 ---
# def parse_extracted_text(text_list: list) -> dict:
#     """
#     Parses a *reconstructed list* of text to find key information.
#     """
#     parsed_data = {
#         "company": "Not Found",
#         "date": "Not Found",
#         "items": [],
#         "total": "Not Found"
#     }
    
#     # --- Regex Patterns ---
#     # More robust date pattern: matches dd/mm/yyyy, dd-mm-yy, dd-mm-yyyy, dd [Month] yyyy
#     date_pattern = re.compile(r'(\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\d{1,2}\s+[A-Za-z]{3,}\s+\d{4})')
#     # This pattern looks for a line that starts with a quantity, has text, and ends with a price
#     # e.g., "1  WALK  1.45" or "1 100PLUS LIME 325ML 1.45"
#     item_pattern = re.compile(r'^\d+\s+[A-Za-z].*?\s+([\d,.]+)$')
    
#     if text_list:
#         # Assume first non-empty, "important" line is the company
#         for line in text_list:
#             if len(line) > 3 and "GST" not in line.upper() and "INVOICE" not in line.upper():
#                 parsed_data["company"] = clean_text(line)
#                 break
    
#     found_total = False
    
#     for i, text in enumerate(text_list):
#         text_upper = text.upper()
        
#         # --- Find Date ---
#         if parsed_data["date"] == "Not Found":
#             match = date_pattern.search(text)
#             if match:
#                 parsed_data["date"] = match.group(0)
        
#         # --- Find Total ---
#         # Look for the last instance of "TOTAL"
#         if "TOTAL" in text_upper:
#             price = clean_price(text)
#             if price and price != "0":
#                 parsed_data["total"] = price
#                 found_total = True
#             # Check the next line if the current line is just the word "TOTAL"
#             elif (i + 1) < len(text_list) and not found_total:
#                 next_line_text = text_list[i+1]
#                 price = clean_price(next_line_text)
#                 if price and price != "0":
#                     parsed_data["total"] = price
#                     found_total = True
        
#         # --- Find Items ---
#         item_match = item_pattern.search(text)
#         if item_match:
#             try:
#                 # Extract the price (the last group)
#                 price = item_match.group(1)
#                 # Assume everything before the price is the item name
#                 name_part = text.rsplit(price, 1)[0].strip()
#                 # Remove the quantity from the beginning of the name
#                 name = re.sub(r'^\d+\s+[Xx]?\s*', '', name_part)
                
#                 # Filter out lines that are not items
#                 if "TOTAL" not in name.upper() and "CASH" not in name.upper() and "CHANGE" not in name.upper() and "GST" not in name_upper:
#                     parsed_data["items"].append({
#                         "name": clean_text(name),
#                         "price": clean_price(price)
#                     })
#             except Exception as e:
#                 print(f"Error parsing item: {e}")

#     # Fallback for Total if not found with keyword (e.g., last line is often the total)
#     if not found_total and text_list:
#         parsed_data["total"] = clean_price(text_list[-1])

#     return parsed_data
from ultralytics import YOLO
import easyocr
import cv2
from pathlib import Path
from collections import defaultdict
import re
import json

# --- Initialization ---
# Initialize the EasyOCR reader once for efficiency.
try:
    reader = easyocr.Reader(['en'], gpu=True)
    print("EasyOCR is using GPU.")
except Exception:
    print("GPU not available for EasyOCR, falling back to CPU.")
    reader = easyocr.Reader(['en'], gpu=False)

# --- IMPORTANT: Point this path to your TEXT detection model ---
MODEL_PATH = Path("results/models/best_text_detector.pt")

# --- Post-Processing Helper Functions ---

def clean_price(text: str) -> str:
    """
    Cleans raw OCR text for a price, finding the most likely monetary value.
    - Replaces common OCR errors (O/o -> 0, S/s -> 5, B -> 8).
    - Uses a robust regex to find number sequences that look like prices.
    """
    if not isinstance(text, str):
        return ""
        
    # Replace common OCR errors for numbers
    text = text.replace('o', '0').replace('O', '0')
    text = text.replace('s', '5').replace('S', '5')
    text = text.replace('B', '8')
    
    # Regex to find number-like strings (e.g., "4.20", "250.00", "3.00", "1,45")
    # This pattern looks for digits separated by dots, commas, or spaces, ending in digits
    matches = re.findall(r'(\d+[\s.,]+\d{2}\b)', text)
    
    if not matches:
        # If no decimal match, find any number
        matches = re.findall(r'([\d,.]+)', text)
        if not matches:
            return ""

    # Find the most number-like string (usually the last one)
    target_string = matches[-1]
    
    # Remove all non-digit characters to get the final number
    just_digits = re.sub(r'\D', '', target_string)
    
    return just_digits

def clean_text(text: str) -> str:
    """Cleans general text by normalizing whitespace."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r'[^A-Za-z0-9\s.,%-]+', '', text)
    return " ".join(text.split()).strip()

# --- Line Reconstruction ---
def reconstruct_lines(ocr_results):
    """
    Groups OCR results into lines based on their vertical position.
    ocr_results is a list of tuples: (box, text)
    box = [[x1,y1], [x2,y1], [x2,y2], [x1,y2]]
    """
    if not ocr_results:
        return []

    # Sort boxes primarily by their top-Y coordinate (top-to-bottom)
    ocr_results.sort(key=lambda res: res[0][0][1])

    lines = []
    current_line = []
    
    for res in ocr_results:
        box, text = res
        y_center = (box[0][1] + box[2][1]) / 2 # Y-center of the box

        if not current_line:
            current_line.append(res)
        else:
            # Get the Y-center of the last box in the current line
            last_box, _ = current_line[-1]
            last_y_center = (last_box[0][1] + last_box[2][1]) / 2
            
            # Estimate the height of the current box
            box_height = abs(box[2][1] - box[0][1])
            
            # If Y-centers are close (within 70% of a box height), consider it the same line
            if abs(y_center - last_y_center) < (box_height * 0.7):
                current_line.append(res)
            else:
                # New line detected, save the old line
                lines.append(current_line)
                current_line = [res]
    
    # Add the last line
    if current_line:
        lines.append(current_line)

    # Now, sort each line by X coordinate and join the text
    reconstructed_text_lines = []
    for line in lines:
        # Sort boxes in the line by their X coordinate (left-to-right)
        line.sort(key=lambda res: res[0][0][0])
        # Join the text of all boxes in this line with appropriate spacing
        line_text = "  ".join([text for _, text in line])
        reconstructed_text_lines.append(line_text)
        
    return reconstructed_text_lines

# --- Main Extraction Function ---
def extract_receipt_info(image_path: str, annotated_image_save_path: str) -> tuple:
    """
    Detects all text, extracts it, parses it, and saves an annotated image.
    Returns: (parsed_data_dict, annotated_image_path, reconstructed_lines_list)
    """
    if not MODEL_PATH.exists():
        return ({"error": f"Model not found at {MODEL_PATH}"}, None, None)

    model = YOLO(MODEL_PATH)
    results = model.predict(image_path, conf=0.25, verbose=False)
    
    image = cv2.imread(image_path)
    if image is None:
        return ({"error": f"Image not found at {image_path}"}, None, None)

    annotated_image = image.copy()
    
    # --- Get full OCR results (boxes + text) for line reconstruction ---
    ocr_results_with_boxes = [] 

    for result in results:
        for box in result.boxes:
            coords = box.xyxy[0].cpu().numpy().astype(int)
            x1, y1, x2, y2 = coords
            
            # Draw the box on the annotated image
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2) # Green box
            
            # --- Perform OCR on this specific box ---
            # Add padding for better OCR
            y1_pad, y2_pad = max(0, y1-5), min(image.shape[0], y2+5)
            x1_pad, x2_pad = max(0, x1-5), min(image.shape[1], x2+5)
            cropped_image = image[y1_pad:y2_pad, x1_pad:x2_pad]
            
            # We need the full detail from EasyOCR (detail=1)
            ocr_results = reader.readtext(cropped_image, detail=1)
            
            if ocr_results:
                for (box_coords, text, conf) in ocr_results:
                    # Map the relative box coordinates back to the original image coordinates
                    original_box_coords = [
                        [int(box_coords[0][0] + x1_pad), int(box_coords[0][1] + y1_pad)], # top-left
                        [int(box_coords[1][0] + x1_pad), int(box_coords[1][1] + y1_pad)], # top-right
                        [int(box_coords[2][0] + x1_pad), int(box_coords[2][1] + y1_pad)], # bottom-right
                        [int(box_coords[3][0] + x1_pad), int(box_coords[3][1] + y1_pad)]  # bottom-left
                    ]
                    ocr_results_with_boxes.append((original_box_coords, text))

    # Save the image with all detected boxes
    cv2.imwrite(annotated_image_save_path, annotated_image)
    
    # --- 4. Reconstruct the bill layout ---
    reconstructed_lines = reconstruct_lines(ocr_results_with_boxes)
    
    # --- 5. Parse the reconstructed text ---
    final_structured_data = parse_extracted_text(reconstructed_lines)

    return (final_structured_data, annotated_image_save_path, reconstructed_lines)


# --- SMARTER PARSING FUNCTION V4.0 ---
def parse_extracted_text(text_list: list) -> dict:
    """
    Parses a *reconstructed list* of text to find key information.
    """
    parsed_data = {"company": "Not Found", "date": "Not Found", "items": [], "total": "Not Found"}
    
    # --- NEW: More robust regex patterns ---
    date_pattern = re.compile(
        r'(\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\d{1,2}\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4})', 
        re.IGNORECASE
    )
    # This pattern looks for a line that starts with a quantity, has text, and ends with a price
    item_pattern = re.compile(r'^\d+\s+[A-Za-z].*?\s+([\d,.]+)$')
    total_keywords = ['TOTAL', 'AMOUNT DUE', 'AMT', 'TOTAL AMOUNT']

    # --- NEW: Smarter Company Logic ---
    if text_list:
        for line in text_list[:4]: # Check first 4 lines
            line_upper = line.upper()
            # Find the first "important" looking line that isn't just noise
            if line and len(line) > 3 and not any(kw in line_upper for kw in ['GST', 'INVOICE', 'TEL', 'DATE', 'RECEIPT']):
                parsed_data["company"] = clean_text(line)
                break
        if parsed_data["company"] == "Not Found":
             parsed_data["company"] = clean_text(text_list[0]) # Fallback

    found_total = False
    
    for i, text in enumerate(text_list):
        text_upper = text.upper()
        
        # --- Find Date (using new pattern) ---
        if parsed_data["date"] == "Not Found":
            match = date_pattern.search(text)
            if match:
                parsed_data["date"] = match.group(0)
        
        # --- Find Total (using new keywords) ---
        if any(kw in text_upper for kw in total_keywords):
            price = clean_price(text)
            if price and price != "0":
                parsed_data["total"] = price
                found_total = True
            # Check the next line if the current line is just the word "TOTAL"
            elif (i + 1) < len(text_list) and not found_total: 
                price = clean_price(text_list[i+1])
                if price and price != "0":
                    parsed_data["total"] = price
                    found_total = True

        # --- Find Items (with better filtering) ---
        item_match = item_pattern.search(text)
        if item_match:
            try:
                price_str = item_match.group(1)
                name_part = text.rsplit(price_str, 1)[0].strip()
                # Remove the quantity from the beginning of the name
                name = re.sub(r'^\d+\s+[Xx]?\s*', '', name_part)
                
                # Filter out junk lines
                if not any(kw in name.upper() for kw in ['TOTAL', 'CASH', 'CHANGE', 'GST', 'SUBTOTAL']):
                    parsed_data["items"].append({
                        "name": clean_text(name),
                        "price": clean_price(price_str)
                    })
            except Exception as e:
                print(f"Error parsing item: {e}")

    # --- Fallback for Total ---
    # If no "TOTAL" keyword found, check last lines for cash/change
    if not found_total and len(text_list) > 2:
        if "CASH" in text_list[-2].upper():
             parsed_data["total"] = clean_price(text_list[-2])
        elif "TOTAL" in text_list[-1].upper():
             parsed_data["total"] = clean_price(text_list[-1])

    return parsed_data