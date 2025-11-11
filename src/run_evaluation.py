import json
from pathlib import Path
from tqdm import tqdm
import sys

# --- Add project root to path to import our own modules ---
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

# --- Import our project's functions ---
from src.extraction_logic.extractor import extract_receipt_info, clean_price, clean_text

def load_ground_truth(gt_path, box_path, img_w, img_h):
    """
    Loads the ground truth data from the SROIE dataset.
    """
    try:
        with open(gt_path, 'r', encoding='utf-8') as f:
            gt_data = json.load(f)
        
        # Clean the ground truth text for a fair comparison
        gt_total = clean_price(gt_data.get("total", ""))
        gt_date = gt_data.get("date", "")
        gt_company = clean_text(gt_data.get("company", ""))
        
        return {
            "total": gt_total,
            "date": gt_date,
            "company": gt_company
        }
    except Exception as e:
        print(f"Error loading ground truth {gt_path}: {e}")
        return None

def run_evaluation():
    """
    Runs the full evaluation pipeline on the TEST set
    and prints a final accuracy report.
    """
    print("Starting evaluation on the TEST set...")
    
    base_path = Path.cwd()
    raw_data_path = base_path / "data" / "raw_sroie" / "SROIE2019" # Ensure this is your raw data folder
    
    # --- UPDATED: Point to the TEST images in the raw data folder ---
    test_images_path = raw_data_path / "test" / "img"
    
    # Get all test images
    test_images = list(test_images_path.glob("*.jpg"))
    if not test_images:
        print(f"Error: No test images found in {test_images_path}")
        return

    # --- Counters for our accuracy report ---
    total_files = len(test_images)
    correct_total = 0
    correct_date = 0
    correct_company = 0
    
    (base_path / "temp").mkdir(exist_ok=True)

    # --- Loop through every image in the TEST set ---
    for img_path in tqdm(test_images, desc="Evaluating Model"):
        # Define paths
        img_name = img_path.name
        
        # --- UPDATED: Point to the TEST ground truth ---
        gt_entities_path = raw_data_path / "test" / "entities" / (img_path.stem + ".txt")
        gt_box_path = raw_data_path / "test" / "box" / (img_path.stem + ".txt")
        temp_annotated_path = base_path / "temp" / f"annot_{img_name}"

        if not gt_entities_path.exists() or not gt_box_path.exists():
            print(f"Warning: Missing ground truth for {img_name}, skipping.")
            total_files -= 1
            continue
            
        # 1. Get Ground Truth (The "Correct Answer")
        gt_data = load_ground_truth(gt_entities_path, gt_box_path, 1000, 1000) # Size doesn't matter
        if gt_data is None:
            total_files -= 1
            continue

        # 2. Get Our Model's Prediction
        predicted_data, _, _ = extract_receipt_info(str(img_path), str(temp_annotated_path))

        # 3. Compare Prediction to Ground Truth
        if predicted_data.get("total") == gt_data.get("total"):
            correct_total += 1
            
        if predicted_data.get("date") == gt_data.get("date"):
            correct_date += 1
            
        pred_company = predicted_data.get("company", "PRED_NONE")
        if gt_data.get("company") and pred_company in gt_data.get("company"):
            correct_company += 1

    # --- 4. Calculate and Print the Final Report ---
    print("\n--- 🏁 End-to-End Evaluation Report (TEST SET) ---")
    print(f"Processed {total_files} test images.\n")
    
    total_accuracy = (correct_total / total_files) * 100 if total_files > 0 else 0
    date_accuracy = (correct_date / total_files) * 100 if total_files > 0 else 0
    company_accuracy = (correct_company / total_files) * 100 if total_files > 0 else 0

    print(f"🧾 Company Name Accuracy: {company_accuracy:.2f}% ({correct_company}/{total_files})")
    print(f"📅 Date Accuracy:         {date_accuracy:.2f}% ({correct_date}/{total_files})")
    print(f"💰 Total Amount Accuracy: {total_accuracy:.2f}% ({correct_total}/{total_files})")
    print("\n-----------------------------------------")

if __name__ == "__main__":
    run_evaluation()

