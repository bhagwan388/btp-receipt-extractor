# **End-to-End Receipt Digitization and Extraction**

This is a B.Tech Project that implements a full-stack system to read and understand retail receipts from an image. The system uses a hybrid AI pipeline to first find all text on the receipt and then intelligently parse that text to extract key information.

This project uses the **SROIE dataset** and a 2-step **"Scanner-Accountant"** approach:

1. **The "Scanner" (YOLOv8):** An AI model trained to find every single line of text on a receipt (96.4% mAP accuracy).  
2. **The "Accountant" (Python Parser):** A smart Python script that takes the detected text, reconstructs the bill's layout, and uses rules and regular expressions (regex) to find and categorize the important fields like company, date, total, and line items.

## **🚀 Features**

* **High-Accuracy Text Detection:** Uses a YOLOv8 model trained on the SROIE dataset, achieving **96.4% mAP@50** for finding text.  
* **Intelligent Text Extraction:** Reconstructs the bill's original line-by-line layout from the detected bounding boxes.  
* **Smart Parsing:** Uses advanced regex in Python to find and categorize key information (company, date, total, items) from the raw text.  
* **Web Interface:** A simple, interactive web app built with **Streamlit** to upload receipts and view the results.  
* **End-to-End Evaluation:** Includes a script to measure the final accuracy of the entire pipeline (detection \+ parsing) on the unseen test set.

## **🛠️ System Architecture**

The project is built on a 2-step hybrid pipeline:

1. **YOLOv8 (Text Detector):** A YOLOv8n model is trained on the SROIE dataset for one task: detecting all text blocks (a single class: text).  
2. **EasyOCR (Text Recognizer):** This library is used to read the text inside each bounding box found by YOLO.  
3. **Python Parser (extractor.py):** This is the core logic.  
   * **Reconstructs Lines:** Groups text boxes based on their Y-coordinates to rebuild the original lines of the receipt.  
   * **Parses Data:** Applies a series of regular expressions and rules to the reconstructed text to find and categorize the key information.  
4. **Streamlit (app.py):** This provides a simple web front-end to interact with the system.

## **📂 Project Structure**

sroie\_btp/  
├── data/  
│   ├── raw\_sroie/          \# Original SROIE dataset files  
│   └── yolo\_format/        \# Processed data for YOLO training  
├── results/  
│   └── models/             \# Trained model files (e.g., best\_text\_detector.pt)  
├── src/  
│   ├── data\_preprocessing/ \# Scripts to convert SROIE to YOLO format  
│   ├── extraction\_logic/ \# The main "Accountant" (extractor.py)  
│   └── web\_app/          \# The Streamlit app (app.py)  
├── .gitignore              \# Tells Git to ignore data/ and model/  
├── data.yaml               \# YOLO dataset configuration  
├── README.md               \# This file  
├── requirements.txt        \# All Python dependencies  
└── run\_evaluation.py       \# Script to test the final system accuracy

## **🏁 How to Run This Project**

### **1\. Prerequisites**

* Python 3.10+  
* Conda (for environment management)  
* A trained model file (best\_text\_detector.pt) located in results/models/

### **2\. Installation**

1. **Clone the repository:**  
   git clone \[https://github.com/bhagwan388/btp-receipt-extractor.git\](https://github.com/bhagwan388/btp-receipt-extractor.git)  
   cd sroie\_btp

2. **Create and activate the Conda environment:**  
   conda create \-n sroie\_ocr python=3.10  
   conda activate sroie\_ocr

3. Install all required libraries:  
   Note: This installs the CPU-only version of PyTorch to save space. Your trained model will run on the CPU.  
   pip install \-r requirements.txt

### **3\. Run the Web Application 🚀**

This is the main command to run the project.

python \-m streamlit run src/web\_app/app.py

This will automatically open the application in your web browser. You can then upload a receipt image (e.g., from data/raw\_sroie/SROIE2019/test/img/) to see the results.
<img width="1408" height="759" alt="image" src="https://github.com/user-attachments/assets/94222f50-9f21-469d-9d7a-4cae8f08a85d" />


### **4\. (Optional) Run the Final Evaluation**

To get the final accuracy scores for the parser, run this command in your terminal:

python src/run\_evaluation.py  
