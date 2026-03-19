import os
from nv_ingest import Ingestor
from nv_ingest.schema import Document
from PIL import Image
from pdf2image import convert_from_path
import pytesseract
import cv2
import numpy as np


# -------------------------------
# CONFIG
# -------------------------------
INPUT_FILE = "/mnt/data/SECOND FLOOR-1-1.pdf"
OUTPUT_TEXT_FILE = "extracted_text.txt"
TEMP_IMG_DIR = "temp_pages"
USE_GPU = True


# -------------------------------
# HELPER: Convert PDF to Images
# -------------------------------
def pdf_to_images(pdf_path):
    os.makedirs(TEMP_IMG_DIR, exist_ok=True)
    images = convert_from_path(pdf_path, dpi=300)

    image_paths = []
    for i, img in enumerate(images):
        path = os.path.join(TEMP_IMG_DIR, f"page_{i}.png")
        img.save(path, "PNG")
        image_paths.append(path)

    return image_paths


# -------------------------------
# HELPER: Preprocess image for OCR
# -------------------------------
def preprocess_image(img_path):
    img = cv2.imread(img_path)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Improve text visibility in architectural drawings
    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )

    return thresh


# -------------------------------
# OCR using Tesseract fallback
# -------------------------------
def fallback_ocr(img_path):
    processed = preprocess_image(img_path)
    text = pytesseract.image_to_string(processed)
    return text


# -------------------------------
# NV-INGEST PIPELINE
# -------------------------------
def run_nv_ingest(image_paths):
    ingestor = Ingestor(
        enable_ocr=True,
        enable_layout=True,   # critical for drawings
        device="cuda" if USE_GPU else "cpu"
    )

    documents = []
    for img_path in image_paths:
        doc = Document.from_file(img_path)
        documents.append(doc)

    results = ingestor.ingest(documents)

    extracted_text = []

    for res in results:
        if res.text and len(res.text.strip()) > 10:
            extracted_text.append(res.text)
        else:
            # fallback OCR if NV-Ingest fails (common in drawings)
            fallback = fallback_ocr(res.source)
            extracted_text.append(fallback)

    return "\n\n".join(extracted_text)


# -------------------------------
# MAIN PIPELINE
# -------------------------------
def extract_text(file_path):
    if file_path.lower().endswith(".pdf"):
        image_paths = pdf_to_images(file_path)
    else:
        image_paths = [file_path]

    text = run_nv_ingest(image_paths)

    with open(OUTPUT_TEXT_FILE, "w") as f:
        f.write(text)

    print("✅ Extraction completed!")
    print(f"📄 Output saved to: {OUTPUT_TEXT_FILE}")


# -------------------------------
# RUN
# -------------------------------
if __name__ == "__main__":
    extract_text(INPUT_FILE)






Link-

https://chatgpt.com/share/69bba5f4-dd30-8000-9f5a-3f9933b71af8
