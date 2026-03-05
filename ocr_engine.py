import pytesseract
import numpy as np
from PIL import Image
from surya.models import load_predictors
from surya.inference import run_ocr

predictors = load_predictors()

def surya_ocr(image):
    try:
        result = run_ocr([np.array(image)], predictors)
        text = "\n".join([line["text"] for line in result[0]["lines"]])
        return text
    except Exception:
        return ""

def tesseract_ocr(image):
    return pytesseract.image_to_string(image)

def extract_text(image):
    text = surya_ocr(image)
    if len(text.strip()) < 20:
        text = tesseract_ocr(image)
    return text