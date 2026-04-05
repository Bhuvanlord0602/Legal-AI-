import pdfplumber
import pytesseract
from PIL import Image
import re

def extract_text_from_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text

def extract_text_from_image(file):
    return pytesseract.image_to_string(Image.open(file), lang='kan')

def extract_entities(text):
    dates = re.findall(r'\d{2}/\d{2}/\d{4}', text)
    return {"dates": dates}

def summarize_text(text):
    return text[:300]  # simple baseline