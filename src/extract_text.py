import pdfplumber
import pytesseract
from pdf2image import convert_from_path
import os
import re

def clean_text(text):
    """Cleans extracted text by removing references, special characters, and excessive spaces."""
    text = re.sub(r'\(.*?\d{4}.*?\)', '', text)  # Remove references like (Smith, 2000)
    text = re.sub(r'[^a-zA-Z0-9.,!?;:\s\n]', '', text)  # Remove special characters
    text = re.sub(r'\s+', ' ', text).strip()  # Normalize whitespace
    return text


def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF, using OCR if necessary."""
    extracted_text = ""

    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:  
                    extracted_text += text + "\n"
                else:  # If no text is found, use OCR
                    print(f"OCR needed for page {page.page_number}.")
                    images = convert_from_path(pdf_path, first_page=page.page_number, last_page=page.page_number)
                    for image in images:
                        extracted_text += pytesseract.image_to_string(image) + "\n"
    except Exception as e:
        print(f"Error processing PDF: {e}")

    return extracted_text.strip()