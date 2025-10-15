import pdfplumber
import pytesseract
import cv2
import os

PDF_PATH = "行政制度汇编_带图表.pdf"
OUTPUT_DIR = "output_images"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Extract tables and text
with pdfplumber.open(PDF_PATH) as pdf:
    for i, page in enumerate(pdf.pages):
        # Extract tables
        tables = page.extract_tables()
        for j, table in enumerate(tables):
            print(f"Page {i+1} - Table {j+1}:")
            for row in table:
                print(row)
            print("-" * 40)
        # Extract text
        text = page.extract_text()
        if text:
            print(f"Page {i+1} Text:\n{text}\n{'='*40}")

        # Extract images
        for k, img in enumerate(page.images):
            bbox = (img['x0'], img['top'], img['x1'], img['bottom'])
            cropped = page.crop(bbox).to_image(resolution=300)
            img_path = os.path.join(OUTPUT_DIR, f"page_{i+1}_img_{k+1}.png")
            cropped.save(img_path, format="PNG")
            # OCR on image
            image_cv = cv2.imread(img_path)
            gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
            text_img = pytesseract.image_to_string(gray)
            print(f"OCR Text from Page {i+1} Image {k+1}:\n{text_img}\n{'='*40}")