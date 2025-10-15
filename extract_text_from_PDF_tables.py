# 目标：从PDF中提取文本、表格（保留表格格式 None）和图像，图像专门存储在extracted_images目录下，并对图像进行OCR
# 使用 pdfplumber 进行PDF处理，pytesseract 进行OCR
# 支持中英文混合识别
# 结构化输出，便于后续RAG处理
# 
import pdfplumber
from pdf2image import convert_from_path
import pytesseract
import os
from pathlib import Path

def extract_text_and_images_from_pdf(pdf_path, output_image_dir="extracted_images"):
    # 创建图像输出目录
    os.makedirs(output_image_dir, exist_ok=True)

    all_text = []
    all_tables = []
    all_images = []

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            # === 1. 原始文本 ===
            text = page.extract_text()
            if text:
                all_text.append(f"[Page {page_num}]\n[Text]\n{text}")

            # === 2. 结构化表格（完全保留原始格式，包括 None）===
            tables = page.extract_tables()
            for i, table in enumerate(tables):
                all_tables.append({
                    "page": page_num,
                    "table_id": i + 1,
                    "data": table  # ✅ 直接返回原始结构，不清洗
                })
                # 可选：在文本中添加表格占位符
                all_text.append(f"[Table {i+1} on Page {page_num}]")

            # === 3. 图像提取 ===
            if page.images:
                for img_idx, img in enumerate(page.images):
                    # 构造文件名
                    img_filename = f"{Path(pdf_path).stem}_page{page_num}_img{img_idx+1}.png"
                    img_path = os.path.join(output_image_dir, img_filename)

                    # 裁剪并保存
                    bbox = (img["x0"], img["top"], img["x1"], img["bottom"])
                    cropped = page.crop(bbox)
                    cropped.to_image(resolution=150).save(img_path)

                    # OCR
                    try:
                        pil_img = cropped.to_image(resolution=150).original
                        # 可选：图像预处理
                        pil_img = pil_img.convert("L")  # 灰度
                        ocr_text = pytesseract.image_to_string(pil_img, lang='chi_sim+eng').strip()
                    except Exception as e:
                        ocr_text = f"(OCR Error: {e})"



                    all_images.append({
                        "page": page_num,
                        "image_id": img_idx + 1,
                        "path": img_path,
                        "ocr_text": ocr_text
                    })

                    all_text.append(f"[Image {img_idx+1}]\nSaved: {img_path}\n[OCR]\n{ocr_text}")

    # 合并所有文本块
    full_text = "\n\n".join(all_text) + "\n"

    return {
        "text": full_text,
        "tables": all_tables,    # ✅ 结构化表格，保留 None
        "images": all_images,
        "pdf_path": pdf_path
    }

if __name__ == "__main__":
    result = extract_text_and_images_from_pdf("行政制度汇编_带图表.pdf")

    print(result["text"])  # 可读文本

    print("\n" + "="*60)
    print("📋 STRUCTURED TABLES (Preserving None):")
    for tbl in result["tables"]:
        print(f"Page {tbl['page']} - Table {tbl['table_id']}:")
        for row in tbl["data"]:
            print(row)  # 直接打印 list，保留 None 和 ''