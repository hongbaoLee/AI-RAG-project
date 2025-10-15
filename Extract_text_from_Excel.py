# Extract text and tables from an Excel file, including all sheets.
import pandas as pd

def extract_text_from_excel(file_path):
    df = pd.read_excel(file_path, sheet_name=None)  # Load all sheets
    all_text = ""
    for sheet_name, sheet_df in df.items():
        all_text += f"Sheet: {sheet_name}\n"
        all_text += sheet_df.fillna('').astype(str).to_csv(sep='\t', index=False, header=True)
        all_text += "\n"
    return all_text

def extract_tables_from_excel(file_path):
    df = pd.read_excel(file_path, sheet_name=None)
    tables = []
    for sheet_name, sheet_df in df.items():
        tables.append((sheet_name, sheet_df))
    return tables

if __name__ == "__main__":
    file_path = "查勘人员考核办法_new.xlsx"
    # Extract text
    all_text = extract_text_from_excel(file_path)
    print("Extracted Text:")
    print(all_text)
    # Extract tables
    tables = extract_tables_from_excel(file_path)
    print("Extracted Tables:")
    for sheet_name, table in tables:
        print(f"Sheet: {sheet_name}")
        print(table)
        print()




