"""
Study-plan table extraction from `10033-abit.pdf`.

DEPENDENCIES
------------
pip install --upgrade pdfplumber pandas openpyxl

USAGE
-----
python extract_study_plan.py <optional_output_folder>

The script:
1. Opens the PDF with pdfplumber.
2. Iterates through every page, extracting tabular structures.
3. Normalises headings and removes blank/None columns.
4. Concatenates all partial tables into one DataFrame.
5. Saves results to study_plan.csv and study_plan.xlsx.
"""

import sys, pathlib, pdfplumber, pandas as pd

######################################################################
# 0. CONFIGURATION
######################################################################
# PDF_FILE         = pathlib.Path("10033-abit.pdf")
# OUTPUT_DIR       = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else pathlib.Path(".")
# CSV_OUT          = OUTPUT_DIR / "study_plan.csv"
# EXCEL_OUT        = OUTPUT_DIR / "study_plan.xlsx"
KEEP_EMPTY_ROWS  = False       # set True if you want raw output
# PRINT_PREVIEW    = True        # prints first five rows to console

######################################################################
# 1. HELPER FUNCTIONS
######################################################################
def _clean_table(raw_table):
    """
    Remove columns that are entirely empty or contain only None/''.
    Trim stray newline characters inside cells.
    """
    df = pd.DataFrame(raw_table)
    # drop columns where every value is NaN / None / empty string
    df = df.dropna(axis=1, how="all")
    df = df.replace({"\\n": " "}, regex=True).applymap(lambda x: str(x).strip() if pd.notna(x) else x)
    if not KEEP_EMPTY_ROWS:
        df = df.dropna(how="all")
    return df.reset_index(drop=True)

######################################################################
# 2. MAIN EXTRACTION LOGIC
######################################################################
def extract_study_plan(pdf_path: pathlib.Path) -> pd.DataFrame:
    if not pdf_path.exists():
        raise FileNotFoundError(f"{pdf_path} not found.")

    tables = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            # pdfplumber returns a list of raw matrices (list-of-lists)
            raw_tables = page.extract_tables({
                "vertical_strategy": "lines",
                "horizontal_strategy": "lines",
                "snap_tolerance": 3,
                "join_tolerance": 3,
            })
            for raw_tab in raw_tables:
                cleaned = _clean_table(raw_tab)
                if cleaned.shape[1] >= 2:     # crude filter to avoid tiny garbage tables
                    cleaned.insert(0, "page", page_number)
                    tables.append(cleaned)

    if not tables:
        raise ValueError("No tables found in PDF. "
                         "Check PDF quality or tweak extraction settings.")

    # Concatenate page-wise results and reset row index
    full_df = pd.concat(tables, ignore_index=True)
    return full_df

def post_process_table(df) -> dict:
    df = df[[1,3]].iloc[2:]
    res_dict = {}
    for i in range(df.shape[0]):
        res_dict[df.iloc[i][1]] = df.iloc[i][3]
    return res_dict

