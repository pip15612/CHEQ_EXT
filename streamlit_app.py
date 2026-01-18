import streamlit as st
import cv2
import pytesseract
import easyocr
import numpy as np
import pandas as pd
from pdf2image import convert_from_path
import re
import os
import requests
from io import BytesIO
import tempfile
from datetime import datetime
from pprint import pprint

# ==========================================
# STREAMLIT PAGE CONFIG
# ==========================================
st.set_page_config(
    page_title="Thai Cheque OCR Processor",
    page_icon="📄",
    layout="wide"
)

# ==========================================
# DOWNLOAD e13b.traineddata
# ==========================================
@st.cache_resource
def download_e13b_traineddata():
    """Download e13b.traineddata for MICR recognition"""
    try:
        # Check if tessdata directory exists
        tessdata_paths = [
            '/usr/share/tesseract-ocr/4.00/tessdata',
            '/usr/share/tesseract-ocr/5/tessdata',
            '/app/tessdata',
            './tessdata'
        ]
        
        # Create local tessdata directory if none exist
        tessdata_dir = './tessdata'
        os.makedirs(tessdata_dir, exist_ok=True)
        
        target_path = os.path.join(tessdata_dir, 'e13b.traineddata')
        
        # Check if file already exists
        if os.path.exists(target_path):
            st.success("✅ e13b.traineddata already exists!")
            # Set TESSDATA_PREFIX environment variable
            os.environ['TESSDATA_PREFIX'] = tessdata_dir
            return True
        
        # Download from GitHub
        url = "https://github.com/DoubangoTelecom/tesseractMICR/raw/master/tessdata_best/e13b.traineddata"
        st.info(f"🔄 Downloading e13b.traineddata...")
        
        r = requests.get(url, timeout=30)
        if r.status_code == 200:
            with open(target_path, 'wb') as f:
                f.write(r.content)
            
            # Set TESSDATA_PREFIX environment variable
            os.environ['TESSDATA_PREFIX'] = tessdata_dir
            
            st.success("✅ e13b.traineddata downloaded successfully!")
            return True
        else:
            st.error(f"❌ Download failed with status code: {r.status_code}")
            return False
            
    except Exception as e:
        st.error(f"❌ Error downloading e13b.traineddata: {e}")
        return False

# Download e13b.traineddata on startup
download_e13b_traineddata()

# ==========================================
# GLOBAL DEBUG SWITCH
# ==========================================
DEBUG = True
DEBUG_DIR = "debug_out"
os.makedirs(DEBUG_DIR, exist_ok=True)

# ==========================================
# 1. Initialize OCR
# ==========================================
@st.cache_resource
def initialize_easyocr():
    """Initialize EasyOCR reader (cached to avoid reloading)"""
    try:
        reader = easyocr.Reader(['th', 'en'], gpu=False)  # Use cpu=True for deployment
        return reader
    except Exception as e:
        st.error(f"Error initializing EasyOCR: {e}")
        return None

# ==========================================
# 2. Helper Functions
# ==========================================

def _is_template_date_line(text: str) -> bool:
    """
    กันบรรทัดแม่แบบ เช่น:
    'วันที่ day เดือน/ month ปี/ year'
    """
    t = (text or "").strip().lower()
    template_words = [
        "day", "month", "year",
        "dd", "mm", "yyyy", "yyny",
        "วว", "ดด", "ปปปป",
        "วัน", "เดือน", "ปี"
    ]
    # ถ้ามีคำแม่แบบหลายคำ + ไม่มีเลขเยอะ -> ถือว่า template
    hits = sum(1 for w in template_words if w in t)
    digit_count = len(re.findall(r"\d", t))
    return (hits >= 2 and digit_count <= 4)

def _validate_date(d: str, m: str, y: str):
    """คืน string dd/mm/yyyy ถ้าผ่าน validate ไม่งั้นคืน '' """
    try:
        di, mi, yi = int(d), int(m), int(y)
        yi_check = yi - 543 if yi > 2400 else yi
        if 1 <= di <= 31 and 1 <= mi <= 12 and 1990 <= yi_check <= 2040:
            return f"{d}/{m}/{y}"
    except:
        return ""
    return ""


def clean_messy_date(text, debug=False):
    """
    Robust date parser:
    (1) dd/mm/yyyy (supports / - . and spaces)
    (2) dd mm yyyy (spaces only)
    (3) fallback digits with SLIDING WINDOW to avoid noise digits
        - try any 8-digit window as ddmmyyyy
        - then try 7-digit heuristic (missing leading zero)
    Handles OCR typo: วันที
    """
    if not text:
        return ""

    raw = text

    # Remove prefixes (including OCR typo วันที)
    text = re.sub(r'(?i)(วันที่|วันที|date|of)\s*[:\-]?\s*', '', text).strip()

    if debug:
        print("\n[DATE DEBUG] raw:", repr(raw))
        print("[DATE DEBUG] after_prefix_removed:", repr(text))

    # --- skip template lines ---
    try:
        if _is_template_date_line(text):
            if debug:
                print("[DATE DEBUG] template line -> skip")
            return ""
    except NameError:
        pass

    # (1) dd/mm/yyyy or dd-mm-yyyy or dd.mm.yyyy
    m = re.search(r'(\d{1,2})\s*[\/\-\.\s]\s*(\d{1,2})\s*[\/\-\.\s]\s*(\d{2,4})', text)
    if m:
        d, mo, y = m.group(1), m.group(2), m.group(3)
        if len(d) == 1: d = "0" + d
        if len(mo) == 1: mo = "0" + mo
        if len(y) == 2: y = "20" + y
        out = _validate_date(d, mo, y)
        if not out:
            try:
                di, mi, yi = int(d), int(mo), int(y)
                yi_check = yi - 543 if yi > 2400 else yi
                if 1 <= di <= 31 and 1 <= mi <= 12 and 1990 <= yi_check <= 2040:
                    out = f"{d}/{mo}/{y}"
            except:
                out = ""
        if out:
            if debug:
                print("[DATE DEBUG] sep regex matched ->", out)
            return out

    # (2) dd mm yyyy (spaces only)
    m2 = re.search(r'(\d{1,2})\s+(\d{1,2})\s+(\d{2,4})', text)
    if m2:
        d, mo, y = m2.group(1), m2.group(2), m2.group(3)
        if len(d) == 1: d = "0" + d
        if len(mo) == 1: mo = "0" + mo
        if len(y) == 2: y = "20" + y
        out = _validate_date(d, mo, y)
        if not out:
            try:
                di, mi, yi = int(d), int(mo), int(y)
                yi_check = yi - 543 if yi > 2400 else yi
                if 1 <= di <= 31 and 1 <= mi <= 12 and 1990 <= yi_check <= 2040:
                    out = f"{d}/{mo}/{y}"
            except:
                out = ""
        if out:
            if debug:
                print("[DATE DEBUG] spaced regex matched ->", out)
            return out

    # (3) fallback digits
    digits = "".join([c for c in text if c.isdigit()])
    if debug:
        print("[DATE DEBUG] digits_joined:", digits, "len=", len(digits))

    # helper validate
    def _val(d, mo, y):
        try:
            di, mi, yi = int(d), int(mo), int(y)
            yi_check = yi - 543 if yi > 2400 else yi
            if 1 <= di <= 31 and 1 <= mi <= 12 and 1990 <= yi_check <= 2040:
                return f"{d}/{mo}/{y}"
        except:
            return ""
        return ""

    # ✅ NEW: sliding 8-digit window (ddmmyyyy)
    if len(digits) >= 8:
        for start in range(0, len(digits) - 8 + 1):
            w = digits[start:start+8]
            d, mo, y = w[:2], w[2:4], w[4:]
            out = _val(d, mo, y)
            if out:
                if debug:
                    print(f"[DATE DEBUG] window8 matched at {start}: {w} -> {out}")
                return out

    # ✅ keep 7-digit heuristic for missing leading zero
    if len(digits) == 7:
        year = digits[-4:]
        prefix = digits[:-4]  # 3 digits
        # d(1)+m(2)
        out1 = _val("0"+prefix[0], prefix[1:], year)
        if out1:
            if debug:
                print("[DATE DEBUG] heuristic7 matched ->", out1)
            return out1
        # d(2)+m(1)
        out2 = _val(prefix[:2], "0"+prefix[2], year)
        if out2:
            if debug:
                print("[DATE DEBUG] heuristic7-alt matched ->", out2)
            return out2

    if debug:
        print("[DATE DEBUG] FAIL -> return empty")
    return ""


def extract_cheque_digit(micr_text):
    """Cheque digit = ตัวเลข 2 ตัวแรกของ MICR"""
    if not micr_text:
        return ""
    digits = "".join(c for c in micr_text if c.isdigit())
    return digits[:2] if len(digits) >= 2 else ""


def parse_micr_thai(micr_text):
    """
    Parse MICR into: Cheque No, Bank, Branch, Account
    Supports:
      A) [txn][cheque][bank+branch][account]
      B) [txn][cheque][bank][branch][account]
    """
    parts = [re.sub(r'[^\d]', '', p.strip())
             for p in re.split(r'[⑆⑇⑈⑉]', micr_text)
             if p.strip()]

    chq_no = bank_cd = branch_cd = acc_no = ""

    if len(parts) < 2:
        return chq_no, bank_cd, branch_cd, acc_no

    # parts[0] = transaction code (skip)
    chq_no = parts[1]

    # Case B: bank + branch separate
    if len(parts) >= 5 and len(parts[2]) == 3 and len(parts[3]) in (3, 4, 5) and len(parts[4]) >= 6:
        bank_cd = parts[2]
        branch_cd = parts[3]
        acc_no = parts[4]
        return chq_no, bank_cd, branch_cd, acc_no

    # Case A: bank+branch combined
    if len(parts) >= 3:
        raw_bank_data = parts[2]
        if len(raw_bank_data) >= 3:
            bank_cd = raw_bank_data[:3]
        if len(raw_bank_data) >= 7:
            branch_cd = raw_bank_data[3:7]
            rest = raw_bank_data[7:]
            if rest and not acc_no:
                acc_no = rest
        elif len(raw_bank_data) > 3:
            branch_cd = raw_bank_data[3:]

    if not acc_no and len(parts) >= 4:
        acc_no = parts[3]

    return chq_no, bank_cd, branch_cd, acc_no


def debug_print_lines(lines, title="OCR LINES"):
    print(f"\n========== {title} ==========")
    for idx, line in enumerate(lines):
        texts = [item[1] for item in line]
        print(f"[LINE {idx:02d}] -> {' | '.join(texts)}")
    print("================================\n")


def debug_draw_ocr_boxes(image_bgr, raw_results, out_path):
    """วาดกล่อง OCR ลงบนภาพแล้วเซฟออกมา"""
    vis = image_bgr.copy()
    for bbox, text, conf in raw_results:
        pts = np.array(bbox, dtype=np.int32)
        cv2.polylines(vis, [pts], isClosed=True, thickness=2, color=(0, 255, 0))
        x, y = int(pts[0][0]), int(pts[0][1])
        cv2.putText(vis, text[:25], (x, max(0, y - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.imwrite(out_path, vis)


def clean_amount_garbage(text):
    text = re.sub(r'(?i)(baht|bath|amount|จ่าย|pay|[^ก-๙\s])', '', text)
    return text.replace("*", "").replace("=", "").replace("(", "").replace(")", "").strip()


def clean_payee_final(text):
    typos = {"บรบท": "บริษัท", "บริบัท": "บริษัท", "จากัด": "จำกัด"}
    for w, c in typos.items():
        text = text.replace(w, c)
    return re.sub(r'(หรือผู้ถือ|Or Bearer).*$', '', text, flags=re.IGNORECASE).strip(" .-_^$#/")


# ==========================================
# 3. Utility: Crop + MICR crop helper
# ==========================================

def robust_auto_crop(image):
    img_h, img_w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    edged = cv2.Canny(blur, 50, 150)
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(edged, kernel, iterations=2)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
    screenCnt = None
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4 and cv2.contourArea(c) > (img_w * img_h * 0.35):
            screenCnt = approx
            break
    if screenCnt is None:
        return image[int(img_h * 0.15):int(img_h * 0.85), int(img_w * 0.02):int(img_w * 0.98)]
    x, y, w, h = cv2.boundingRect(screenCnt)
    return image[max(0, y - 20):min(img_h, y + h + 20), max(0, x - 60):min(img_w, x + w + 60)]


def crop_micr_region(image_bgr):
    h, w = image_bgr.shape[:2]
    return image_bgr[int(h * 0.78):h, 0:w]


def extract_micr(image):
    crop = crop_micr_region(image)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    try:
        raw_text = pytesseract.image_to_string(thresh, lang='e13b', config='--psm 7').strip()
    except:
        return ""
    mapping = {'A': '⑆', 'B': '⑇', 'C': '⑈', 'D': '⑉'}
    for k, v in mapping.items():
        raw_text = raw_text.replace(k, v).replace(k.lower(), v)
    return "".join([c for c in raw_text if c in "0123456789⑆⑇⑈⑉ "])


def group_text_into_lines(ocr_results):
    lines = []
    sorted_res = sorted(ocr_results, key=lambda r: r[0][0][1])
    for bbox, text, conf in sorted_res:
        matched = False
        for line in lines:
            y_min, y_max = line[0][0][0][1], line[0][0][2][1]
            curr_min, curr_max = bbox[0][1], bbox[2][1]
            inter = min(y_max, curr_max) - max(y_min, curr_min)
            if inter > 0 and (inter / min(y_max - y_min, curr_max - curr_min)) > 0.5:
                line.append((bbox, text))
                matched = True
                break
        if not matched:
            lines.append([(bbox, text)])
    for line in lines:
        line.sort(key=lambda item: item[0][0][0])
    return lines


# ==========================================
# 4. Core Extraction Engine
# ==========================================

def extract_thai_data(image, reader):
    img_h, img_w, _ = image.shape

    raw_results = reader.readtext(image, detail=1, paragraph=False)
    lines = group_text_into_lines(raw_results)

    if DEBUG:
        debug_print_lines(lines, "OCR LINES (GROUPED)")

    data = {"Date": "", "Payee": "", "Amount_Text": "", "Amount_Num": ""}
    money_kws = ["บาท", "Baht", "ถ้วน", "ล้าน", "แสน", "หมื่น", "พัน", "ร้อย", "สิบ"]

    for i, line in enumerate(lines):
        full_line_text = " ".join([item[1] for item in line]).strip()

        # ---- DATE LOGIC ----
        date_kw_hit = (
            ("วันที่" in full_line_text) or
            ("วันที" in full_line_text) or
            (re.search(r'(?i)\bdate\b', full_line_text) is not None)
        )

        if date_kw_hit and not data["Date"]:
            if _is_template_date_line(full_line_text):
                pass
            else:
                d0 = clean_messy_date(full_line_text, debug=DEBUG)
                if d0:
                    data["Date"] = d0
                else:
                    candidates = []
                    for k in [1, 2]:
                        if i + k < len(lines):
                            nxt = " ".join([item[1] for item in lines[i + k]]).strip()
                            candidates.append(("combo", full_line_text + " " + nxt))
                            candidates.append(("next_only", nxt))

                    found = False
                    for src, txt in candidates:
                        if _is_template_date_line(txt):
                            continue
                        dt = clean_messy_date(txt, debug=DEBUG)
                        if dt:
                            data["Date"] = dt
                            found = True
                            break

        # ---- DATE FALLBACK ----
        if not data["Date"]:
            digit_count = len(re.findall(r"\d", full_line_text))
            has_sep = any(s in full_line_text for s in ["/", "-", "."])

            if _is_template_date_line(full_line_text):
                continue

            if (digit_count >= 6) or (has_sep and digit_count >= 6):
                dt = clean_messy_date(full_line_text, debug=DEBUG)
                if dt:
                    data["Date"] = dt

        # ---- Amount Text ----
        if any(k in full_line_text for k in money_kws) and re.search(r'[ก-๙]', full_line_text):
            cleaned = clean_amount_garbage(full_line_text)
            if len(cleaned) > len(data["Amount_Text"]):
                data["Amount_Text"] = cleaned

        # ---- Payee ----
        pay_kws = ["จ่าย", "Pay", "แก่", "to"]
        if any(kw in full_line_text for kw in pay_kws) and not any(k in full_line_text for k in money_kws):
            name = full_line_text
            for k in pay_kws:
                name = name.replace(k, "")
            name = name.split("วันที่")[0].strip(" .-_/^*")
            if len(name) > 2 and not data["Payee"]:
                data["Payee"] = clean_payee_final(name)

    # ---- Amount_Num ----
    for line in lines:
        for bbox, text in line:
            # พิจารณาเฉพาะข้อความฝั่งขวาของภาพ
            if (bbox[0][0] + bbox[1][0]) / 2 > img_w * 0.5:
                money_pattern = r'\d{1,3}(?:,\d{3})*\.\d{2}'
                clean_t = text.replace(" ", "").replace("b", "").replace("B", "").replace("฿", "")
                matches = re.findall(money_pattern, clean_t)

                if matches:
                    candidate = max(matches, key=len)
                    if len(candidate) >= len(data["Amount_Num"]):
                        data["Amount_Num"] = candidate

    if DEBUG:
        print("[FINAL DATA]")
        pprint(data)

    return data, raw_results


# ==========================================
# 5. Main Process
# ==========================================

def process_cheque(file_path, reader):
    file_name = os.path.basename(file_path)
    base = os.path.splitext(file_name)[0]
    
    input_images = []
    if file_path.lower().endswith('.pdf'):
        pil_pages = convert_from_path(file_path, dpi=300)
        for p in pil_pages:
            input_images.append(cv2.cvtColor(np.array(p), cv2.COLOR_RGB2BGR))
    else:
        input_images.append(cv2.imread(file_path))

    results = []
    for page_idx, img in enumerate(input_images, start=1):
        cropped = robust_auto_crop(img)

        if DEBUG:
            st.write(f"**Page {page_idx}** - Cropped shape: {cropped.shape}")

        data, raw_results = extract_thai_data(cropped, reader)

        micr_raw = extract_micr(cropped)
        cheque_digit = extract_cheque_digit(micr_raw)
        chq_no, bank_cd, br_cd, acc_no = parse_micr_thai(micr_raw)

        results.append({
            "File Name": file_name,
            "Page": page_idx,
            "Date": data["Date"],
            "Payee": data["Payee"],
            "Amount": data["Amount_Num"],
            "Amount (Text)": data["Amount_Text"],
            "Cheque digit": cheque_digit,
            "Cheque Number": chq_no,
            "Bank Code": bank_cd,
            "Branch Code": br_cd,
            "Account number": acc_no,
        })

    return results


# ==========================================
# 6. Template Processing Functions (from GUI)
# ==========================================

def xlookup(lookup_value, lookup_array, return_array, if_not_found=None):
    """Mimics Excel's XLOOKUP function with automatic type conversion"""
    try:
        if not isinstance(lookup_array, pd.Series):
            lookup_array = pd.Series(lookup_array)
        if not isinstance(return_array, pd.Series):
            return_array = pd.Series(return_array)
        
        # If lookup_value is numeric and lookup_array is string, convert lookup_array to numeric
        if isinstance(lookup_value, (int, float)):
            try:
                lookup_array_numeric = pd.to_numeric(lookup_array, errors='coerce')
                mask = lookup_array_numeric == lookup_value
            except:
                mask = lookup_array == lookup_value
        else:
            # Try direct comparison first
            mask = lookup_array == lookup_value
            # If no match and lookup_value is string that looks like a number, try numeric comparison
            if not mask.any() and isinstance(lookup_value, str) and lookup_value.replace('.','').replace('-','').isdigit():
                try:
                    lookup_array_numeric = pd.to_numeric(lookup_array, errors='coerce')
                    lookup_value_numeric = pd.to_numeric(lookup_value, errors='coerce')
                    mask = lookup_array_numeric == lookup_value_numeric
                except:
                    pass
        
        if mask.any():
            idx = mask.idxmax()
            return return_array.iloc[idx]
        else:
            return if_not_found
    except:
        return if_not_found


def process_template_filling(pdf_file, fchn_file, master_file, template_file, business_partner=""):
    """Process template filling with lookups from FCHN and Master files"""
    import openpyxl
    
    # Load Template
    template_wb = openpyxl.load_workbook(template_file)
    template_sheet = template_wb['TEMPLATE (TR Teams) ']
    cash_sheet = template_wb['TEMPLATE (Cash Teams)']
    
    # Load data files
    pdf_df = pd.read_excel(pdf_file, sheet_name=0, dtype={'Account number': str, 'Cheque Number': str})
    fchn_df = pd.read_excel(fchn_file, sheet_name=0, dtype=str)
    master_df = pd.read_excel(master_file, sheet_name=0, dtype=str)
    
    # Clear existing data in TR Teams
    start_row = 11
    end_row = template_sheet.max_row
    for row_num in range(start_row, end_row + 1):
        for col_num in range(1, 35):
            template_sheet.cell(row_num, col_num).value = None
    
    # Process each row for TR Teams
    for idx, pdf_row in pdf_df.iterrows():
        row_num = start_row + idx
        
        cheque_number = pdf_row['Cheque Number']
        amount = pdf_row['Amount']
        account_number = pdf_row['Account number']
        
        # Determine Business Partner
        if business_partner:
            bp = business_partner
        else:
            bp = xlookup(account_number, master_df.iloc[:, 4], master_df.iloc[:, 6])
        
        # Fill Business Partner to Template
        if bp:
            template_sheet.cell(row_num, 2).value = str(bp)
        
        template_sheet.cell(row_num, 6).value = "23.12.2025"
        template_sheet.cell(row_num, 10).value = "23.12.2025"
        template_sheet.cell(row_num, 8).value = amount
        template_sheet.cell(row_num, 15).value = f"CHQ{cheque_number}"
        template_sheet.cell(row_num, 31).value = str(account_number)
        
        # Lookups from FCHN
        cheque_str = str(cheque_number)
        cheque_last8 = int(cheque_str[-8:]) if len(cheque_str) >= 8 else int(cheque_str)
        
        p_result = xlookup(cheque_last8, fchn_df.iloc[:, 0], fchn_df.iloc[:, 5])
        if p_result is not None:
            template_sheet.cell(row_num, 16).value = str(p_result)
        
        # Lookups from Master
        if bp:
            lookup_key = str(bp) + str(account_number)
            
            i_result = xlookup(lookup_key, master_df.iloc[:, 12], master_df.iloc[:, 11])
            if i_result is not None:
                template_sheet.cell(row_num, 9).value = str(i_result)
            
            k_result = xlookup(lookup_key, master_df.iloc[:, 12], master_df.iloc[:, 7])
            if k_result is not None:
                template_sheet.cell(row_num, 11).value = str(k_result)
                template_sheet.cell(row_num, 17).value = str(k_result)
            
            y_result = xlookup(lookup_key, master_df.iloc[:, 12], master_df.iloc[:, 8])
            if y_result is not None:
                template_sheet.cell(row_num, 25).value = str(y_result)
                template_sheet.cell(row_num, 37).value = str(y_result)
            
            ac_result = xlookup(lookup_key, master_df.iloc[:, 12], master_df.iloc[:, 10])
            if ac_result is not None:
                template_sheet.cell(row_num, 29).value = str(ac_result)
        
        # Company Code to Column A
        a_result = xlookup(account_number, master_df.iloc[:, 4], master_df.iloc[:, 0])
        if a_result is not None:
            template_sheet.cell(row_num, 1).value = str(a_result)
        
        # Cost Center to Column R
        r_result = xlookup(account_number, master_df.iloc[:, 4], master_df.iloc[:, 9])
        if r_result is not None:
            template_sheet.cell(row_num, 18).value = str(r_result)
        
        s_result = xlookup(account_number, master_df.iloc[:, 4], master_df.iloc[:, 1])
        if s_result is not None:
            template_sheet.cell(row_num, 19).value = str(s_result).zfill(4)
    
    # Process Cash Teams sheet
    max_row = cash_sheet.max_row
    for row in range(6, max_row + 1):
        for col in range(1, 40):
            cash_sheet.cell(row, col).value = None
    
    cash_start_row = 6
    for idx, pdf_row in pdf_df.iterrows():
        cash_row = cash_start_row + idx
        
        cheque_number = pdf_row['Cheque Number']
        amount = pdf_row['Amount']
        account_number = pdf_row['Account number']
        
        # Determine Business Partner
        if business_partner:
            bp = business_partner
        else:
            bp = xlookup(account_number, master_df.iloc[:, 4], master_df.iloc[:, 6])
        
        # A: Company Code
        company_code = xlookup(account_number, master_df.iloc[:, 4], master_df.iloc[:, 0])
        if company_code:
            cash_sheet.cell(cash_row, 1).value = str(company_code)
        
        # B: Business Place
        business_place = xlookup(account_number, master_df.iloc[:, 4], master_df.iloc[:, 1])
        if business_place:
            cash_sheet.cell(cash_row, 2).value = str(business_place).zfill(4)
        
        # E: Start Date
        cash_sheet.cell(cash_row, 5).value = "23.12.2025"
        
        # F: Payment Amount
        cash_sheet.cell(cash_row, 6).value = amount
        
        # G: Bank Account Number
        cash_sheet.cell(cash_row, 7).value = str(account_number)
        
        # C: Company Name from FCHN column H
        company_name = xlookup(str(account_number), fchn_df.iloc[:, 8], fchn_df.iloc[:, 7])
        if company_name and str(company_name).lower() not in ['none', 'nan', '']:
            cash_sheet.cell(cash_row, 3).value = str(company_name)
        
        # D: House Bank from FCHN column C
        house_bank = xlookup(str(account_number), fchn_df.iloc[:, 8], fchn_df.iloc[:, 2])
        if house_bank and str(house_bank).lower() not in ['none', 'nan', '']:
            import re
            bank_name_only = re.sub(r'\d+', '', str(house_bank)).strip()
            cash_sheet.cell(cash_row, 4).value = bank_name_only
        
        # H: Assignment
        cash_sheet.cell(cash_row, 8).value = f"CHQ{cheque_number}"
        
        # I: Business Partner
        if bp:
            cash_sheet.cell(cash_row, 9).value = str(bp)
    
    # Save to BytesIO
    output = BytesIO()
    template_wb.save(output)
    template_wb.close()
    output.seek(0)
    
    return output, len(pdf_df)


# ==========================================
# 7. STREAMLIT UI
# ==========================================

def main():
    st.title("📄 Thai Cheque Processing System")
    st.markdown("Complete solution for Thai cheque OCR and template processing")
    
    # Create tabs
    tab1, tab2 = st.tabs(["🔍 OCR Extraction", "📋 Template Processing"])
    
    # ==========================================
    # TAB 1: OCR EXTRACTION
    # ==========================================
    with tab1:
        st.header("🔍 Extract Data from Cheques")
        st.markdown("Upload cheque images/PDFs to extract data using OCR")
        
        # Initialize OCR
        with st.spinner("🚀 Initializing EasyOCR..."):
            reader = initialize_easyocr()
        
        if reader is None:
            st.error("Failed to initialize OCR. Please check the logs.")
        else:
            st.success("✅ OCR initialized successfully!")
            
            # File uploader
            st.markdown("---")
            st.subheader("📂 Upload Cheque Files")
            uploaded_files = st.file_uploader(
                "Choose PDF or image files",
                type=['pdf', 'png', 'jpg', 'jpeg'],
                accept_multiple_files=True,
                key="ocr_uploader"
            )
            
            if uploaded_files:
                st.info(f"📎 {len(uploaded_files)} file(s) uploaded")
                
                if st.button("🚀 Extract Data", type="primary", key="extract_btn"):
                    all_results = []
                    
                    # Progress bar
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    for idx, uploaded_file in enumerate(uploaded_files):
                        status_text.text(f"Processing {uploaded_file.name}...")
                        
                        # Save uploaded file temporarily
                        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                            tmp_file.write(uploaded_file.getvalue())
                            tmp_path = tmp_file.name
                        
                        try:
                            # Process the file
                            results = process_cheque(tmp_path, reader)
                            all_results.extend(results)
                            
                        except Exception as e:
                            st.error(f"Error processing {uploaded_file.name}: {e}")
                        
                        finally:
                            # Clean up temp file
                            if os.path.exists(tmp_path):
                                os.unlink(tmp_path)
                        
                        # Update progress
                        progress_bar.progress((idx + 1) / len(uploaded_files))
                    
                    status_text.text("✅ Extraction complete!")
                    
                    # Display results
                    if all_results:
                        df = pd.DataFrame(all_results)
                        
                        st.markdown("---")
                        st.subheader("📊 Extracted Data")
                        st.dataframe(df, use_container_width=True)
                        
                        # Download Excel
                        output = BytesIO()
                        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                            df.to_excel(writer, index=False, sheet_name='Sheet1')
                            workbook = writer.book
                            worksheet = writer.sheets['Sheet1']
                            text_format = workbook.add_format({'num_format': '@'})
                            
                            for col_name in ["Cheque Number", "Account number", "Cheque digit", "Bank Code", "Branch Code"]:
                                if col_name in df.columns:
                                    col_idx = df.columns.get_loc(col_name)
                                    worksheet.set_column(col_idx, col_idx, 20, text_format)
                        
                        output.seek(0)
                        
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        st.download_button(
                            label="📥 Download Extracted Data (Excel)",
                            data=output,
                            file_name=f"cheque_extracted_{timestamp}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            key="download_extracted"
                        )
                        
                        st.success("💡 **Next step:** Use this file in 'Template Processing' tab!")
                    else:
                        st.warning("No results extracted from the files.")
            
            # Instructions
            st.markdown("---")
            st.markdown("""
            ### 📝 Instructions
            1. Upload Thai cheque images (PNG, JPG) or PDF files
            2. Click **Extract Data** to process
            3. Download the extracted Excel file
            4. Use the Excel file in **Template Processing** tab
            """)
    
    # ==========================================
    # TAB 2: TEMPLATE PROCESSING
    # ==========================================
    with tab2:
        st.header("📋 Fill Template from Extracted Data")
        st.markdown("Upload extracted data and lookup files to fill TR & Cash templates")
        
        st.markdown("---")
        
        # File uploaders
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📂 Required Files")
            
            template_file = st.file_uploader(
                "1️⃣ Template File",
                type=['xlsx'],
                help="Template TR & Cash.xlsx",
                key="template_uploader"
            )
            
            pdf_file = st.file_uploader(
                "2️⃣ Extracted Data (from Tab 1)",
                type=['xlsx'],
                help="Excel file from OCR extraction",
                key="pdf_uploader"
            )
        
        with col2:
            st.subheader("📊 Lookup Files")
            
            fchn_file = st.file_uploader(
                "3️⃣ FCHN File",
                type=['xlsx'],
                help="FCHN.xlsx for lookups",
                key="fchn_uploader"
            )
            
            master_file = st.file_uploader(
                "4️⃣ Master File",
                type=['xlsx'],
                help="Copy of Master File*.xlsx",
                key="master_uploader"
            )
        
        # Business Partner input
        st.markdown("---")
        st.subheader("⚙️ Configuration")
        business_partner = st.text_input(
            "Business Partner (Optional)",
            placeholder="e.g. UOB0052, CIM0199, TNB0497",
            help="Leave empty for auto-lookup from Master file"
        )
        
        # Process button
        st.markdown("---")
        
        if st.button("🚀 Process Template", type="primary", key="process_template_btn", disabled=not all([template_file, pdf_file, fchn_file, master_file])):
            if all([template_file, pdf_file, fchn_file, master_file]):
                try:
                    with st.spinner("Processing template..."):
                        # Save uploaded files temporarily
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as tmp_template:
                            tmp_template.write(template_file.getvalue())
                            tmp_template_path = tmp_template.name
                        
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as tmp_pdf:
                            tmp_pdf.write(pdf_file.getvalue())
                            tmp_pdf_path = tmp_pdf.name
                        
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as tmp_fchn:
                            tmp_fchn.write(fchn_file.getvalue())
                            tmp_fchn_path = tmp_fchn.name
                        
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as tmp_master:
                            tmp_master.write(master_file.getvalue())
                            tmp_master_path = tmp_master.name
                        
                        # Process template filling
                        output, total_rows = process_template_filling(
                            tmp_pdf_path,
                            tmp_fchn_path,
                            tmp_master_path,
                            tmp_template_path,
                            business_partner
                        )
                        
                        # Clean up temp files
                        for tmp_path in [tmp_template_path, tmp_pdf_path, tmp_fchn_path, tmp_master_path]:
                            if os.path.exists(tmp_path):
                                os.unlink(tmp_path)
                        
                        st.success(f"✅ Template processed successfully! ({total_rows} rows)")
                        
                        # Download button
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        st.download_button(
                            label="📥 Download Filled Template",
                            data=output,
                            file_name=f"Template_PDF_Filled_{timestamp}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            key="download_template"
                        )
                
                except Exception as e:
                    st.error(f"❌ Error processing template: {str(e)}")
                    st.exception(e)
            else:
                st.warning("⚠️ Please upload all required files!")
        
        # Instructions
        st.markdown("---")
        st.markdown("""
        ### 📝 Instructions
        1. Upload **Template TR & Cash.xlsx** file
        2. Upload **Extracted Data** from Tab 1 (or any compatible Excel)
        3. Upload **FCHN.xlsx** for lookups
        4. Upload **Master File** for lookups
        5. (Optional) Enter Business Partner code
        6. Click **Process Template**
        7. Download the filled template
        
        ### ℹ️ What this does
        - Fills **TR Teams** sheet with lookups from FCHN & Master
        - Fills **Cash Teams** sheet with bank information
        - Applies formulas and formatting automatically
        - Ready to import into SAP system
        """)


if __name__ == "__main__":
    main()
