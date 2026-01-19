
import streamlit as st
import cv2
import numpy as np
import pytesseract
import easyocr
from pdf2image import convert_from_bytes
from PIL import Image
import pandas as pd
import re
from io import BytesIO
import os
import traceback
import shutil
from pathlib import Path

# ========== Setup Tessdata for MICR ==========
@st.cache_resource
def setup_tessdata():
    """จัดการ e13b.traineddata สำหรับ MICR"""
    try:
        tessdata_dir = Path("/tmp/tessdata")
        tessdata_dir.mkdir(parents=True, exist_ok=True)
        
        local_e13b = Path("e13b.traineddata")
        target_e13b = tessdata_dir / "e13b.traineddata"
        
        if local_e13b.exists() and not target_e13b.exists():
            shutil.copy(local_e13b, target_e13b)
        
        return str(tessdata_dir)
    except Exception as e:
        st.warning(f"⚠️ ไม่สามารถตั้งค่า e13b: {str(e)}")
        return None

CUSTOM_TESSDATA = setup_tessdata()

# ========== Initialize EasyOCR ==========
@st.cache_resource
def initialize_easyocr():
    """Initialize EasyOCR reader (cached)"""
    with st.spinner("🔄 กำลังโหลด EasyOCR models... (ครั้งแรกใช้เวลา 2-3 นาที)"):
        return easyocr.Reader(['th', 'en'], gpu=False)

# ========== Helper Functions ==========
def _is_template_date_line(text: str) -> bool:
    """กันบรรทัดแม่แบบวันที่"""
    t = (text or "").strip().lower()
    template_words = ["day", "month", "year", "dd", "mm", "yyyy", "วว", "ดด", "ปปปป", "วัน", "เดือน", "ปี"]
    hits = sum(1 for w in template_words if w in t)
    digit_count = len(re.findall(r"\d", t))
    return (hits >= 2 and digit_count <= 4)

def _validate_date(d: str, m: str, y: str):
    """Validate และคืน dd/mm/yyyy"""
    try:
        di, mi, yi = int(d), int(m), int(y)
        yi_check = yi - 543 if yi > 2400 else yi
        if 1 <= di <= 31 and 1 <= mi <= 12 and 1990 <= yi_check <= 2040:
            return f"{d}/{m}/{y}"
    except:
        return ""
    return ""

def clean_messy_date(text):
    """Robust date parser with sliding window"""
    if not text:
        return ""
    
    # Remove prefixes
    text = re.sub(r'(?i)(วันที่|วันที|date|of)\s*[:\-]?\s*', '', text).strip()
    
    # Skip template lines
    if _is_template_date_line(text):
        return ""
    
    # (1) dd/mm/yyyy with separators
    m = re.search(r'(\d{1,2})\s*[\/\-\.\s]\s*(\d{1,2})\s*[\/\-\.\s]\s*(\d{2,4})', text)
    if m:
        d, mo, y = m.group(1), m.group(2), m.group(3)
        if len(d) == 1: d = "0" + d
        if len(mo) == 1: mo = "0" + mo
        if len(y) == 2: y = "20" + y
        out = _validate_date(d, mo, y)
        if out:
            return out
    
    # (2) dd mm yyyy (spaces only)
    m2 = re.search(r'(\d{1,2})\s+(\d{1,2})\s+(\d{2,4})', text)
    if m2:
        d, mo, y = m2.group(1), m2.group(2), m2.group(3)
        if len(d) == 1: d = "0" + d
        if len(mo) == 1: mo = "0" + mo
        if len(y) == 2: y = "20" + y
        out = _validate_date(d, mo, y)
        if out:
            return out
    
    # (3) Sliding 8-digit window
    digits = "".join([c for c in text if c.isdigit()])
    if len(digits) >= 8:
        for start in range(0, len(digits) - 8 + 1):
            w = digits[start:start+8]
            d, mo, y = w[:2], w[2:4], w[4:]
            out = _validate_date(d, mo, y)
            if out:
                return out
    
    # (4) 7-digit heuristic
    if len(digits) == 7:
        year = digits[-4:]
        prefix = digits[:-4]
        out1 = _validate_date("0"+prefix[0], prefix[1:], year)
        if out1:
            return out1
        out2 = _validate_date(prefix[:2], "0"+prefix[2], year)
        if out2:
            return out2
    
    return ""

def extract_cheque_digit(micr_text):
    """Cheque digit = 2 ตัวแรกของ MICR"""
    if not micr_text:
        return ""
    digits = "".join(c for c in micr_text if c.isdigit())
    return digits[:2] if len(digits) >= 2 else ""

def parse_micr_thai(micr_text):
    """Parse MICR: [txn][cheque][bank][branch][account]"""
    parts = [re.sub(r'[^\d]', '', p.strip())
             for p in re.split(r'[⑆⑇⑈⑉]', micr_text)
             if p.strip()]
    
    chq_no = bank_cd = branch_cd = acc_no = ""
    
    if len(parts) < 2:
        return chq_no, bank_cd, branch_cd, acc_no
    
    chq_no = parts[1]
    
    # Case B: bank + branch separate
    if len(parts) >= 5 and len(parts[2]) == 3:
        bank_cd = parts[2]
        branch_cd = parts[3]
        acc_no = parts[4]
        return chq_no, bank_cd, branch_cd, acc_no
    
    # Case A: bank+branch combined
    if len(parts) >= 3:
        raw = parts[2]
        if len(raw) >= 3:
            bank_cd = raw[:3]
        if len(raw) >= 7:
            branch_cd = raw[3:7]
            acc_no = raw[7:] if len(raw) > 7 else ""
        elif len(raw) > 3:
            branch_cd = raw[3:]
    
    if not acc_no and len(parts) >= 4:
        acc_no = parts[3]
    
    return chq_no, bank_cd, branch_cd, acc_no

def clean_amount_garbage(text):
    """ลบคำที่ไม่เกี่ยวข้องออกจากจำนวนเงินคำอ่าน"""
    text = re.sub(r'(?i)(baht|bath|amount|จ่าย|pay|[^ก-๙\s])', '', text)
    return text.replace("*", "").replace("=", "").strip()

def clean_payee_final(text):
    """แก้ typo ชื่อผู้รับเงิน"""
    typos = {"บรบท": "บริษัท", "บริบัท": "บริษัท", "จากัด": "จำกัด"}
    for w, c in typos.items():
        text = text.replace(w, c)
    return re.sub(r'(หรือผู้ถือ|Or Bearer).*$', '', text, flags=re.IGNORECASE).strip(" .-_^$#/")

# ========== Image Processing ==========
def robust_auto_crop(image):
    """Auto-crop เช็คออกจากพื้นหลัง"""
    img_h, img_w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    edged = cv2.Canny(blur, 50, 150)
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(edged, kernel, iterations=2)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
    
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4 and cv2.contourArea(c) > (img_w * img_h * 0.35):
            x, y, w, h = cv2.boundingRect(approx)
            return image[max(0, y-20):min(img_h, y+h+20), max(0, x-60):min(img_w, x+w+60)]
    
    return image[int(img_h*0.15):int(img_h*0.85), int(img_w*0.02):int(img_w*0.98)]

def crop_micr_region(image_bgr):
    """Crop MICR region (ส่วนล่าง 22%)"""
    h, w = image_bgr.shape[:2]
    return image_bgr[int(h * 0.78):h, 0:w]

def extract_micr(image):
    """Extract MICR code ด้วย Tesseract e13b"""
    crop = crop_micr_region(image)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    micr_text = ""
    if CUSTOM_TESSDATA:
        try:
            micr_text = pytesseract.image_to_string(
                thresh, 
                config=f'--psm 7 --tessdata-dir {CUSTOM_TESSDATA} -l e13b'
            ).strip()
        except:
            pass
    
    if not micr_text:
        try:
            micr_text = pytesseract.image_to_string(thresh, lang='eng', config='--psm 7').strip()
        except:
            return ""
    
    # Map A/B/C/D to MICR symbols
    mapping = {'A': '⑆', 'B': '⑇', 'C': '⑈', 'D': '⑉'}
    for k, v in mapping.items():
        micr_text = micr_text.replace(k, v).replace(k.lower(), v)
    
    return "".join([c for c in micr_text if c in "0123456789⑆⑇⑈⑉ "])

def group_text_into_lines(ocr_results):
    """จัดกลุ่มข้อความเป็นบรรทัด"""
    lines = []
    sorted_res = sorted(ocr_results, key=lambda r: r[0][0][1])
    
    for bbox, text, conf in sorted_res:
        matched = False
        for line in lines:
            y_min = min([item[0][0][1] for item in line])
            y_max = max([item[0][2][1] for item in line])
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

# ========== Core Extraction Engine ==========
def extract_thai_data(image, reader):
    """ดึงข้อมูลหลักจากเช็ค"""
    img_h, img_w, _ = image.shape
    
    raw_results = reader.readtext(image, detail=1, paragraph=False)
    lines = group_text_into_lines(raw_results)
    
    data = {"Date": "", "Payee": "", "Amount_Text": "", "Amount_Num": ""}
    money_kws = ["บาท", "Baht", "ถ้วน", "ล้าน", "แสน", "หมื่น", "พัน", "ร้อย", "สิบ"]
    
    for i, line in enumerate(lines):
        full_line_text = " ".join([item[1] for item in line]).strip()
        
        # === DATE EXTRACTION ===
        date_kw_hit = (
            ("วันที่" in full_line_text) or 
            ("วันที" in full_line_text) or
            (re.search(r'(?i)\bdate\b', full_line_text) is not None)
        )
        
        if date_kw_hit and not data["Date"]:
            if not _is_template_date_line(full_line_text):
                d0 = clean_messy_date(full_line_text)
                if d0:
                    data["Date"] = d0
                else:
                    # Look ahead 1-2 lines
                    for k in [1, 2]:
                        if i + k < len(lines):
                            nxt = " ".join([item[1] for item in lines[i + k]]).strip()
                            if not _is_template_date_line(nxt):
                                dt = clean_messy_date(full_line_text + " " + nxt)
                                if dt:
                                    data["Date"] = dt
                                    break
        
        # Date fallback
        if not data["Date"]:
            digit_count = len(re.findall(r"\d", full_line_text))
            if digit_count >= 6 and not _is_template_date_line(full_line_text):
                dt = clean_messy_date(full_line_text)
                if dt:
                    data["Date"] = dt
        
        # === AMOUNT TEXT ===
        if any(k in full_line_text for k in money_kws) and re.search(r'[ก-๙]', full_line_text):
            cleaned = clean_amount_garbage(full_line_text)
            if len(cleaned) > len(data["Amount_Text"]):
                data["Amount_Text"] = cleaned
        
        # === PAYEE ===
        pay_kws = ["จ่าย", "Pay", "แก่", "to"]
        if any(kw in full_line_text for kw in pay_kws) and not any(k in full_line_text for k in money_kws):
            name = full_line_text
            for k in pay_kws:
                name = name.replace(k, "")
            name = name.split("วันที่")[0].strip(" .-_/^*")
            if len(name) > 2 and not data["Payee"]:
                data["Payee"] = clean_payee_final(name)
    
    # === AMOUNT NUMBER (ไม่ verify กับคำอ่าน) ===
    for line in lines:
        for bbox, text in line:
            if (bbox[0][0] + bbox[1][0]) / 2 > img_w * 0.5:
                money_pattern = r'\d{1,3}(?:,\d{3})*\.\d{2}'
                clean_t = text.replace(" ", "").replace("b", "").replace("B", "").replace("฿", "")
                matches = re.findall(money_pattern, clean_t)
                
                if matches:
                    candidate = max(matches, key=len)
                    if len(candidate) >= len(data["Amount_Num"]):
                        data["Amount_Num"] = candidate
    
    return data

# ========== Main Process Function ==========
def process_cheque(uploaded_file):
    """ประมวลผลไฟล์เช็ค (PDF/Image)"""
    try:
        # Initialize EasyOCR
        reader = initialize_easyocr()
        
        uploaded_file.seek(0)
        file_bytes = uploaded_file.read()
        
        # แปลง PDF เป็นภาพ
        st.info("📄 กำลังแปลงไฟล์...")
        if uploaded_file.name.lower().endswith('.pdf'):
            images = convert_from_bytes(file_bytes, dpi=250)  # ลด DPI เพื่อประหยัด memory
            image = images[0]
        else:
            image = Image.open(BytesIO(file_bytes))
        
        # Resize ถ้าใหญ่เกินไป (memory optimization)
        max_dim = 3000
        if max(image.size) > max_dim:
            ratio = max_dim / max(image.size)
            new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
            image = image.resize(new_size, Image.Resampling.LANCZOS)
        
        # แปลงเป็น OpenCV format
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Auto-crop
        st.info("✂️ กำลัง crop เช็ค...")
        cropped = robust_auto_crop(cv_image)
        
        # Extract data
        st.info("🔍 กำลังดึงข้อความจากเช็ค...")
        data = extract_thai_data(cropped, reader)
        
        # Extract MICR
        st.info("🔢 กำลังดึง MICR code...")
        micr_raw = extract_micr(cropped)
        cheque_digit = extract_cheque_digit(micr_raw)
        chq_no, bank_cd, br_cd, acc_no = parse_micr_thai(micr_raw)
        
        result = {
            "วันที่": data["Date"],
            "ผู้รับเงิน": data["Payee"],
            "จำนวนเงิน": data["Amount_Num"],
            "จำนวนเงิน (คำอ่าน)": data["Amount_Text"],
            "Cheque digit": cheque_digit,
            "หมายเลขเช็ค": chq_no,
            "รหัสธนาคาร": bank_cd,
            "รหัสสาขา": br_cd,
            "เลขบัญชี": acc_no,
            "MICR (ดิบ)": micr_raw[:100]
        }
        
        return result, cropped
        
    except Exception as e:
        st.error(f"❌ เกิดข้อผิดพลาด: {str(e)}")
        st.code(traceback.format_exc())
        return None, None

def process_template_filling(data_file, template_file):
    """ประมวลผล Template Filling"""
    try:
        df_data = pd.read_excel(data_file)
        
        with pd.ExcelFile(template_file) as xls:
            df_tr = pd.read_excel(xls, 'TR')
            df_cash = pd.read_excel(xls, 'Cash')
        
        lookup_dict = df_data.set_index('รหัสบุคคล')[['ชื่อ', 'จำนวนเงิน', 'หมายเหตุ']].to_dict('index')
        
        def xlookup(code, field):
            if pd.notna(code) and code in lookup_dict:
                return lookup_dict[code].get(field, '')
            return ''
        
        df_tr['ชื่อ'] = df_tr['รหัสบุคคล'].apply(lambda x: xlookup(x, 'ชื่อ'))
        df_tr['จำนวนเงิน'] = df_tr['รหัสบุคคล'].apply(lambda x: xlookup(x, 'จำนวนเงิน'))
        df_tr['หมายเหตุ'] = df_tr['รหัสบุคคล'].apply(lambda x: xlookup(x, 'หมายเหตุ'))
        
        df_cash['ชื่อ'] = df_cash['รหัสบุคคล'].apply(lambda x: xlookup(x, 'ชื่อ'))
        df_cash['จำนวนเงิน'] = df_cash['รหัสบุคคล'].apply(lambda x: xlookup(x, 'จำนวนเงิน'))
        df_cash['หมายเหตุ'] = df_cash['รหัสบุคคล'].apply(lambda x: xlookup(x, 'หมายเหตุ'))
        
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df_tr.to_excel(writer, sheet_name='TR', index=False)
            df_cash.to_excel(writer, sheet_name='Cash', index=False)
        output.seek(0)
        
        return output, len(df_tr) + len(df_cash)
        
    except Exception as e:
        st.error(f"❌ เกิดข้อผิดพลาด: {str(e)}")
        st.code(traceback.format_exc())
        return None, 0

# ========== Streamlit UI ==========
def main():
    st.set_page_config(page_title="Thai Cheque OCR", page_icon="🏦", layout="wide")
    
    st.title("🏦 ระบบดึงข้อความจากเช็คไทย")
    st.caption("📌 ใช้ EasyOCR (ไทย + อังกฤษ) + Tesseract MICR (e13b)")
    
    tab1, tab2 = st.tabs(["📄 ดึงข้อความจากเช็ค", "📊 เติมข้อมูล Template"])
    
    # ===== TAB 1: OCR Extraction =====
    with tab1:
        st.markdown("### 📤 อัปโหลดไฟล์เช็ค")
        uploaded_file = st.file_uploader(
            "เลือกไฟล์ PDF หรือ รูปภาพ",
            type=['pdf', 'jpg', 'jpeg', 'png'],
            help="รองรับไฟล์ PDF และรูปภาพ (JPG, PNG)"
        )
        
        if uploaded_file:
            col1, col2 = st.columns([1, 1])
            
            with col1:
                # แสดงตัวอย่างไฟล์
                try:
                    uploaded_file.seek(0)
                    if uploaded_file.name.lower().endswith('.pdf'):
                        images = convert_from_bytes(uploaded_file.read(), dpi=150)
                        st.image(images[0], caption="ไฟล์ที่อัปโหลด", use_container_width=True)
                    else:
                        st.image(uploaded_file, caption="ไฟล์ที่อัปโหลด", use_container_width=True)
                except Exception as e:
                    st.warning(f"⚠️ ไม่สามารถแสดงตัวอย่าง: {str(e)}")
            
            with col2:
                if st.button("🚀 เริ่มประมวลผล", type="primary", use_container_width=True):
                    with st.spinner("⏳ กำลังประมวลผล... (EasyOCR ครั้งแรกใช้เวลา 2-3 นาที)"):
                        result, cropped = process_cheque(uploaded_file)
                        
                        if result:
                            st.success("✅ ประมวลผลสำเร็จ!")
                            
                            # แสดงผล
                            df_result = pd.DataFrame([result]).T
                            df_result.columns = ['ข้อมูล']
                            st.dataframe(df_result, use_container_width=True)
                            
                            # Download CSV
                            csv = pd.DataFrame([result]).to_csv(index=False, encoding='utf-8-sig')
                            st.download_button(
                                label="📥 ดาวน์โหลด CSV",
                                data=csv,
                                file_name="cheque_data.csv",
                                mime="text/csv"
                            )
                            
                            # แสดงภาพที่ crop แล้ว
                            with st.expander("🖼️ ดูภาพที่ Crop แล้ว"):
                                st.image(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB), use_container_width=True)
    
    # ===== TAB 2: Template Filling =====
    with tab2:
        st.markdown("### 📤 อัปโหลดไฟล์")
        col1, col2 = st.columns(2)
        
        with col1:
            data_file = st.file_uploader(
                "📊 ไฟล์ข้อมูล (Data)",
                type=['xlsx'],
                help="ไฟล์ Excel ที่มีคอลัมน์: รหัสบุคคล, ชื่อ, จำนวนเงิน, หมายเหตุ"
            )
        
        with col2:
            template_file = st.file_uploader(
                "📋 ไฟล์ Template",
                type=['xlsx'],
                help="ไฟล์ Excel Template ที่มี 2 sheets: TR และ Cash"
            )
        
        if data_file and template_file:
            if st.button("🚀 เริ่มเติมข้อมูล", type="primary", use_container_width=True):
                with st.spinner("⏳ กำลังประมวลผล..."):
                    output, count = process_template_filling(data_file, template_file)
                    
                    if output:
                        st.success(f"✅ เติมข้อมูลสำเร็จ! จำนวน {count} แถว")
                        
                        st.download_button(
                            label="📥 ดาวน์โหลดไฟล์ที่เติมข้อมูลแล้ว",
                            data=output,
                            file_name="template_filled.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )

if __name__ == "__main__":
    main()
