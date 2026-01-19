
"""
Thai Cheque OCR System - Tesseract Only Version (Optimized for Streamlit Cloud)
Uses only Tesseract OCR (~100MB RAM) instead of EasyOCR (~1.2GB RAM)
"""

import streamlit as st
import pytesseract
from PIL import Image
import pandas as pd
import numpy as np
import cv2
import re
import os
from io import BytesIO
import tempfile
from pdf2image import convert_from_bytes
import traceback

# =====================================================================
# 1. PAGE CONFIG
# =====================================================================
st.set_page_config(
    page_title="Thai Cheque OCR - Tesseract Edition",
    page_icon="🔍",
    layout="wide"
)

# =====================================================================
# 2. DOWNLOAD E13B TRAINEDDATA
# =====================================================================
def download_e13b_traineddata():
    """Download and setup e13b.traineddata for MICR recognition"""
    tessdata_dir = '/tmp/tessdata'
    os.makedirs(tessdata_dir, exist_ok=True)
    
    e13b_path = os.path.join(tessdata_dir, 'e13b.traineddata')
    
    # Check if already exists
    if os.path.exists(e13b_path):
        st.success(f"✅ e13b.traineddata พร้อมใช้งาน")
        os.environ['TESSDATA_PREFIX'] = '/tmp/'
        return True
    
    # Try to copy from local repo
    local_e13b = 'e13b.traineddata'
    if os.path.exists(local_e13b):
        import shutil
        shutil.copy(local_e13b, e13b_path)
        st.success(f"✅ คัดลอก e13b.traineddata สำเร็จ")
        os.environ['TESSDATA_PREFIX'] = '/tmp/'
        return True
    
    # Try to download from GitHub
    try:
        import urllib.request
        url = 'https://github.com/DoubangoTelecom/tesseractMICR/raw/master/tessdata_best/e13b.traineddata'
        with st.spinner('⏳ กำลังดาวน์โหลด e13b.traineddata... (ครั้งแรกอาจใช้เวลา 30 วินาที)'):
            urllib.request.urlretrieve(url, e13b_path)
        st.success(f"✅ ดาวน์โหลด e13b.traineddata สำเร็จ")
        os.environ['TESSDATA_PREFIX'] = '/tmp/'
        return True
    except Exception as e:
        st.warning(f"⚠️ ไม่สามารถโหลด e13b.traineddata: {e}")
        st.info("ℹ️ จะใช้ eng สำหรับ MICR แทน")
        os.environ['TESSDATA_PREFIX'] = '/tmp/'
        return False

# =====================================================================
# 3. TEXT EXTRACTION WITH TESSERACT
# =====================================================================
def extract_text_tesseract(image):
    """Extract text using Tesseract OCR (Thai + English)"""
    try:
        # Use Thai + English languages
        custom_config = r'--oem 3 --psm 6'
        text = pytesseract.image_to_string(image, lang='tha+eng', config=custom_config)
        return text
    except Exception as e:
        st.error(f"❌ Tesseract OCR Error: {e}")
        return ""

# =====================================================================
# 4. EXTRACT MICR (Bottom of Cheque)
# =====================================================================
def extract_micr(image):
    """Extract MICR code from bottom 15% of cheque image"""
    try:
        # Convert PIL to OpenCV if needed
        if not isinstance(image, np.ndarray):
            image = np.array(image)
            if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        height, width = image.shape[:2]
        bottom_crop = int(height * 0.85)
        micr_region = image[bottom_crop:height, 0:width]
        
        # Convert to grayscale and threshold
        gray = cv2.cvtColor(micr_region, cv2.COLOR_BGR2GRAY) if len(micr_region.shape) == 3 else micr_region
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Try e13b first, fallback to eng
        try:
            micr_text = pytesseract.image_to_string(binary, lang='e13b', config='--psm 7')
        except:
            micr_text = pytesseract.image_to_string(binary, lang='eng', config='--psm 7')
        
        return micr_text.strip()
    except Exception as e:
        return ""

# =====================================================================
# 5. PARSE MICR CODE
# =====================================================================
def parse_micr_thai(micr_text):
    """Parse MICR text to extract cheque components"""
    result = {
        'cheque_number': '',
        'bank_code': '',
        'branch_code': '',
        'account_number': ''
    }
    
    if not micr_text:
        return result
    
    # Extract all number groups (4+ digits)
    number_groups = re.findall(r'\d{4,}', micr_text)
    
    if len(number_groups) >= 4:
        result['cheque_number'] = number_groups[0]
        result['bank_code'] = number_groups[1]
        result['branch_code'] = number_groups[2]
        result['account_number'] = number_groups[3]
    elif len(number_groups) >= 1:
        result['cheque_number'] = number_groups[0]
    
    return result

# =====================================================================
# 6. EXTRACT AMOUNT
# =====================================================================
def extract_amount(text):
    """Extract amount from Thai cheque text"""
    # Pattern: ####,###.## or ####.##
    patterns = [
        r'(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)\s*บาท',
        r'จำนวนเงิน[:\s]*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)',
        r'Amount[:\s]*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)',
        r'\*+\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)\s*\*+',
        r'(\d{1,3}(?:,\d{3})+\.\d{2})'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            amount_str = match.group(1).replace(',', '')
            try:
                return float(amount_str)
            except:
                continue
    return None

# =====================================================================
# 7. EXTRACT DATE
# =====================================================================
def clean_messy_date(text, window_size=8):
    """Extract date using sliding window approach"""
    text = re.sub(r'[^\d/\-\s]', '', text)
    text = re.sub(r'\s+', '', text)
    
    best_date = None
    best_score = 0
    
    for i in range(len(text) - window_size + 1):
        chunk = text[i:i + window_size]
        
        # DD/MM/YY or DD-MM-YY
        match = re.match(r'(\d{2})[/\-](\d{2})[/\-](\d{2})', chunk)
        if match:
            day, month, year = match.groups()
            if 1 <= int(day) <= 31 and 1 <= int(month) <= 12:
                score = 3
                if score > best_score:
                    best_score = score
                    best_date = f"{day}/{month}/20{year}"
    
    return best_date

def extract_date(text):
    """Extract date from cheque text"""
    # Pattern 1: DD/MM/YYYY or DD-MM-YYYY
    pattern1 = r'(\d{1,2})[/\-](\d{1,2})[/\-](\d{2,4})'
    matches = re.findall(pattern1, text)
    for day, month, year in matches:
        try:
            if 1 <= int(day) <= 31 and 1 <= int(month) <= 12:
                year_full = f"20{year}" if len(year) == 2 else year
                return f"{int(day):02d}/{int(month):02d}/{year_full}"
        except:
            continue
    
    # Pattern 2: Thai month names
    thai_months = {
        'ม.ค.': '01', 'ก.พ.': '02', 'มี.ค.': '03', 'เม.ย.': '04',
        'พ.ค.': '05', 'มิ.ย.': '06', 'ก.ค.': '07', 'ส.ค.': '08',
        'ก.ย.': '09', 'ต.ค.': '10', 'พ.ย.': '11', 'ธ.ค.': '12'
    }
    for thai_month, month_num in thai_months.items():
        if thai_month in text:
            pattern = rf'(\d{{1,2}})\s*{re.escape(thai_month)}\s*(\d{{2,4}})'
            match = re.search(pattern, text)
            if match:
                day, year = match.groups()
                year_full = f"20{year}" if len(year) == 2 else year
                return f"{int(day):02d}/{month_num}/{year_full}"
    
    # Pattern 3: Messy date with sliding window
    return clean_messy_date(text)

# =====================================================================
# 8. EXTRACT PAYEE
# =====================================================================
def extract_payee(text):
    """Extract payee name from cheque text"""
    patterns = [
        r'จ่ายให้[:\s]*([^\n\r]{5,50})',
        r'Pay\s+to[:\s]*([^\n\r]{5,50})',
        r'ชื่อ[:\s]*([^\n\r]{5,50})',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            payee = match.group(1).strip()
            # Clean up
            payee = re.sub(r'\*+', '', payee)
            payee = re.sub(r'\s+', ' ', payee)
            return payee[:100]
    
    return None

# =====================================================================
# 9. MAIN PROCESSING FUNCTION
# =====================================================================
def process_cheque(uploaded_file, progress_callback=None):
    """Main function to process cheque image/PDF"""
    try:
        # Reset file pointer
        uploaded_file.seek(0)
        
        # Update progress
        if progress_callback:
            progress_callback(0.1, "📄 กำลังอ่านไฟล์...")
        
        # Convert PDF to image if needed
        if uploaded_file.type == "application/pdf":
            if progress_callback:
                progress_callback(0.2, "📄 กำลังแปลง PDF เป็นรูปภาพ...")
            
            pdf_bytes = uploaded_file.read()
            images = convert_from_bytes(pdf_bytes, dpi=300, first_page=1, last_page=1)
            if not images:
                return {"error": "ไม่สามารถแปลง PDF เป็นรูปภาพได้"}
            image = images[0]
        else:
            image = Image.open(uploaded_file)
        
        # Convert to RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Update progress
        if progress_callback:
            progress_callback(0.3, "🔍 กำลังดึงข้อความด้วย Tesseract OCR...")
        
        # Extract text with Tesseract
        all_text = extract_text_tesseract(image)
        
        # Limit text for display
        display_text = all_text[:500] + "..." if len(all_text) > 500 else all_text
        
        # Update progress
        if progress_callback:
            progress_callback(0.5, "🔢 กำลังดึงข้อมูล MICR...")
        
        # Extract MICR
        micr_text = extract_micr(image)
        micr_data = parse_micr_thai(micr_text)
        
        # Update progress
        if progress_callback:
            progress_callback(0.7, "💰 กำลังวิเคราะห์จำนวนเงิน วันที่ และชื่อผู้รับเงิน...")
        
        # Extract structured data
        amount = extract_amount(all_text)
        date = extract_date(all_text)
        payee = extract_payee(all_text)
        
        # Update progress
        if progress_callback:
            progress_callback(1.0, "✅ ประมวลผลเสร็จสิ้น!")
        
        return {
            "success": True,
            "extracted_text": display_text,
            "full_text_length": len(all_text),
            "amount": amount,
            "date": date,
            "payee": payee,
            "micr_raw": micr_text,
            "cheque_number": micr_data['cheque_number'],
            "bank_code": micr_data['bank_code'],
            "branch_code": micr_data['branch_code'],
            "account_number": micr_data['account_number'],
            "image_size": f"{image.width}x{image.height}",
            "ocr_engine": "Tesseract (tha+eng)"
        }
        
    except Exception as e:
        error_msg = f"❌ เกิดข้อผิดพลาด: {str(e)}\n\n{traceback.format_exc()}"
        return {"error": error_msg}

# =====================================================================
# 10. TEMPLATE FILLING
# =====================================================================
def process_template_filling(template_file, data_file):
    """Fill Excel template with OCR data (XLOOKUP-style)"""
    try:
        # Load template and data
        template_df = pd.read_excel(template_file, sheet_name=None)
        data_df = pd.read_excel(data_file)
        
        # Create lookup dictionary
        lookup_dict = {}
        for _, row in data_df.iterrows():
            key = f"{row.get('cheque_number', '')}_{row.get('amount', '')}"
            lookup_dict[key] = row.to_dict()
        
        # Fill template sheets
        filled_sheets = {}
        for sheet_name, sheet_df in template_df.items():
            if 'cheque_number' in sheet_df.columns and 'amount' in sheet_df.columns:
                for idx, row in sheet_df.iterrows():
                    key = f"{row['cheque_number']}_{row['amount']}"
                    if key in lookup_dict:
                        for col in lookup_dict[key].keys():
                            if col in sheet_df.columns:
                                sheet_df.at[idx, col] = lookup_dict[key][col]
            filled_sheets[sheet_name] = sheet_df
        
        # Save to BytesIO
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            for sheet_name, sheet_df in filled_sheets.items():
                sheet_df.to_excel(writer, sheet_name=sheet_name, index=False)
        output.seek(0)
        
        return {"success": True, "file": output, "sheets": list(filled_sheets.keys())}
        
    except Exception as e:
        return {"error": f"Template filling error: {str(e)}"}

# =====================================================================
# 11. MAIN APP
# =====================================================================
def main():
    st.title("🔍 Thai Cheque OCR System")
    st.caption("Tesseract Edition - Optimized for Streamlit Cloud (ใช้ RAM ~100MB)")
    
    # Download e13b.traineddata
    with st.spinner('⚙️ กำลังติดตั้ง MICR model...'):
        download_e13b_traineddata()
    
    st.success("✅ ระบบพร้อมใช้งาน (Tesseract OCR)")
    
    # Create tabs
    tab1, tab2 = st.tabs(["📸 OCR Extraction", "📋 Template Processing"])
    
    # ===== TAB 1: OCR EXTRACTION =====
    with tab1:
        st.header("📸 ดึงข้อความจากเช็ค")
        
        uploaded_file = st.file_uploader(
            "เลือกไฟล์เช็ค (รองรับ JPG, PNG, PDF)",
            type=['jpg', 'jpeg', 'png', 'pdf'],
            help="อัปโหลดรูปเช็คหรือไฟล์ PDF"
        )
        
        if uploaded_file:
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("📄 ไฟล์ที่อัปโหลด")
                st.info(f"**ชื่อไฟล์:** {uploaded_file.name}\n\n**ขนาด:** {uploaded_file.size / 1024:.1f} KB")
                
                # Show preview
                try:
                    if uploaded_file.type == "application/pdf":
                        st.caption("🔄 ตัวอย่าง PDF (หน้าแรก)")
                        pdf_bytes = uploaded_file.read()
                        images = convert_from_bytes(pdf_bytes, dpi=150, first_page=1, last_page=1)
                        if images:
                            st.image(images[0], use_container_width=True)
                        uploaded_file.seek(0)
                    else:
                        st.image(uploaded_file, use_container_width=True)
                except Exception as e:
                    st.error(f"ไม่สามารถแสดงตัวอย่างได้: {e}")
            
            with col2:
                st.subheader("🚀 เริ่มประมวลผล")
                
                if st.button("🚀 เริ่มประมวลผล", type="primary", use_container_width=True):
                    # Progress bar
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    def update_progress(value, text):
                        progress_bar.progress(value)
                        status_text.text(text)
                    
                    # Process cheque
                    result = process_cheque(uploaded_file, progress_callback=update_progress)
                    
                    if "error" in result:
                        st.error(result["error"])
                    else:
                        st.success("✅ ประมวลผลสำเร็จ!")
                        
                        # Display results
                        st.subheader("📊 ผลลัพธ์")
                        
                        # Key information
                        col_a, col_b, col_c = st.columns(3)
                        with col_a:
                            st.metric("💰 จำนวนเงิน", f"{result['amount']:,.2f}" if result['amount'] else "ไม่พบ")
                        with col_b:
                            st.metric("📅 วันที่", result['date'] if result['date'] else "ไม่พบ")
                        with col_c:
                            st.metric("📝 ชื่อผู้รับเงิน", result['payee'][:20] + "..." if result['payee'] and len(result['payee']) > 20 else result['payee'] or "ไม่พบ")
                        
                        # MICR Data
                        st.subheader("🔢 ข้อมูล MICR")
                        col_1, col_2, col_3, col_4 = st.columns(4)
                        with col_1:
                            st.text_input("เลขที่เช็ค", value=result['cheque_number'], disabled=True)
                        with col_2:
                            st.text_input("รหัสธนาคาร", value=result['bank_code'], disabled=True)
                        with col_3:
                            st.text_input("รหัสสาขา", value=result['branch_code'], disabled=True)
                        with col_4:
                            st.text_input("เลขบัญชี", value=result['account_number'], disabled=True)
                        
                        # Full text
                        with st.expander("📝 ข้อความที่ดึงได้ทั้งหมด (แสดง 500 ตัวอักษรแรก)"):
                            st.text_area(
                                "Extracted Text",
                                value=result['extracted_text'],
                                height=300,
                                disabled=True,
                                label_visibility="collapsed"
                            )
                            st.caption(f"ความยาวข้อความทั้งหมด: {result['full_text_length']} ตัวอักษร")
                        
                        # Technical info
                        with st.expander("ℹ️ ข้อมูลทางเทคนิค"):
                            st.json({
                                "OCR Engine": result['ocr_engine'],
                                "Image Size": result['image_size'],
                                "MICR Raw": result['micr_raw'],
                            })
    
    # ===== TAB 2: TEMPLATE PROCESSING =====
    with tab2:
        st.header("📋 Template Processing")
        st.info("🔄 อัปโหลด Template และ Data file เพื่อทำการ Fill ข้อมูล (XLOOKUP-style)")
        
        col1, col2 = st.columns(2)
        
        with col1:
            template_file = st.file_uploader("เลือก Template File (Excel)", type=['xlsx'], key="template")
        
        with col2:
            data_file = st.file_uploader("เลือก Data File (Excel)", type=['xlsx'], key="data")
        
        if template_file and data_file:
            if st.button("🚀 เริ่มประมวลผล Template", type="primary"):
                with st.spinner('กำลังประมวลผล...'):
                    result = process_template_filling(template_file, data_file)
                
                if "error" in result:
                    st.error(result["error"])
                else:
                    st.success(f"✅ สำเร็จ! ประมวลผล {len(result['sheets'])} sheets")
                    
                    # Download button
                    st.download_button(
                        label="📥 ดาวน์โหลดไฟล์ที่ Fill แล้ว",
                        data=result['file'],
                        file_name="filled_template.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                    
                    st.info(f"**Sheets:** {', '.join(result['sheets'])}")

if __name__ == "__main__":
    main()
