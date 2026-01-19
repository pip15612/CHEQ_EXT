import streamlit as st
import cv2
import numpy as np
import pytesseract
from pdf2image import convert_from_bytes
from PIL import Image
import pandas as pd
import re
from io import BytesIO
import os
import traceback
import shutil
from pathlib import Path

# ========== ตั้งค่า Tesseract (ไม่ต้องเปลี่ยน TESSDATA_PREFIX) ==========
def setup_tessdata():
    """จัดการ e13b.traineddata โดยไม่รบกวนระบบ tessdata หลัก"""
    try:
        # สร้างโฟลเดอร์ /tmp/tessdata สำหรับ e13b
        tessdata_dir = Path("/tmp/tessdata")
        tessdata_dir.mkdir(parents=True, exist_ok=True)
        
        # คัดลอก e13b.traineddata จาก repo ไปยัง /tmp/tessdata
        local_e13b = Path("e13b.traineddata")
        target_e13b = tessdata_dir / "e13b.traineddata"
        
        if local_e13b.exists() and not target_e13b.exists():
            shutil.copy(local_e13b, target_e13b)
            st.success(f"✅ คัดลอก e13b.traineddata ไปยัง {target_e13b}")
        elif target_e13b.exists():
            st.info(f"✅ e13b.traineddata พร้อมใช้งานที่ {target_e13b}")
        else:
            st.warning("⚠️ ไม่พบ e13b.traineddata (จะใช้ eng สำหรับ MICR)")
        
        return str(tessdata_dir)
    except Exception as e:
        st.error(f"❌ ไม่สามารถตั้งค่า tessdata: {str(e)}")
        return None

# เรียกใช้ตอนเริ่มต้น
CUSTOM_TESSDATA = setup_tessdata()

# ========== ฟังก์ชันประมวลผลเช็ค ==========
def extract_text_from_image(image):
    """ดึงข้อความจากภาพเช็คด้วย Tesseract (ไทย+อังกฤษ)"""
    try:
        # ใช้ระบบ tessdata เริ่มต้น (ไม่ระบุ --tessdata-dir)
        text = pytesseract.image_to_string(image, lang='tha+eng', config='--psm 6')
        return text
    except Exception as e:
        st.error(f"❌ Tesseract OCR Error: {str(e)}")
        return ""

def extract_micr(image):
    """ดึง MICR code จากส่วนล่างของเช็ค"""
    try:
        h, w = image.shape[:2]
        roi = image[int(h * 0.85):h, :]  # ส่วนล่าง 15%
        
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # ลองใช้ e13b ก่อน (ถ้ามี)
        micr_text = ""
        if CUSTOM_TESSDATA:
            try:
                micr_text = pytesseract.image_to_string(
                    binary, 
                    config=f'--psm 6 --tessdata-dir {CUSTOM_TESSDATA} -l e13b'
                )
            except:
                pass
        
        # ถ้าไม่ได้ผล ใช้ eng
        if not micr_text.strip():
            micr_text = pytesseract.image_to_string(binary, lang='eng', config='--psm 6')
        
        return micr_text.strip()
    except Exception as e:
        st.warning(f"⚠️ ไม่สามารถดึง MICR: {str(e)}")
        return ""

def parse_micr_thai(micr_text):
    """แยก MICR เป็นส่วนต่างๆ"""
    parts = re.findall(r'\d{4,}', micr_text)
    result = {
        "cheque_number": parts[0] if len(parts) > 0 else "",
        "bank_code": parts[1] if len(parts) > 1 else "",
        "branch_code": parts[2] if len(parts) > 2 else "",
        "account_number": parts[3] if len(parts) > 3 else ""
    }
    return result

def clean_messy_date(text):
    """ดึงวันที่แบบ sliding window"""
    text = re.sub(r'[^\d]', '', text)
    
    for i in range(len(text) - 7):
        window = text[i:i+8]
        if len(window) == 8:
            try:
                day = int(window[:2])
                month = int(window[2:4])
                year = int(window[4:])
                
                if 1 <= day <= 31 and 1 <= month <= 12 and 2000 <= year <= 2100:
                    return f"{day:02d}/{month:02d}/{year}"
            except:
                continue
    return "ไม่พบวันที่"

def process_cheque(uploaded_file):
    """ประมวลผลไฟล์เช็ค (PDF/Image)"""
    try:
        uploaded_file.seek(0)
        file_bytes = uploaded_file.read()
        
        # แปลง PDF เป็นภาพ
        if uploaded_file.name.lower().endswith('.pdf'):
            images = convert_from_bytes(file_bytes, dpi=300)
            image = images[0]
        else:
            image = Image.open(BytesIO(file_bytes))
        
        # แปลงเป็น OpenCV format
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # ดึงข้อความหลัก
        st.info("🔍 กำลังดึงข้อความจากเช็ค...")
        all_text = extract_text_from_image(cv_image)
        
        # ดึง MICR
        st.info("🔍 กำลังดึง MICR code...")
        micr_text = extract_micr(cv_image)
        micr_data = parse_micr_thai(micr_text)
        
        # ดึงจำนวนเงิน
        amount_match = re.search(r'[\*\s]*([\d,]+\.\d{2})[\*\s]*', all_text)
        amount = amount_match.group(1).replace(',', '') if amount_match else "ไม่พบ"
        
        # ดึงวันที่
        date_str = clean_messy_date(all_text)
        
        # ดึงผู้รับเงิน
        thai_lines = [line for line in all_text.split('\n') if re.search(r'[ก-๙]', line)]
        payee = thai_lines[0] if thai_lines else "ไม่พบ"
        
        result = {
            "หมายเลขเช็ค": micr_data["cheque_number"],
            "รหัสธนาคาร": micr_data["bank_code"],
            "รหัสสาขา": micr_data["branch_code"],
            "เลขบัญชี": micr_data["account_number"],
            "จำนวนเงิน": amount,
            "วันที่": date_str,
            "ผู้รับเงิน": payee,
            "MICR (ดิบ)": micr_text[:100]
        }
        
        return result, cv_image
        
    except Exception as e:
        st.error(f"❌ เกิดข้อผิดพลาด: {str(e)}")
        st.code(traceback.format_exc())
        return None, None

def process_template_filling(data_file, template_file):
    """ประมวลผล Template Filling (XLOOKUP-style)"""
    try:
        # อ่านข้อมูล
        df_data = pd.read_excel(data_file)
        
        # อ่าน template (มี 2 sheets: TR และ Cash)
        with pd.ExcelFile(template_file) as xls:
            df_tr = pd.read_excel(xls, 'TR')
            df_cash = pd.read_excel(xls, 'Cash')
        
        # สร้าง lookup dictionary
        lookup_dict = df_data.set_index('รหัสบุคคล')[['ชื่อ', 'จำนวนเงิน', 'หมายเหตุ']].to_dict('index')
        
        # ฟังก์ชัน XLOOKUP
        def xlookup(code, field):
            if pd.notna(code) and code in lookup_dict:
                return lookup_dict[code].get(field, '')
            return ''
        
        # เติมข้อมูลใน TR sheet
        df_tr['ชื่อ'] = df_tr['รหัสบุคคล'].apply(lambda x: xlookup(x, 'ชื่อ'))
        df_tr['จำนวนเงิน'] = df_tr['รหัสบุคคล'].apply(lambda x: xlookup(x, 'จำนวนเงิน'))
        df_tr['หมายเหตุ'] = df_tr['รหัสบุคคล'].apply(lambda x: xlookup(x, 'หมายเหตุ'))
        
        # เติมข้อมูลใน Cash sheet
        df_cash['ชื่อ'] = df_cash['รหัสบุคคล'].apply(lambda x: xlookup(x, 'ชื่อ'))
        df_cash['จำนวนเงิน'] = df_cash['รหัสบุคคล'].apply(lambda x: xlookup(x, 'จำนวนเงิน'))
        df_cash['หมายเหตุ'] = df_cash['รหัสบุคคล'].apply(lambda x: xlookup(x, 'หมายเหตุ'))
        
        # บันทึกเป็นไฟล์ Excel
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

# ========== หน้าจอหลัก ==========
def main():
    st.set_page_config(page_title="Thai Cheque OCR", page_icon="🏦", layout="wide")
    
    st.title("🏦 ระบบดึงข้อความจากเช็คไทย (Tesseract OCR)")
    st.caption("📌 ใช้ Tesseract OCR (ไทย + อังกฤษ + MICR e13b)")
    
    # สร้างแท็บ
    tab1, tab2 = st.tabs(["📄 ดึงข้อความจากเช็ค", "📊 เติมข้อมูล Template"])
    
    # ===== แท็บ 1: ดึงข้อความจากเช็ค =====
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
                # แสดงตัวอย่างไฟล์ (แปลง PDF เป็นรูปภาพก่อน)
                try:
                    uploaded_file.seek(0)
                    if uploaded_file.name.lower().endswith('.pdf'):
                        images = convert_from_bytes(uploaded_file.read(), dpi=150)
                        st.image(images[0], caption="ไฟล์ที่อัปโหลด (หน้าแรก)", use_container_width=True)
                    else:
                        st.image(uploaded_file, caption="ไฟล์ที่อัปโหลด", use_container_width=True)
                except Exception as e:
                    st.warning(f"⚠️ ไม่สามารถแสดงตัวอย่างไฟล์: {str(e)}")
            
            with col2:
                if st.button("🚀 เริ่มประมวลผล", type="primary", use_container_width=True):
                    with st.spinner("⏳ กำลังประมวลผล..."):
                        result, _ = process_cheque(uploaded_file)
                        
                        if result:
                            st.success("✅ ประมวลผลสำเร็จ!")
                            
                            # แสดงผลในตาราง
                            df_result = pd.DataFrame([result]).T
                            df_result.columns = ['ข้อมูล']
                            st.dataframe(df_result, use_container_width=True)
                            
                            # ปุ่มดาวน์โหลด CSV
                            csv = pd.DataFrame([result]).to_csv(index=False, encoding='utf-8-sig')
                            st.download_button(
                                label="📥 ดาวน์โหลด CSV",
                                data=csv,
                                file_name="cheque_data.csv",
                                mime="text/csv"
                            )
    
    # ===== แท็บ 2: Template Filling =====
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
