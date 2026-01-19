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
import time
import shutil

# =============================================================================
# Configuration
# =============================================================================
DEBUG = False
MAX_FILES_PER_BATCH = 5
TESSERACT_CMD = r'C:\Program Files\Tesseract-OCR\tesseract.exe' if os.name == 'nt' else 'tesseract'

# =============================================================================
# Helper Functions
# =============================================================================
def download_e13b_traineddata():
    """โหลด e13b.traineddata จาก repo แล้วคัดลอกไปยัง /tmp/tessdata"""
    tessdata_path = '/tmp/tessdata'
    
    try:
        os.makedirs(tessdata_path, exist_ok=True)
    except Exception as e:
        st.warning(f'⚠️ ไม่สามารถสร้าง tessdata folder: {e}')
        return False
    
    e13b_file = os.path.join(tessdata_path, 'e13b.traineddata')
    
    # ตั้งค่า TESSDATA_PREFIX ให้ Tesseract รู้ว่าไฟล์อยู่ที่ไหน
    os.environ['TESSDATA_PREFIX'] = '/tmp/'
    
    if os.path.exists(e13b_file):
        st.success('✅ MICR model พร้อมใช้งานแล้ว')
        return True
    
    st.info('🔄 กำลังโหลด MICR recognition model...')
    
    # ไฟล์อยู่ที่ root ของ repo
    local_e13b = 'e13b.traineddata'
    
    try:
        if os.path.exists(local_e13b):
            shutil.copy(local_e13b, e13b_file)
            st.success('✅ โหลด MICR model สำเร็จ!')
            return True
        else:
            st.warning('⚠️ ไม่พบ e13b.traineddata ใน repo')
            # ลองดาวน์โหลดจาก GitHub ถ้าไม่มีในไฟล์
            url = "https://github.com/DoubangoTelecom/tesseractMICR/raw/master/tessdata_best/e13b.traineddata"
            r = requests.get(url, timeout=60)
            if r.status_code == 200:
                with open(e13b_file, 'wb') as f:
                    f.write(r.content)
                st.success('✅ ดาวน์โหลด MICR model สำเร็จ!')
                return True
            return False
    except Exception as e:
        st.warning(f'⚠️ ไม่สามารถโหลด e13b.traineddata ได้: {str(e)}')
        return False

@st.cache_resource
@st.cache_resource(show_spinner=False)
def initialize_easyocr():
    """Initialize EasyOCR reader (cached)"""
    try:
        with st.spinner('🔄 กำลังโหลด OCR Model ครั้งแรก... (ใช้เวลา 2-3 นาที) กรุณารอสักครู่'):
            reader = easyocr.Reader(['th', 'en'], gpu=False, verbose=False, download_enabled=True)
        st.success('✅ โหลด EasyOCR สำเร็จ!')
        return reader
    except Exception as e:
        st.error(f'❌ ไม่สามารถโหลด EasyOCR ได้: {e}')
        st.info('💡 ทดลองใช้ Tesseract แทน...')
        return None

def clean_messy_date(text):
    """แยกวันที่จาก text ที่ยุ่งเหยิง โดยใช้ sliding window หา pattern 8 หลัก"""
    if not text or len(text) < 8:
        return None
    
    text_clean = re.sub(r'[^\d]', '', text)
    
    for i in range(len(text_clean) - 7):
        segment = text_clean[i:i+8]
        if len(segment) == 8:
            day = segment[:2]
            month = segment[2:4]
            year = segment[4:8]
            
            try:
                day_int = int(day)
                month_int = int(month)
                year_int = int(year)
                
                if 1 <= day_int <= 31 and 1 <= month_int <= 12:
                    if 2500 <= year_int <= 2600:
                        year_int -= 543
                    elif year_int < 100:
                        year_int += 2000
                    
                    if 1900 <= year_int <= 2100:
                        return f"{day}/{month}/{year_int}"
            except:
                continue
    
    return None

def extract_micr(image_np):
    """ดึงข้อมูล MICR จากด้านล่างของเช็ค"""
    try:
        # ตั้งค่า tesseract command
        if os.name == 'nt' and os.path.exists(TESSERACT_CMD):
            pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
        
        # บน Linux/Streamlit ใช้ /tmp/tessdata
        if os.name != 'nt':
            os.environ['TESSDATA_PREFIX'] = '/tmp/'
        
        height, width = image_np.shape[:2]
        micr_roi = image_np[int(height * 0.85):height, :]
        gray = cv2.cvtColor(micr_roi, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # ลองใช้ e13b ก่อน ถ้าไม่ได้ใช้ eng แทน
        try:
            micr_text = pytesseract.image_to_string(binary, lang='e13b', config='--psm 6')
        except:
            # Fallback ใช้ eng
            micr_text = pytesseract.image_to_string(binary, lang='eng', config='--psm 6 -c tessedit_char_whitelist=0123456789')
        
        return micr_text.strip()
    except Exception as e:
        if DEBUG:
            st.warning(f'MICR extraction error: {e}')
        return ''

def parse_micr_thai(micr_text):
    """แปลง MICR text เป็นข้อมูล Cheque Number, Bank Code, Branch, Account"""
    result = {
        'cheque_number': '',
        'bank_code': '',
        'branch_code': '',
        'account_number': ''
    }
    
    if not micr_text:
        return result
    
    parts = re.findall(r'[0-9]+', micr_text)
    if len(parts) >= 4:
        result['cheque_number'] = parts[0]
        result['bank_code'] = parts[1]
        result['branch_code'] = parts[2]
        result['account_number'] = parts[3]
    
    return result

def process_cheque(uploaded_file, reader, progress_callback=None):
    """ประมวลผลไฟล์เช็ค (PDF/Image)"""
    try:
        start_time = time.time()
        
        # อ่านไฟล์
        file_bytes = uploaded_file.read()
        file_ext = uploaded_file.name.lower().split('.')[-1]
        
        if progress_callback:
            progress_callback(0.2, 'กำลังแปลงไฟล์...')
        
        # แปลง PDF เป็น Image
        if file_ext == 'pdf':
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(file_bytes)
                tmp_path = tmp_file.name
            
            try:
                images = convert_from_path(tmp_path, dpi=300)
                image_np = cv2.cvtColor(np.array(images[0]), cv2.COLOR_RGB2BGR)
            finally:
                os.unlink(tmp_path)
        else:
            nparr = np.frombuffer(file_bytes, np.uint8)
            image_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image_np is None:
            return {'error': 'ไม่สามารถอ่านไฟล์ได้'}
        
        if progress_callback:
            progress_callback(0.4, 'กำลังทำ OCR...')
        
        # OCR ด้วย EasyOCR
        results = reader.readtext(image_np)
        
        if progress_callback:
            progress_callback(0.6, 'กำลังดึง MICR...')
        
        # ดึง MICR
        micr_text = extract_micr(image_np)
        micr_data = parse_micr_thai(micr_text)
        
        if progress_callback:
            progress_callback(0.8, 'กำลังวิเคราะห์ข้อมูล...')
        
        # ดึงข้อมูลจาก OCR
        all_text = ' '.join([text for _, text, _ in results])
        
        # หาจำนวนเงิน
        amount_patterns = [
            r'(?:บาท|BAHT)[^\d]*([\d,]+\.?\d*)',
            r'([\d,]+\.?\d*)\s*(?:บาท|BAHT)',
            r'THB\s*([\d,]+\.?\d*)',
        ]
        
        amount = ''
        for pattern in amount_patterns:
            match = re.search(pattern, all_text, re.IGNORECASE)
            if match:
                amount = match.group(1).replace(',', '')
                break
        
        # หาวันที่
        date_str = ''
        date_patterns = [
            r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
            r'(\d{8})',
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, all_text)
            if match:
                raw_date = match.group(1)
                date_str = clean_messy_date(raw_date)
                if date_str:
                    break
        
        # หาชื่อผู้รับเงิน (บรรทัดที่มี "จ่าย" หรือ "PAY")
        payee = ''
        for _, text, _ in results:
            if 'จ่าย' in text or 'PAY' in text.upper():
                payee = text
                break
        
        elapsed_time = time.time() - start_time
        
        if progress_callback:
            progress_callback(1.0, 'เสร็จสิ้น')
        
        return {
            'filename': uploaded_file.name,
            'cheque_number': micr_data['cheque_number'],
            'bank_code': micr_data['bank_code'],
            'branch_code': micr_data['branch_code'],
            'account_number': micr_data['account_number'],
            'amount': amount,
            'date': date_str,
            'payee': payee,
            'all_text': all_text,
            'processing_time': f'{elapsed_time:.2f}s'
        }
        
    except Exception as e:
        return {'error': str(e), 'filename': uploaded_file.name}

def process_template_filling(template_file, data_file):
    """เติมข้อมูลลงใน Template (TR & Cash) ด้วย XLOOKUP logic"""
    try:
        # อ่าน Template
        template_df = pd.read_excel(template_file, sheet_name=None, engine='openpyxl')
        
        # อ่าน Data Source
        data_df = pd.read_excel(data_file, engine='openpyxl')
        
        if 'Ref.No.' not in data_df.columns or 'Trading Name' not in data_df.columns:
            return None, 'Data file ต้องมี columns: Ref.No. และ Trading Name'
        
        # สร้าง lookup dictionary
        lookup_dict = {}
        for idx, row in data_df.iterrows():
            ref_no = str(row.get('Ref.No.', '')).strip()
            if ref_no and ref_no != 'nan':
                lookup_dict[ref_no] = {
                    'Trading Name': row.get('Trading Name', ''),
                    'TAX NAME': row.get('TAX NAME', ''),
                    'Remark': row.get('Remark', ''),
                    'Note': row.get('Note', '')
                }
        
        # Process แต่ละ Sheet
        output_sheets = {}
        for sheet_name, sheet_df in template_df.items():
            if 'Ref.No.' in sheet_df.columns:
                # XLOOKUP logic
                for idx, row in sheet_df.iterrows():
                    ref_no = str(row.get('Ref.No.', '')).strip()
                    if ref_no in lookup_dict:
                        lookup_data = lookup_dict[ref_no]
                        for col, val in lookup_data.items():
                            if col in sheet_df.columns:
                                sheet_df.at[idx, col] = val
            
            output_sheets[sheet_name] = sheet_df
        
        # สร้าง Excel file ใน memory
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            for sheet_name, sheet_df in output_sheets.items():
                sheet_df.to_excel(writer, sheet_name=sheet_name, index=False)
        
        output.seek(0)
        return output, None
        
    except Exception as e:
        return None, str(e)

# =============================================================================
# Main Streamlit App
# =============================================================================
def main():
    st.title('🏦 Thai Cheque OCR System')
    st.markdown('ระบบดึงข้อมูลจากเช็คภาษาไทย (OCR + MICR) และเติม Template')
    
    # Initialize
    download_e13b_traineddata()
    reader = initialize_easyocr()
    
    if reader is None:
        st.error('❌ ไม่สามารถเริ่มระบบ OCR ได้ กรุณาลองใหม่อีกครั้ง')
        return
    
    # Tabs
    tab1, tab2 = st.tabs(['📸 OCR Extraction', '📋 Template Processing'])
    
    # ==================== Tab 1: OCR Extraction ====================
    with tab1:
        st.header('ดึงข้อมูลจากเช็ค')
        
        uploaded_files = st.file_uploader(
            'อัพโหลดไฟล์เช็ค (PDF/JPG/PNG)',
            type=['pdf', 'jpg', 'jpeg', 'png'],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            if len(uploaded_files) > MAX_FILES_PER_BATCH:
                st.warning(f'⚠️ จำกัดไม่เกิน {MAX_FILES_PER_BATCH} ไฟล์ต่อครั้ง (เพื่อประสิทธิภาพ)')
                uploaded_files = uploaded_files[:MAX_FILES_PER_BATCH]
            
            if st.button('🚀 เริ่มประมวลผล'):
                results = []
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for idx, file in enumerate(uploaded_files):
                    status_text.text(f'กำลังประมวลผล: {file.name} ({idx+1}/{len(uploaded_files)})')
                    
                    def update_progress(pct, msg):
                        overall_pct = (idx + pct) / len(uploaded_files)
                        progress_bar.progress(overall_pct)
                        status_text.text(f'{msg} - {file.name}')
                    
                    result = process_cheque(file, reader, update_progress)
                    results.append(result)
                
                progress_bar.progress(1.0)
                status_text.text('✅ เสร็จสิ้น!')
                
                # แสดงผลลัพธ์
                st.success(f'ประมวลผลเสร็จสิ้น: {len(results)} ไฟล์')
                
                # สร้าง DataFrame
                results_df = pd.DataFrame(results)
                
                # แสดงตาราง
                st.dataframe(results_df, use_container_width=True)
                
                # ดาวน์โหลด CSV
                csv = results_df.to_csv(index=False, encoding='utf-8-sig')
                st.download_button(
                    label='📥 ดาวน์โหลด CSV',
                    data=csv,
                    file_name=f'cheque_ocr_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
                    mime='text/csv'
                )
    
    # ==================== Tab 2: Template Processing ====================
    with tab2:
        st.header('เติมข้อมูลลง Template (XLOOKUP)')
        
        col1, col2 = st.columns(2)
        
        with col1:
            template_file = st.file_uploader(
                '📄 Template File (TR & Cash)',
                type=['xlsx'],
                key='template'
            )
        
        with col2:
            data_file = st.file_uploader(
                '📊 Data Source File',
                type=['xlsx'],
                key='data'
            )
        
        if template_file and data_file:
            if st.button('🔄 เติมข้อมูล'):
                with st.spinner('กำลังประมวลผล...'):
                    output, error = process_template_filling(template_file, data_file)
                    
                    if error:
                        st.error(f'❌ เกิดข้อผิดพลาด: {error}')
                    else:
                        st.success('✅ เติมข้อมูลสำเร็จ!')
                        
                        st.download_button(
                            label='📥 ดาวน์โหลดไฟล์ผลลัพธ์',
                            data=output,
                            file_name=f'filled_template_{datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx',
                            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                        )
        
        # คำแนะนำ
        with st.expander('ℹ️ วิธีใช้งาน'):
            st.markdown("""
            **Tab 1: OCR Extraction**
            - อัพโหลดไฟล์เช็ค (PDF/Image)
            - ระบบจะดึงข้อมูล: เลขเช็ค, รหัสธนาคาร, จำนวนเงิน, วันที่, ผู้รับเงิน
            - ดาวน์โหลดผลลัพธ์เป็น CSV
            
            **Tab 2: Template Processing**
            - อัพโหลด Template File (Excel ที่มี Sheet TR & Cash)
            - อัพโหลด Data Source (Excel ที่มี Ref.No. และ Trading Name)
            - ระบบจะเติมข้อมูลแบบ XLOOKUP อัตโนมัติ
            - ดาวน์โหลดไฟล์ที่เติมข้อมูลแล้ว
            
            **หมายเหตุ:**
            - จำกัด 5 ไฟล์ต่อครั้งสำหรับ OCR (เพื่อประสิทธิภาพ)
            - รองรับ Thai & English text
            - ใช้ MICR recognition สำหรับเลขเช็ค (มี fallback ถ้าโหลดไม่ได้)
            """)

if __name__ == '__main__':
    main()

