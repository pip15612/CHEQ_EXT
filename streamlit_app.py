
# ==========================================
# 7. STREAMLIT UI - MAIN FUNCTION
# ==========================================

def main():
    st.title("📄 Thai Cheque Processing System")
    st.markdown("Complete solution for Thai cheque OCR and template processing")
    
    # Check e13b.traineddata
    with st.spinner("🔧 Checking system requirements..."):
        success, message = download_e13b_traineddata()
        if success:
            st.success(message)
        else:
            st.error(message)
            st.warning("⚠️ MICR extraction may not work properly without e13b.traineddata")
    
    # Create tabs
    tab1, tab2 = st.tabs(["🔍 OCR Extraction", "📋 Template Processing"])
    
    # ==========================================
    # TAB 1: OCR EXTRACTION
    # ==========================================
    with tab1:
        st.header("🔍 Extract Data from Cheques")
        st.markdown("Upload cheque images/PDFs to extract data using OCR")
        
        # Initialize OCR
        with st.spinner("🚀 Initializing EasyOCR (this may take a minute)..."):
            reader, error = initialize_easyocr()
        
        if error or reader is None:
            st.error(f"❌ Failed to initialize OCR: {error}")
            st.info("💡 Try refreshing the page or contact support")
            return
        else:
            st.success("✅ OCR initialized successfully!")
        
        st.markdown("---")
        st.subheader("📂 Upload Cheque Files")
        
        # File limits warning
        st.info(f"💡 **Tip:** Process up to {MAX_FILES_PER_BATCH} files at a time for best performance")
        
        uploaded_files = st.file_uploader(
            "Choose PDF or image files",
            type=['pdf', 'png', 'jpg', 'jpeg'],
            accept_multiple_files=True,
            key="ocr_uploader"
        )
        
        if uploaded_files:
            file_count = len(uploaded_files)
            st.info(f"📎 {file_count} file(s) uploaded")
            
            # Warning for many files
            if file_count > MAX_FILES_PER_BATCH:
                st.warning(f"⚠️ You uploaded {file_count} files. Consider processing in batches of {MAX_FILES_PER_BATCH} to avoid timeouts.")
            
            if st.button("🚀 Extract Data", type="primary", key="extract_btn"):
                all_results = []
                failed_files = []
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                start_time = time.time()
                
                for idx, uploaded_file in enumerate(uploaded_files):
                    status_text.text(f"📄 Processing {uploaded_file.name} ({idx+1}/{file_count})...")
                    
                    # Save temp file
                    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_path = tmp_file.name
                    
                    try:
                        # Process with progress callback
                        results, error = process_cheque(
                            tmp_path, 
                            reader,
                            progress_callback=lambda msg: status_text.text(f"📄 {uploaded_file.name}: {msg}")
                        )
                        
                        if error:
                            failed_files.append((uploaded_file.name, error))
                            st.warning(f"⚠️ {uploaded_file.name}: {error}")
                        else:
                            all_results.extend(results)
                            st.success(f"✅ {uploaded_file.name} processed")
                    
                    except Exception as e:
                        failed_files.append((uploaded_file.name, str(e)))
                        st.error(f"❌ {uploaded_file.name}: {str(e)}")
                    
                    finally:
                        if os.path.exists(tmp_path):
                            os.unlink(tmp_path)
                    
                    progress_bar.progress((idx + 1) / file_count)
                
                elapsed = time.time() - start_time
                status_text.text(f"✅ Processing complete! ({elapsed:.1f} seconds)")
                
                # Show results
                if all_results:
                    df = pd.DataFrame(all_results)
                    
                    st.markdown("---")
                    st.subheader("📊 Extracted Data")
                    st.metric("Total Records", len(df))
                    
                    if failed_files:
                        st.warning(f"⚠️ {len(failed_files)} file(s) failed to process")
                        with st.expander("Show failed files"):
                            for fname, err in failed_files:
                                st.text(f"❌ {fname}: {err}")
                    
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
                
                elif not failed_files:
                    st.warning("No results extracted from any files.")
        
        # Instructions
        st.markdown("---")
        st.markdown("""
        ### 📝 Instructions
        1. Upload Thai cheque images (PNG, JPG) or PDF files
        2. Click **Extract Data** to process
        3. Download the extracted Excel file
        4. Use the Excel file in **Template Processing** tab
        
        ### ⚠️ Troubleshooting
        - **503 Error:** Try processing fewer files at once (max 5 recommended)
        - **404 Error:** Check if e13b.traineddata downloaded successfully above
        - **Slow processing:** Normal for large images, please be patient
        - **Memory error:** Process files in smaller batches
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
        
        all_files_uploaded = all([template_file, pdf_file, fchn_file, master_file])
        
        if st.button("🚀 Process Template", type="primary", key="process_template_btn", disabled=not all_files_uploaded):
            if all_files_uploaded:
                try:
                    with st.spinner("⏳ Processing template... This may take a moment..."):
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
                        
                        st.success(f"✅ Template processed successfully! ({total_rows} rows filled)")
                        
                        # Download button
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        st.download_button(
                            label="📥 Download Filled Template",
                            data=output,
                            file_name=f"Template_PDF_Filled_{timestamp}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            key="download_template"
                        )
                        
                        st.balloons()
                
                except Exception as e:
                    st.error(f"❌ Error processing template: {str(e)}")
                    st.exception(e)
                    st.info("💡 Please check that all uploaded files are in the correct format")
            else:
                st.warning("⚠️ Please upload all 4 required files!")
        
        # Show file upload status
        if not all_files_uploaded:
            st.info("📋 Please upload all required files to enable processing")
            missing = []
            if not template_file:
                missing.append("Template File")
            if not pdf_file:
                missing.append("Extracted Data")
            if not fchn_file:
                missing.append("FCHN File")
            if not master_file:
                missing.append("Master File")
            
            if missing:
                st.warning(f"Missing: {', '.join(missing)}")
        
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
        
        ### 📊 Column Mappings
        **TR Teams Sheet:**
        - Company Code → Column A
        - Business Partner → Column B
        - Document Date → Column F
        - Posting Date → Column J
        - Amount → Column H
        - Assignment → Column O (CHQ + Cheque Number)
        - Bank Account → Column AE
        
        **Cash Teams Sheet:**
        - Company Code → Column A
        - Business Place → Column B
        - Company Name → Column C (from FCHN)
        - House Bank → Column D (from FCHN)
        - Start Date → Column E
        - Payment Amount → Column F
        - Bank Account → Column G
        - Assignment → Column H (CHQ + Cheque Number)
        - Business Partner → Column I
        """)
        
        # Footer
        st.markdown("---")
        st.markdown("""
        <div style='text-align: center; color: gray; padding: 20px;'>
            <p>Thai Cheque Processing System v2.0</p>
            <p>Built with Streamlit • EasyOCR • Tesseract</p>
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
