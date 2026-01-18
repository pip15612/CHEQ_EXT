# Thai Cheque OCR Processor

A Streamlit web application for extracting data from Thai cheque images and PDFs using OCR technology.

## Features

- 📄 Extract data from Thai cheques (images and PDFs)
- 🔍 OCR using EasyOCR (Thai + English)
- 🏦 MICR code recognition for bank details
- 📊 Export results to Excel with proper formatting
- 🚀 Easy deployment on Streamlit Cloud

## Extracted Fields

- Date
- Payee name
- Amount (numeric and Thai text)
- Cheque number
- Bank code
- Branch code
- Account number
- Cheque digit

## Local Installation

### Prerequisites

- Python 3.9 or higher
- Tesseract OCR
- Poppler (for PDF processing)

### Setup

1. Clone this repository:
```bash
git clone <your-repo-url>
cd EXCEL_LOOKUPDATA
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

3. Install system dependencies:

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install tesseract-ocr tesseract-ocr-tha tesseract-ocr-eng poppler-utils libgl1-mesa-glx libglib2.0-0
```

**macOS:**
```bash
brew install tesseract tesseract-lang poppler
```

**Windows:**
- Install Tesseract: https://github.com/UB-Mannheim/tesseract/wiki
- Install Poppler: https://github.com/oschwartz10612/poppler-windows/releases

4. Run the application:
```bash
streamlit run streamlit_app.py
```

## Deployment on Streamlit Cloud

### Step 1: Prepare Your Repository

1. Create a GitHub repository
2. Upload these files:
   - `streamlit_app.py`
   - `requirements.txt`
   - `packages.txt`
   - `.streamlit/config.toml`
   - `README.md`

### Step 2: Deploy on Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with your GitHub account
3. Click "New app"
4. Select your repository
5. Set main file path: `streamlit_app.py`
6. Click "Deploy"

### Important Notes for Deployment

- The app will automatically download the `e13b.traineddata` file for MICR recognition on first run
- GPU is disabled for compatibility (uses CPU for OCR)
- Maximum file upload size is set to 200MB (configurable in config.toml)

## Configuration

You can modify settings in `.streamlit/config.toml`:

- `maxUploadSize`: Maximum file upload size in MB
- `primaryColor`: Theme primary color
- Other theme and server settings

## Usage

1. Upload Thai cheque images (PNG, JPG) or PDF files
2. Click "Process Files"
3. Review extracted data in the table
4. Download results as Excel file

## Troubleshooting

### OCR Issues

If OCR is not working properly:
- Ensure Tesseract is installed correctly
- Check that Thai language data is available
- Verify image quality (300 DPI recommended)

### Memory Issues

For large files or multiple files:
- Process files in smaller batches
- Increase server memory limits in Streamlit Cloud settings

### e13b.traineddata Download Issues

If the MICR font download fails:
- Check internet connection
- Manually download from: https://github.com/DoubangoTelecom/tesseractMICR/raw/master/tessdata_best/e13b.traineddata
- Place in `./tessdata/` directory

## Technical Stack

- **Streamlit**: Web framework
- **EasyOCR**: Thai/English text recognition
- **Tesseract**: MICR code recognition (e13b font)
- **OpenCV**: Image processing
- **pandas**: Data manipulation
- **pdf2image**: PDF to image conversion

## License

MIT License

## Support

For issues and questions, please open an issue on GitHub.

## Acknowledgments

- EasyOCR for Thai text recognition
- Tesseract OCR for MICR recognition
- DoubangoTelecom for e13b.traineddata file
