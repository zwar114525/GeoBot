# GeoBot Setup Guide

## Quick Start

### 1. Environment Setup

```bash
# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# or
source venv/bin/activate  # Mac/Linux

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure API Key

Edit `.env` file with your OpenRouter API key:

```env
OPENROUTER_API_KEY=sk-or-v1-your-key-here
LLM_PROVIDER=openai
PRIMARY_MODEL=deepseek/deepseek-chat-v3-0324
```

### 3. Start Qdrant Vector Database

```bash
docker pull qdrant/qdrant
docker run -d -p 6333:6333 -p 6334:6334 \
  -v $(pwd)/qdrant_storage:/qdrant/storage \
  --name geotech_qdrant \
  qdrant/qdrant
```

### 4. Run the Application

```bash
streamlit run app.py
```

---

## OCR Setup for Scanned Documents

### Windows

1. **Install Tesseract OCR:**
   - Download installer: https://github.com/UB-Mannheim/tesseract/wiki
   - Run installer, select additional languages during installation
   - **Important:** Select Chinese Simplified (chi_sim) for HK documents

2. **Add Tesseract to PATH:**
   ```bash
   # Add to system PATH or set environment variable
   setx TESSDATA_PREFIX "C:\Program Files\Tesseract-OCR\tessdata"
   ```

3. **Verify installation:**
   ```bash
   tesseract --version
   ```

### macOS

```bash
# Install via Homebrew
brew install tesseract
brew install tesseract-lang  # For additional languages

# Verify
tesseract --version
```

### Linux

```bash
# Ubuntu/Debian
sudo apt-get install tesseract-ocr
sudo apt-get install libtesseract-dev
sudo apt-get install tesseract-ocr-chi-sim  # Chinese Simplified

# RHEL/CentOS
sudo yum install tesseract
sudo yum install tesseract-langpack-chi_sim
```

### Test OCR

```python
import pytesseract
print(pytesseract.get_tesseract_version())

# Test Chinese OCR
from PIL import Image
img = Image.open("test_chinese.png")
text = pytesseract.image_to_string(img, lang='chi_sim')
print(text)
```

---

## Document Ingestion

### Ingest Single Document

```bash
python -m src.ingestion.ingest single \
  --path "path/to/document.pdf" \
  --id "hk_cop_2017" \
  --name "HK Code of Practice 2017" \
  --type "code"
```

### Ingest Directory

```bash
python -m src.ingestion.ingest directory \
  --path "data/raw_documents" \
  --type "code"
```

### Using the Web UI

1. Go to **📂 Document Manager** tab
2. Upload PDF document
3. Enter document name, ID, and type
4. Click **📥 Ingest Document**

---

## Configuration Options

### Environment Variables (.env)

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENROUTER_API_KEY` | OpenRouter API key | Required |
| `LLM_PROVIDER` | Provider (openai/google) | `openai` |
| `PRIMARY_MODEL` | Main LLM model | `deepseek/deepseek-chat-v3-0324` |
| `OCR_ENABLED` | Enable OCR for scanned docs | `true` |
| `OCR_LANGUAGE` | OCR languages | `eng+chi_sim` |
| `DOCLING_MAX_PAGES` | Max pages for Docling | `300` |
| `CHUNK_MAX_CHARS` | Max chars per chunk | `1500` |

### Model Options (OpenRouter)

| Model | ID | Best For |
|-------|-----|----------|
| DeepSeek V3 | `deepseek/deepseek-chat-v3-0324` | Primary reasoning |
| DeepSeek R1 | `deepseek/deepseek-r1` | Complex analysis |
| Qwen 2.5 72B | `qwen/qwen-2.5-72b-instruct` | Bilingual (EN/ZH) |
| Llama 3.1 8B | `meta-llama/llama-3.1-8b-instruct:free` | Free tier |

---

## Features

### 📚 Knowledge Q&A
- Semantic search across geotechnical codes
- Citation-backed answers
- Equation-focused mode
- Multi-filter retrieval (document, clause, content type)

### 📝 Report Generator
- Conversational project data collection
- Automatic calculation execution
- Structured report generation
- HK geotechnical style

### ✅ Submission Checker
- PDF report validation
- Completeness checks
- Code compliance verification
- Consistency analysis

### 📂 Document Manager
- Structure-aware PDF chunking
- OCR for scanned documents
- Knowledge base inspection
- Retrieval verification

---

## Troubleshooting

### API Key Issues
```
Error: API key expired. Please renew the API key.
```
**Solution:** Update `OPENROUTER_API_KEY` in `.env`

### OCR Not Available
```
WARNING: OCR not available - install pytesseract and PIL
```
**Solution:** Install Tesseract OCR (see OCR Setup section)

### Qdrant Connection Failed
```
Error: Connection refused to localhost:6333
```
**Solution:** Start Qdrant Docker container

### Docling Not Available
```
WARNING: Docling not available. Using fallback PDF processing.
```
**Solution:** Install docling: `pip install docling`

---

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test module
pytest tests/test_json_validation.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

---

## Project Structure

```
GeoBot/
├── app.py                    # Streamlit web UI
├── .env                      # Environment variables
├── requirements.txt          # Python dependencies
├── config/
│   └── settings.py          # Configuration
├── src/
│   ├── agents/              # AI agents
│   │   ├── qa_agent.py
│   │   ├── designer_agent.py
│   │   └── validator_agent.py
│   ├── ingestion/           # Document processing
│   │   ├── pdf_processor.py
│   │   └── pdf_processor_enhanced.py
│   ├── calculations/        # Engineering calculations
│   ├── vectordb/            # Vector database
│   ├── utils/               # Utilities
│   │   ├── llm_client.py
│   │   ├── embeddings.py
│   │   ├── json_validator.py
│   │   └── citation_verifier.py
│   └── schemas/             # Pydantic schemas
├── tests/                   # Test suite
└── data/                    # Document storage
```
