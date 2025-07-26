# Bengali RAG System for HSC Literature

This is a Retrieval-Augmented Generation (RAG) chatbot designed for HSC Bangla 1st Paper students and teachers. It answers questions directly from HSC26-Bangla1st_Paper. 
It builds a multilingual RAG system for HSC Bangla 1st Paper literature using LangChain, FAISS, and Streamlit. It processes Bengali PDFs through Tesseract OCR, creates vector embeddings with multilingual models, and answers questions in Bengali about literary characters and plot details. The system extracts text from scanned PDFs, chunks content with custom Bengali separators, stores embeddings in FAISS vectorstore, and uses Groq's LLM API for answer generation. Includes both a Streamlit web interface and FastAPI endpoints for programmatic access.

## Sample UI 

<img width="1919" height="965" alt="Screenshot 2025-07-26 140431" src="https://github.com/user-attachments/assets/68778d40-7725-4340-bde8-f4e1d75b76c9" />

<img width="740" height="648" alt="Screenshot 2025-07-26 140600" src="https://github.com/user-attachments/assets/18ffa0f2-bf73-4ab0-b762-3c8c1983dc5a" />


<img width="781" height="802" alt="Screenshot 2025-07-26 140632" src="https://github.com/user-attachments/assets/a64dfa22-4c51-4c2e-abdb-8f147167c47d" />

## What This Does
The system can take questions in both Bengali and English about literature content and tries to give you accurate answers. It's specifically trained on HSC Bangla 1st Paper material, so it knows about characters like অনুপম, কল্যাণী, and stories from that curriculum.

## Quick Setup

### 1. Install System Dependencies

First, you need these installed on your system :

**Tesseract OCR:**
- Download from: https://github.com/UB-Mannheim/tesseract/wiki
- Install to: `C:\Program Files\Tesseract-OCR\`
- Make sure Bengali language pack is included
- Add to PATH environment variable

**System Requirements**

Hardware needed:

- Windows 10+ (Linux works too)
- 8GB RAM recommended (OCR is heavy)

**Poppler:**
- Download from: https://github.com/oschwartz10612/poppler-windows/releases
- Extract to: `C:\poppler\`
- Add `C:\poppler\Library\bin` to PATH

### 2. Python Setup

```bash
git clone <your-repo-url>
cd bengali-rag-system
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Environment Variables

Create a `.env` file:
```
GROQ_API_KEY = "GROQ_API_KEY " - Go to GROQ API key website for this
HF_TOKEN = "HF_TOKEN"          - Go to HuggingFace website for this one
```

### 4. Run the System

```bash
# Process PDFs and create vector database, which will create vectorstore - db_faiss with .faiss and .pkl files
python memory_creation.py

# Start the chatbot interface
streamlit run chatbot.py
```

## Tools and Libraries Used

- **LangChain**: Main framework for RAG pipeline
- **Tesseract OCR**: Bengali text extraction from PDFs (pain in the neck but works)
- **FAISS**: Vector database for storing embeddings
- **Streamlit**: Web interface
- **FastAPI**: REST API endpoints
- **HuggingFace Transformers**: Multilingual embeddings
- **Groq**: LLM for answer generation
- **PIL**: Image preprocessing for better OCR

## Sample Queries and Expected Outputs

### Bengali Questions:
```
Q: অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?
A: শুম্ভুনাথ

Q: কাকে অনুপমের ভাগ্য দেবতা বলে উল্লেখ করা হয়েছে?
A: মামাকে

Q: বিয়ের সময় কল্যাণীর প্রকৃত বয়স কত ছিল?
A: ১৫ বছর
```

### English Questions:
```
Q: Who is described as the ideal man according to Anupam?
A: শুম্ভুনাথ

Q: What was Kalyani's actual age at marriage?
A: ১৫ বছর
```

## API Documentation

### Start API Server
```bash
# From the Streamlit interface, click "Start API Server"
# Or run manually: uvicorn chatbot:app --host 0.0.0.0 --port 8000
```

### Endpoints

**POST /ask**
```json
{
  "question": "অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?"
}
```

Response:
```json
{
  "question": "অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?",
  "answer": "শুম্ভুনাথ",
  "sources": [
    {"page": 5, "source": "data/hsc_bangla.pdf"}
  ]
}
```

## Technical Deep Dive

### Text Extraction Method

I used **Tesseract OCR with advanced preprocessing** because the PDF couldn't be read with simple text extraction libraries like PyPDF2. The Bengali text was embedded as images in the PDF, not as actual text layers.

**Challenges faced:**
- Bengali characters getting misrecognized (শুম্ভুনাথ became শব্তুনাথ initially)
- Poor image quality in PDF pages
- Mixed Bengali-English text confusing the OCR
- Had to install Poppler and Tesseract manually and mess with environment variables

**Solution:**
- 5x image upscaling before OCR
- Multiple OCR configurations and taking the best result
- Post-processing to fix common OCR errors
- Custom Bengali character validation

### Chunking Strategy

I went with **character-based chunking (500 chars, 100 overlap)** with custom separators for Bengali text. I experimented with multiple chunk sizes - started with 1000 characters, tried 800, 600, and finally settled on 500. But the real issue is the document quality itself.

**Why this works (sort of):**
- Bengali sentences can be quite long, so paragraph-based was too big
- Started with 1000 chars but retrieval was getting too generic
- 500 chars gives more focused chunks, better for specific answers
- Custom separators (।।, ।, \n\n) respect Bengali sentence structure
- Overlap ensures context isn't lost between chunks

**The real problem:** The PDF document quality is terrible. OCR errors mean even perfect chunking won't fix garbled text.

**Separators used:** `["।।", "।", "\n\n", "?", "!", ".", "\n", " "]`

### Embedding Model

**Model:** `intfloat/multilingual-e5-base`

**Why I chose this:**
- Supports both Bengali and English
- Good performance on semantic similarity tasks
- Reasonable size (not too heavy)
- Works well with mixed-language content

**How it captures meaning:**
- Creates dense vector representations of text chunks
- Similar concepts get similar vectors
- Cross-lingual understanding helps with Bengali-English queries

### Similarity Search

**Method:** FAISS with cosine similarity
**Setup:** Vector database with 5 retrieved chunks, fetch_k=12

**Why this approach:**
- FAISS is fast and memory-efficient
- Cosine similarity works well for text embeddings
- Retrieving multiple chunks gives more context
- fetch_k=12 allows for better candidate selection

### Query Processing

**Query enhancement:**
- Expand Bengali question words (কাকে → কাকে কে কার নাম)
- Handle vague queries by adding context
- Normalize Unicode characters

**What happens with vague queries:**
- System still retrieves top-k similar chunks
- LLM is prompted to say "তথ্য পাওয়া যায়নি" if context insufficient
- Conversation memory helps with follow-up questions

### Results and Improvements

**Current performance:** Pretty decent for the test cases, but OCR quality is the main bottleneck.

**What could be better:**
- Better PDF quality would improve OCR accuracy significantly
- Larger document corpus would provide more context
- Fine-tuning the embedding model on Bengali literature
- Better chunking based on semantic boundaries rather than character count
- More sophisticated post-processing for OCR errors

**Main issues I faced:**
- PDF text extraction was impossible without OCR
- Tesseract installation and setup on Windows was frustrating
- Bengali OCR accuracy isn't perfect, especially with old fonts
- Some character recognition errors still slip through

But hey, it works for the basic test cases and can be improved with better source material!

## File Structure

```
.
├── data/                    # Put your PDF files here
├── vectorstore/            # FAISS database gets created here
├── preprocessing/
│   └── core.py            # Text processing utilities
├── memory_creation.py     # PDF processing and vector DB creation
├── chatbot.py            # Streamlit interface and API
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## Contributing

Feel free to fork and improve! The OCR part especially could use some work. If you have better ideas for Bengali text extraction or processing, I'd love to see them.

## License

MIT - do whatever you want with it!
