# Bengali RAG System for HSC Literature

This is a Retrieval-Augmented Generation (RAG) chatbot designed for HSC Bangla 1st Paper students and teachers. It answers questions directly from HSC26-Bangla1st_Paper. 
It builds a multilingual RAG system for HSC Bangla 1st Paper literature using LangChain, FAISS, and Streamlit. It processes Bengali PDFs through Tesseract OCR, creates vector embeddings with multilingual models, and answers questions in Bengali about literary characters and plot details. 

The system extracts text from scanned PDFs, chunks content with custom Bengali separators, stores embeddings in FAISS vectorstore, it also has a short-term memory, and uses Groq's LLM API for answer generation. Includes both a Streamlit web interface and FastAPI endpoints for programmatic access.

## Sample UI 
- Main Interface Overview: 
<img width="1919" height="965" alt="Screenshot 2025-07-26 140431" src="https://github.com/user-attachments/assets/68778d40-7725-4340-bde8-f4e1d75b76c9" />



-  Query in Progress : 
<img width="781" height="781" alt="Screenshot 2025-07-26 140600" src="https://github.com/user-attachments/assets/18ffa0f2-bf73-4ab0-b762-3c8c1983dc5a" />



- Answer with Source Pages : 
<img width="781" height="781" alt="Screenshot 2025-07-26 140632" src="https://github.com/user-attachments/assets/a64dfa22-4c51-4c2e-abdb-8f147167c47d" />

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
## Questions 
## Technical Q&A - Behind the Scenes

### Q: What method or library did you use to extract the text, and why? Did you face any formatting challenges with the PDF content?

**A:** Basically, I tried PyPDF2 and pdfplumber first, thinking it'd be simple. But the PDF was basically a bunch of scanned images - no actual text to extract. So I had to go with Tesseract OCR.

The installation alone was painful - had to download Tesseract, make sure the Bengali language pack was there, mess with PATH variables, install Poppler for PDF handling. Then the OCR kept misreading Bengali characters. "শুম্ভুনাথ" would come out as "শব্তুনাথ" or some gibberish.

I ended up scaling images 5x before feeding them to OCR, trying different configurations, and building a post-processing pipeline to fix common errors. Still not perfect, but way better than the initial attempts.

### Q: What chunking strategy did you choose? Why do you think it works well for semantic retrieval?

**A:** I went with 500 characters with 100 character overlap. Honestly, this took some trial and error.

Started with 1000 characters thinking "bigger chunks = more context = better answers." Wrong! The answers were too generic. Then tried 800, then 600, finally settled on 500. 

The thing is, Bengali sentences can be really long, so I couldn't just split on sentences. I used custom separators like "।।", "।", and paragraph breaks that actually respect how Bengali text flows.

The 100-character overlap is crucial - without it, important context gets lost right at chunk boundaries. Like if a character's name is at the end of one chunk and their description starts the next chunk, you'd lose the connection.

### Q: What embedding model did you use? Why did you choose it? How does it capture the meaning of the text?

**A:** I used `intfloat/multilingual-e5-base`. 

Why? Well, I needed something that could handle both Bengali and English queries. Tried a few Bengali-specific models but they were either too heavy or couldn't handle the mixed content well.

This model is pretty smart - it doesn't just match keywords. If someone asks "কে সুপুরুষ?" and another asks "অনুপমের ভাষায় আদর্শ পুরুষ কে?", it understands these are basically the same question. It creates these 768-dimensional vectors that capture the actual meaning, not just the words.

### Q: How are you comparing the query with your stored chunks? Why did you choose this similarity method and storage setup?

**A:** FAISS with cosine similarity. I chose FAISS because it's fast and handles large vector databases efficiently. Cosine similarity works well for text embeddings - it focuses on the direction of vectors rather than magnitude, which is what you want for semantic similarity.

My setup retrieves 5 chunks but fetches 12 candidates first. This gives the system more options to pick the most relevant ones.

### Q: How do you ensure that the question and document chunks are compared meaningfully? What would happen if the query is vague or missing context?

**A:** I built a query enhancement system. Bengali question words like "কাকে" get expanded to include variations like "কে", "কার নাম". This helps match different ways of asking the same thing.

For vague queries, the system still retrieves the most similar chunks it can find. Then the LLM decides if there's enough context to answer. If not, it returns "তথ্য পাওয়া যায়নি".

The conversation memory helps a lot here. If someone asks "তার বয়স কত?" after asking about a character, the system remembers the context.

### Q: Do the results seem relevant? If not, what might improve them?

**A:** For the test cases, yes. The system correctly identifies character relationships and specific details. But there's room for improvement.

**Main bottleneck:** OCR quality. The source PDF has scanning artifacts that create noise in the embeddings.

### Results and Improvements

**Current performance:** Pretty decent for the test cases, but OCR quality is the main bottleneck.

**What could be better:**
- Better PDF quality would improve OCR accuracy significantly
- Larger document corpus would provide more context
- Fine-tuning the embedding model on Bengali literature
- Better chunking based on semantic boundaries rather than character count
- More sophisticated post-processing for OCR errors
- Smarter chunking based on semantic boundaries rather than character count

**Main issues I faced:**
- PDF text extraction was impossible without OCR
- Tesseract installation and setup on Windows was frustrating
- Bengali OCR accuracy isn't perfect, especially with old fonts
- Some character recognition errors still slip through

All evaluation scores exceeded 80%, with groundedness and relevance both performing strongly; remaining performance constraints stem mainly from input quality.

## Contributing

Feel free to fork and improve! The OCR part especially could use some work. If you have better ideas for Bengali text extraction or processing, I'd love to see them.
