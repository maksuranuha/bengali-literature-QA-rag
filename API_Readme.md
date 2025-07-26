This API provides a Bengali literature question answering system specifically designed for HSC Bangla 1st Paper content. It uses Retrieval-Augmented Generation (RAG) combining vector search and a powerful LLM to deliver precise, context-aware answers with source citations.

---

## **Key Features**  
- **Bengali and English question support** with enhanced query processing  
- **Semantic retrieval** using FAISS and multilingual embeddings  
- **Context-aware answers** leveraging conversation memory for follow-ups  
- **Answer formatting rules** for MCQs, names, and numeric answers  
- **Source attribution** included with each response  
- Production-ready with FastAPI backend for scalable RESTful access

---

## **Environment Setup**

### **Required Environment Variables**  
- `GROQ_API_KEY` — API key to access the Groq LLM model  
- `HF_TOKEN` (optional) — HuggingFace token for embedding models  

### **Dependencies**  
- Python 3.8+  
- FastAPI, Uvicorn, LangChain, FAISS, HuggingFace Transformers

---

## **Running the API**

Run the server using:

```bash
uvicorn chatbot:app --host 0.0.0.0 --port 8000
## **API Endpoint**

### **POST** `/ask`  
Submit a JSON request containing the user question.

---

### **Request format:**

```json
{
  "question": "অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?"
}
```

## How It Works
### Query Enhancement
The system expands Bengali question keywords (e.g., "কাকে" → "কে", "কার নাম") to improve semantic retrieval coverage.

### Vector Search
The question and text chunks are embedded into a shared vector space using intfloat/multilingual-e5-base. FAISS retrieves the top relevant chunks by cosine similarity.

### Conversational Memory
Short-term memory buffers previous interactions for context-aware follow-up question understanding.

### Answer Generation
The Groq LLM generates answers using strict prompt templates enforcing:

- MCQ answers as the exact choice (no options)
- Names as simple text
- Numeric answers with units
- If no answer is found, it replies "তথ্য পাওয়া যায়নি".

### Source Attribution
Every response includes metadata of the source documents (page number, filename) for user verification.

