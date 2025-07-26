# 🧠 Memory System in Bengali RAG Chatbot

This chatbot doesn’t just retrieve answers — it remembers the conversation while also staying grounded in the textbook. It uses **two different kinds of memory** for two different jobs:  

---

## 🔹 Short-Term Memory: Conversation Buffer

- **Implementation:** `ConversationBufferMemory` from LangChain  
- **Purpose:** Keeps track of the **chat flow** in the current session.

### ✅ What it does
- Stores the last few interactions between the user and the chatbot.
- Makes the chatbot feel more “human” by allowing **follow-up questions**.

**Example:**
User: কল্যাণীর বয়স কত?
Bot: ১৫ বছর
User: তার বাবার নাম কী?
Bot: শম্ভুনাথ সেন

The bot knows “তার” means কল্যাণী because that context lives in the buffer.

### ✅ Why use a buffer?
- **Lightweight:** No heavy databases or complicated infrastructure.
- **Clean sessions:** When a chat session ends, the memory resets, avoiding old irrelevant data bleeding into new conversations.
- **Perfect for educational chatbots:** Teachers and students ask questions in quick bursts — a buffer memory handles that naturally.

### 🚀 Alternatives for Production Scaling
If the chatbot needs to handle **persistent conversations across sessions**:
- **ConversationBufferWindowMemory** – Keeps *only the last N turns* for lighter context windows.
- **ConversationKGMemory** – Stores facts as a knowledge graph (“কল্যাণী → বয়স → ১৫”) for structured recall.
- **Redis / Postgres-backed Memory** – Makes the chatbot “remember” users across sessions (useful for tutoring systems or long-term study plans).

---

## 🔹 Long-Term Memory: Vector Database

- **Implementation:** FAISS (Facebook AI Similarity Search)
- **Purpose:** Stores and retrieves **textbook knowledge** efficiently.

### ✅ How it works
1. Text from HSC Bangla 1st Paper PDF is processed through **OCR** and cleaned.
2. The cleaned text is broken into **chunks** (500 characters each, with 100-character overlaps).
3. Each chunk is converted into a **768-dimensional embedding** using the multilingual-e5-base model.
4. These embeddings are stored in FAISS — which acts like a **search engine for meaning**.

**When you ask a question:**
- The question is converted into an embedding.
- FAISS finds the top matching chunks instantly.
- The LLM only reads those chunks → **guaranteeing grounded answers.**

### ✅ Why FAISS?
- ⚡ **Speed**: Handles thousands of chunks in milliseconds.
- 💾 **Efficiency**: Stores large volumes of data without blowing up memory.
- 📈 **Scalability**: Can handle multiple textbooks and even multiple subjects later.

### 🚀 Alternatives for Production Scaling
- **Chroma** – Easier to set up for small projects, good LangChain integration.
- **Weaviate / Pinecone** – Cloud-based vector DBs for massive datasets.
- **Postgres + pgvector** – A relational DB solution if your team wants full SQL control.

---

## 🌟 Why This Dual Memory Setup Works
✅ **Short-term memory** = The bot can hold a conversation naturally, understand “তার” and “ওটা.”  
✅ **Long-term memory** = The bot always stays grounded in the HSC Bangla 1st Paper text.
