import os
import re
import streamlit as st
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_groq import ChatGroq
from preprocessing.core import TextProcessor
from dotenv import load_dotenv
import threading

load_dotenv()

QA_TEMPLATE = """
তুমি বাংলা সাহিত্যের একজন সহায়ক। 
Context এর বাইরে কিছু যোগ কোরো না, আর নিজের ভাষায় নতুন করে লিখবে না। 
Context থেকে সরাসরি উত্তরটি তুলে দাও।  

### নিয়ম:
1. MCQ হলে শুধু সঠিক উত্তরটি দাও (ক, খ, গ, ঘ লিখবে না)।
2. নাম জিজ্ঞেস করলে শুধুমাত্র নামটি বলবে।
3. বয়স/সংখ্যা হলে শুধু সংখ্যা ও একক বলবে (যেমন: ১৫ বছর)।
4. Context এ তথ্য না থাকলে লিখবে: "তথ্য পাওয়া যায়নি"।
5. ব্যবহারকারী শুভেচ্ছা দিলে বিনীতভাবে শুভেচ্ছা জানাবে।

### Context:
{context}

### প্রশ্ন:
{question}

### চূড়ান্ত উত্তর:
"""


CONDENSE_TEMPLATE = """Given the following conversation and a follow up question, rephrase the follow up question to be standalone.
If the follow up question refers to previous context (like "তার", "সেটি", "ওটা"), include that context in the standalone question.

Chat History:
{chat_history}
Follow Up: {question}

Standalone question:"""

DB_PATH = "vectorstore/db_faiss"

class ImprovedTextProcessor:
    @staticmethod
    def enhance_query(query):
        expansions = {
            'কাকে': ['কাকে', 'কে', 'কার নাম'],
            'কার': ['কার', 'কে', 'কার নাম'],
            'কোথায়': ['কোথায়', 'কোন স্থানে', 'স্থান'],
            'কত': ['কত', 'কত বছর', 'বয়স'],
            'কী': ['কী', 'কি', 'কোন'],
            'কেন': ['কেন', 'কেনো', 'কারণ']
        }
        
        enhanced = query
        for key, values in expansions.items():
            if key in query:
                enhanced += ' ' + ' '.join(values)
                break
        
        return enhanced
    
    @staticmethod
    def extract_answer(answer):
        if not answer:
            return "উত্তর পাওয়া যায়নি"
        
        answer = answer.strip()
        
        prefixes = [r'^উত্তর\s*[:：-]?\s*', r'^Answer\s*[:：-]?\s*', r'^\([ক-ঘ]\)\s*']
        for prefix in prefixes:
            answer = re.sub(prefix, '', answer, flags=re.IGNORECASE)
        
        bengali_words = re.findall(r'[\u0980-\u09FF][\u0980-\u09FF\s]*[\u0980-\u09FF]', answer)
        if bengali_words:
            for word in bengali_words:
                word = word.strip()
                if len(word) > 2:
                    return word
        
        number_match = re.search(r'(\d+|[০-৯]+)\s*(বছর|year)?', answer)
        if number_match:
            return number_match.group(0).strip()
        
        sentences = re.split(r'[।!?.\n]', answer)
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 3:
                return sentence
        
        return answer[:50] if answer else "উত্তর পাওয়া যায়নি"

@st.cache_resource
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-base")
    db = FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)
    return db

def setup_chain(vectorstore):
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
        max_token_limit=2000
    )

    qa_prompt = PromptTemplate(template=QA_TEMPLATE, input_variables=["context", "question"])
    condense_prompt = PromptTemplate(template=CONDENSE_TEMPLATE, input_variables=["chat_history", "question"])

    llm = ChatGroq(
        model_name="llama-3.1-8b-instant",
        temperature=0.0,
        groq_api_key=os.environ["GROQ_API_KEY"]
    )
    
    retriever = vectorstore.as_retriever(
        search_type="similarity", 
        search_kwargs={"k": 5, "fetch_k": 12}
    )
    
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        condense_question_prompt=condense_prompt,
        combine_docs_chain_kwargs={"prompt": qa_prompt},
        return_source_documents=True
    )

    return chain

app = FastAPI(title="Bengali RAG API")
global_chain = None
global_vectorstore = None
processor = ImprovedTextProcessor()

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list = []

@app.on_event("startup")
async def startup():
    global global_chain, global_vectorstore
    try:
        global_vectorstore = load_vectorstore()
        global_chain = setup_chain(global_vectorstore)
        print("API initialized successfully")
    except Exception as e:
        print(f"API initialization failed: {e}")

@app.post("/ask", response_model=QueryResponse)
async def ask_question(request: QueryRequest):
    if not global_chain:
        raise HTTPException(status_code=500, detail="System not initialized")
    
    try:
        enhanced_question = processor.enhance_query(request.question)
        response = global_chain.invoke({"question": enhanced_question})
        clean_answer = processor.extract_answer(response["answer"])
        
        sources = [{"page": doc.metadata.get("page", "N/A"), 
                   "source": doc.metadata.get("source", "N/A")} 
                  for doc in response["source_documents"]]
        
        return QueryResponse(
            question=request.question,
            answer=clean_answer,
            sources=sources
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

def run_api():
    uvicorn.run(app, host="0.0.0.0", port=8000)

def main():
    st.set_page_config(page_title="Bengali RAG System", page_icon="📚")
    st.title("Bengali Literature RAG System")
    st.write("HSC Bangla 1st Paper Question Answering")
    
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    if 'chain' not in st.session_state:
        try:
            with st.spinner("Loading system..."):
                vectorstore = load_vectorstore()
                st.session_state.chain = setup_chain(vectorstore)
                st.success("System loaded successfully!")
        except Exception as e:
            st.error(f"Error loading system: {e}")
            st.stop()
    
    for msg in st.session_state.messages:
        with st.chat_message(msg['role']):
            st.markdown(msg['content'])

    if prompt := st.chat_input("আপনার প্রশ্ন লিখুন..."):
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        try:
            with st.chat_message("assistant"):
                with st.spinner("উত্তর খোঁজা হচ্ছে..."):
                    enhanced_prompt = processor.enhance_query(prompt)
                    response = st.session_state.chain.invoke({"question": enhanced_prompt})
                    result = processor.extract_answer(response["answer"])
                    
                    st.markdown(result)
                    
                    if response.get("source_documents"):
                        with st.expander("Sources"):
                            for i, doc in enumerate(response["source_documents"]):
                                st.write(f"Page {doc.metadata.get('page', 'N/A')}")
            
            st.session_state.messages.append({"role": "assistant", "content": result})

        except Exception as e:
            error_msg = "দুঃখিত, উত্তর দিতে পারলাম না।"
            with st.chat_message("assistant"):
                st.markdown(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})

if __name__ == "__main__":
    main()
