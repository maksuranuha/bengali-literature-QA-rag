import json
import re
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

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
        
        enhanced = query.strip()
        for key, values in expansions.items():
            if key in query:
                enhanced += ' ' + ' '.join(values[:2])
                break
        
        return enhanced
    
    @staticmethod
    def extract_answer(answer):
        if not answer or not answer.strip():
            return "উত্তর পাওয়া যায়নি"
        
        answer = answer.strip()
        
        prefixes = [
            r'^উত্তর\s*[:：-]?\s*',
            r'^Answer\s*[:：-]?\s*',
            r'^\([ক-ঘ]\)\s*',
            r'^[ক-ঘ]\)\s*',
            r'^[ক-ঘ][\.\)]\s*'
        ]
        
        for prefix in prefixes:
            answer = re.sub(prefix, '', answer, flags=re.IGNORECASE)
        
        answer = answer.strip()
        
        bengali_name_pattern = r'[\u0980-\u09FF]+(?:\s+[\u0980-\u09FF]+)*'
        bengali_matches = re.findall(bengali_name_pattern, answer)
        
        for match in bengali_matches:
            match = match.strip()
            if len(match) > 2 and not re.match(r'^[\u09E6-\u09EF]+$', match):
                return match
        
        number_pattern = r'(\d+|[০-৯]+)\s*(?:বছর|year)?'
        number_match = re.search(number_pattern, answer)
        if number_match:
            return number_match.group(0).strip()
        
        sentences = re.split(r'[।!?.\n]', answer)
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 3:
                return sentence
        
        return answer[:100] if answer else "উত্তর পাওয়া যায়নি"

class RAGEvaluator:
    def __init__(self):
        self.processor = ImprovedTextProcessor()
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name="intfloat/multilingual-e5-base",
                model_kwargs={'device': 'cpu'}
            )
        except Exception as e:
            print(f"Embedding model loading error: {e}")
            raise
        
        self.setup_rag_system()
        
        self.test_cases = [
            {
                "question": "অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?",
                "expected": "শুম্ভুনাথ",
                "keywords": ["শুম্ভুনাথ", "সুপুরুষ"]
            },
            {
                "question": "কাকে অনুপমের ভাগ্য দেবতা বলে উল্লেখ করা হয়েছে?",
                "expected": "মামাকে",
                "keywords": ["মামা", "ভাগ্য দেবতা"]
            },
            {
                "question": "বিয়ের সময় কল্যাণীর প্রকৃত বয়স কত ছিল?",
                "expected": "১৫ বছর",
                "keywords": ["১৫", "বয়স", "কল্যাণী"]
            },
            {
                "question": "অনুপমের বন্ধুর নাম কী?",
                "expected": "শুম্ভুনাথ",
                "keywords": ["শুম্ভুনাথ", "বন্ধু"]
            },
            {
                "question": "কল্যাণীর বাবার নাম কী?",
                "expected": "হরিশচন্দ্র",
                "keywords": ["হরিশচন্দ্র", "বাবা"]
            }
        ]
    
    def setup_rag_system(self):
        QA_TEMPLATE = """তুমি বাংলা সাহিত্যের একজন বিশেষজ্ঞ। প্রদত্ত Context থেকে প্রশ্নের উত্তর দাও।

নিয়ম:
- শুধুমাত্র Context এ থাকা তথ্য ব্যবহার করো
- MCQ প্রশ্নে শুধু সঠিক উত্তরটি লিখো
- নাম জিজ্ঞেস করলে শুধু নামটি লিখো
- বয়স/সংখ্যা জিজ্ঞেস করলে শুধু সংখ্যাটি লিখো
- সংক্ষিপ্ত ও সরাসরি উত্তর দাও

Context:
{context}

প্রশ্ন: {question}

উত্তর:"""

        try:
            if not os.path.exists("vectorstore/db_faiss"):
                raise FileNotFoundError("Vector database not found at vectorstore/db_faiss")
            
            vectorstore = FAISS.load_local(
                "vectorstore/db_faiss", 
                self.embeddings, 
                allow_dangerous_deserialization=True
            )
        except Exception as e:
            print(f"Error loading vectorstore: {e}")
            raise
        
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        
        if not os.environ.get("GROQ_API_KEY"):
            raise ValueError("GROQ_API_KEY not found in environment variables")
        
        llm = ChatGroq(
            model_name="llama-3.1-8b-instant",
            temperature=0.1,
            groq_api_key=os.environ["GROQ_API_KEY"],
            max_tokens=512
        )
        
        qa_prompt = PromptTemplate(
            template=QA_TEMPLATE, 
            input_variables=["context", "question"]
        )
        
        retriever = vectorstore.as_retriever(
            search_type="similarity", 
            search_kwargs={"k": 3, "fetch_k": 8}
        )
        
        self.chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory,
            combine_docs_chain_kwargs={"prompt": qa_prompt},
            return_source_documents=True,
            verbose=False
        )
    
    def semantic_similarity_score(self, predicted, expected):
        if not predicted or not expected:
            return 0.0
        
        try:
            pred_embedding = self.embeddings.embed_query(str(predicted))
            exp_embedding = self.embeddings.embed_query(str(expected))
            
            pred_array = np.array(pred_embedding).reshape(1, -1)
            exp_array = np.array(exp_embedding).reshape(1, -1)
            
            similarity = cosine_similarity(pred_array, exp_array)[0][0]
            return max(0.0, min(1.0, similarity))
        
        except Exception as e:
            print(f"Semantic similarity error: {e}")
            return 0.0
    
    def relevance_score(self, question, retrieved_docs):
        if not retrieved_docs or not question:
            return 0.0
        
        try:
            question_embedding = self.embeddings.embed_query(question)
            question_array = np.array(question_embedding).reshape(1, -1)
            
            relevance_scores = []
            for doc in retrieved_docs[:3]:
                doc_text = doc.page_content[:800] if hasattr(doc, 'page_content') else str(doc)[:800]
                if not doc_text.strip():
                    continue
                
                doc_embedding = self.embeddings.embed_query(doc_text)
                doc_array = np.array(doc_embedding).reshape(1, -1)
                
                similarity = cosine_similarity(question_array, doc_array)[0][0]
                relevance_scores.append(similarity)
            
            return np.mean(relevance_scores) if relevance_scores else 0.0
        
        except Exception as e:
            print(f"Relevance score error: {e}")
            return 0.0
    
    def groundedness_score(self, answer, context_docs):
        if not context_docs or not answer:
            return 0.0
        
        try:
            combined_context = ""
            for doc in context_docs[:3]:
                if hasattr(doc, 'page_content'):
                    combined_context += doc.page_content + " "
                else:
                    combined_context += str(doc) + " "
            
            combined_context = combined_context.strip()[:1500]
            
            if not combined_context:
                return 0.0
            
            answer_embedding = self.embeddings.embed_query(str(answer))
            context_embedding = self.embeddings.embed_query(combined_context)
            
            answer_array = np.array(answer_embedding).reshape(1, -1)
            context_array = np.array(context_embedding).reshape(1, -1)
            
            similarity = cosine_similarity(answer_array, context_array)[0][0]
            return max(0.0, min(1.0, similarity))
        
        except Exception as e:
            print(f"Groundedness score error: {e}")
            return 0.0
    
    def evaluate_single(self, test_case):
        question = test_case["question"]
        expected = test_case["expected"]
        
        try:
            enhanced_question = self.processor.enhance_query(question)
            response = self.chain.invoke({"question": enhanced_question})
            
            predicted = self.processor.extract_answer(response.get("answer", ""))
            retrieved_docs = response.get("source_documents", [])
            
            semantic_sim = self.semantic_similarity_score(predicted, expected)
            relevance = self.relevance_score(question, retrieved_docs)
            groundedness = self.groundedness_score(predicted, retrieved_docs)
            
            return {
                "question": question,
                "expected": expected,
                "predicted": predicted,
                "scores": {
                    "semantic_similarity": semantic_sim,
                    "relevance": relevance,
                    "groundedness": groundedness
                },
                "sources_count": len(retrieved_docs)
            }
        
        except Exception as e:
            print(f"Error evaluating question '{question}': {e}")
            return {
                "question": question,
                "expected": expected,
                "predicted": "Error occurred",
                "scores": {
                    "semantic_similarity": 0.0,
                    "relevance": 0.0,
                    "groundedness": 0.0
                },
                "sources_count": 0
            }
    
    def evaluate_all(self):
        results = []
        
        print("Starting RAG Evaluation...")
        print("=" * 60)
        
        for i, test_case in enumerate(self.test_cases, 1):
            print(f"\nTest Case {i}:")
            print(f"Question: {test_case['question']}")
            
            result = self.evaluate_single(test_case)
            results.append(result)
            
            print(f"Expected: {result['expected']}")
            print(f"Predicted: {result['predicted']}")
            print(f"Semantic Similarity: {result['scores']['semantic_similarity']:.3f}")
            print(f"Relevance: {result['scores']['relevance']:.3f}")
            print(f"Groundedness: {result['scores']['groundedness']:.3f}")
            print(f"Sources Retrieved: {result['sources_count']}")
        
        all_scores = {"semantic_similarity": [], "relevance": [], "groundedness": []}
        
        for result in results:
            for metric, score in result["scores"].items():
                all_scores[metric].append(score)
        
        avg_scores = {metric: np.mean(scores) for metric, scores in all_scores.items()}
        
        print("\n" + "=" * 60)
        print("OVERALL EVALUATION RESULTS:")
        print("=" * 60)
        
        print(f"Semantic Similarity: {avg_scores['semantic_similarity']:.3f}")
        print(f"Relevance: {avg_scores['relevance']:.3f}")
        print(f"Groundedness: {avg_scores['groundedness']:.3f}")
        
        overall_score = (
            avg_scores["semantic_similarity"] * 0.4 +
            avg_scores["relevance"] * 0.3 +
            avg_scores["groundedness"] * 0.3
        )
        
        print(f"\nOverall RAG Score: {overall_score:.3f}")
        
        if overall_score >= 0.8:
            performance = "Excellent"
        elif overall_score >= 0.6:
            performance = "Good"
        elif overall_score >= 0.4:
            performance = "Fair"
        else:
            performance = "Needs Improvement"
        
        print(f"Performance Level: {performance}")
        
        evaluation_report = {
            "overall_scores": avg_scores,
            "overall_rag_score": overall_score,
            "performance_level": performance,
            "detailed_results": results,
            "test_cases_count": len(results)
        }
        
        try:
            with open("rag_evaluation_report.json", "w", encoding="utf-8") as f:
                json.dump(evaluation_report, f, ensure_ascii=False, indent=2)
            print(f"\nReport saved to: rag_evaluation_report.json")
        except Exception as e:
            print(f"Error saving report: {e}")
        
        return evaluation_report

if __name__ == "__main__":
    try:
        evaluator = RAGEvaluator()
        evaluator.evaluate_all()
    except Exception as e:
        print(f"Evaluation failed: {e}")