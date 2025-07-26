import re
import unicodedata
from langchain.schema import Document

class TextProcessor:
    
    @staticmethod
    def clean(text):
        if not text:
            return ""
        
        text = unicodedata.normalize("NFKC", text)
        text = re.sub(r'^\d+\s*$', '', text, flags=re.MULTILINE)
        text = re.sub(r'^পৃষ্ঠা\s*\d+\s*$', '', text, flags=re.MULTILINE)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        
        lines = []
        for line in text.split('\n'):
            line = line.strip()
            if len(line) >= 3:
                bn_chars = len(re.findall(r'[\u0980-\u09FF]', line))
                en_chars = len(re.findall(r'[a-zA-Z]', line))
                if bn_chars >= 2 or en_chars >= 5:
                    lines.append(line)
        
        return '\n'.join(lines).strip()
    
    @staticmethod
    def enhance_query(q):
        q = q.strip()
        
        expansions = {
            'কাকে': 'কাকে কে',
            'কার': 'কার কে', 
            'বয়স': 'বয়স বছর',
            'কোথায়': 'কোথায় স্থান'
        }
        
        for k, v in expansions.items():
            if k in q and len(q.split()) <= 4:
                q = q.replace(k, v)
                break
        
        return q
    
    @staticmethod
    def extract_answer(ans):
        ans = ans.strip()
        ans = re.sub(r'^(উত্তর|Answer)\s*[:\-–—]?\s*', '', ans)
        ans = re.sub(r'^\([ক-ঘ]\)\s*', '', ans)
        
        mcq_pattern = r'\([ক-ঘ]\)\s*([^()\n]+?)(?=\s*\([ক-ঘ]\)|\s*$)'
        for match in re.finditer(mcq_pattern, ans):
            result = match.group(1).strip()
            if len(result) > 1:
                return result
        
        ans_pattern = r'উত্তর\s*[:：]\s*(?:\([ক-ঘ]\)\s*)?([^।\n]+)'
        match = re.search(ans_pattern, ans)
        if match:
            result = match.group(1).strip()
            if len(result) > 1:
                return result
        
        bn_match = re.search(r'[ঀ-৾০-৯][ঀ-৾০-৯\s]*', ans)
        if bn_match:
            result = bn_match.group(0).strip()
            result = re.sub(r'[।,:;।\-\s]+$', '', result)
            if len(result) > 0:
                return result
        
        return ans if ans else "তথ্য পাওয়া যায়নি"

def process_docs(docs):
    processor = TextProcessor()
    cleaned = []
    
    for doc in docs:
        clean_content = processor.clean(doc.page_content)
        if clean_content and len(clean_content) > 50:
            cleaned.append(Document(
                page_content=clean_content,
                metadata=doc.metadata
            ))
    
    return cleaned