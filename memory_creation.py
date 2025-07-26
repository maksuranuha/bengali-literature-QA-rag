import os
import re
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import pytesseract
import pdf2image
import numpy as np
from preprocessing.core import TextProcessor

TESSERACT_CMD = os.getenv('TESSERACT_CMD', r"C:\Program Files\Tesseract-OCR\tesseract.exe")
POPPLER_PATH = os.getenv('POPPLER_PATH', r"C:\poppler\Library\bin")

def check_setup():
    try:
        pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
        version = pytesseract.get_tesseract_version()
        print(f"Tesseract: {version}")
        
        langs = pytesseract.get_languages()
        if 'ben' not in langs:
            print("Bengali language not found")
            return False
            
        return True
    except Exception as e:
        print(f"Setup error: {e}")
        return False

def preprocess_image_advanced(img):
    if img.mode == 'RGBA':
        bg = Image.new('RGB', img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[-1])
        img = bg
    
    if len(np.array(img).shape) == 3:
        img = img.convert('L')
    
    img = img.resize((img.width * 5, img.height * 5), Image.Resampling.LANCZOS)
    img = img.filter(ImageFilter.MedianFilter(size=5))
    img = img.filter(ImageFilter.GaussianBlur(radius=0.5))
    
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(4.0)
    
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(1.2)
    
    enhancer = ImageEnhance.Sharpness(img)
    img = enhancer.enhance(3.0)
    
    img = ImageOps.autocontrast(img)
    img = ImageOps.equalize(img)
    
    return img

def post_process_bengali_text(text):
    if not text:
        return text
    
    corrections = {
        'শব্তুনাথ': 'শুম্ভুনাথ',  
        'শম্ভুনাথ': 'শুম্ভুনাথ',
        'শুম্বুনাথ': 'শুম্ভুনাথ',
        'কাল্যানী': 'কল্যাণী',
        'কল্যানী': 'কল্যাণী',
        'অনুপমের': 'অনুপম',
    }
    
    corrected_text = text
    for wrong, correct in corrections.items():
        corrected_text = corrected_text.replace(wrong, correct)
    
    return corrected_text

def extract_text_enhanced(img):
    configs = [
        {'lang': 'ben+eng', 'config': '--oem 3 --psm 3 -c tessedit_char_whitelist=০১২৩৪৫৬৭৮৯abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ\u0980-\u09FF\u0020\u002E\u002C\u003F\u0021\u003A\u003B।'},
        {'lang': 'ben+eng', 'config': '--oem 3 --psm 6'},
        {'lang': 'ben+eng', 'config': '--oem 1 --psm 3'},
        {'lang': 'ben', 'config': '--oem 3 --psm 3'},
        {'lang': 'ben', 'config': '--oem 3 --psm 6'},
    ]
    
    results = []
    
    for cfg in configs:
        try:
            text = pytesseract.image_to_string(img, lang=cfg['lang'], config=cfg['config'])
            if text:
                text = post_process_bengali_text(text)
                bn_chars = len(re.findall(r'[\u0980-\u09FF]', text))
                total_chars = len(text.strip())
                
                score = bn_chars * 2 + total_chars * 0.1
                
                results.append({
                    'text': text,
                    'score': score,
                    'bn_chars': bn_chars,
                    'config': cfg['config']
                })
                
        except Exception as e:
            continue
    
    if results:
        best_result = max(results, key=lambda x: x['score'])
        return best_result['text']
    
    return ""

def process_pdf(pdf_path, max_pages=None):
    print(f"Processing: {pdf_path}")
    
    try:
        if max_pages:
            pages = pdf2image.convert_from_path(pdf_path, dpi=350, poppler_path=POPPLER_PATH, first_page=1, last_page=max_pages)
        else:
            pages = pdf2image.convert_from_path(pdf_path, dpi=350, poppler_path=POPPLER_PATH)
        
        docs = []
        processor = TextProcessor()
        
        for i, page_img in enumerate(pages, 1):
            print(f"Page {i}/{len(pages)}")
            
            try:
                processed_img = preprocess_image_advanced(page_img)
                raw_text = extract_text_enhanced(processed_img)
                
                if raw_text and len(raw_text.strip()) > 20:
                    cleaned = processor.clean(raw_text)
                    
                    if cleaned and len(cleaned) > 50:
                        doc = Document(
                            page_content=cleaned,
                            metadata={"source": pdf_path, "page": i, "length": len(cleaned)}
                        )
                        docs.append(doc)
                        print(f"✓ Page {i}: {len(cleaned)} chars")
                    else:
                        print(f"✗ Page {i}: Too short")
                else:
                    print(f"✗ Page {i}: OCR failed")
                    
            except Exception as e:
                print(f"✗ Page {i}: {e}")
                continue
        
        print(f"Success: {len(docs)}/{len(pages)} pages")
        return docs
        
    except Exception as e:
        print(f"PDF failed: {e}")
        return []

def create_chunks(docs):
    if not docs:
        return []
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,     
        chunk_overlap=100,
        separators=["।।", "।", "\n\n", "?", "!", ".", "\n", " "],
        length_function=len
    )
    
    chunks = splitter.split_documents(docs)
    filtered = []
    
    for chunk in chunks:
        content = chunk.page_content.strip()
        
        if len(content) < 40:
            continue
            
        bn_chars = len(re.findall(r'[\u0980-\u09FF]', content))
        
        if bn_chars >= 5:  
            filtered.append(chunk)
    
    print(f"Chunks: {len(filtered)} from {len(chunks)}")
    return filtered

def build_vectorstore(docs, save_path="vectorstore/db_faiss"):
    chunks = create_chunks(docs)
    
    if not chunks:
        print("No chunks available")
        return None
    
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="intfloat/multilingual-e5-base",
            model_kwargs={'device': 'cpu'}
        )
        
        vectorstore = FAISS.from_documents(chunks, embeddings)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        vectorstore.save_local(save_path)
        
        print(f"Vectorstore: {vectorstore.index.ntotal} vectors")
        return vectorstore
        
    except Exception as e:
        print(f"Vectorstore failed: {e}")
        return None

def main():
    if not check_setup():
        return
    
    data_folder = "data/"
    if not os.path.exists(data_folder):
        print(f"Folder {data_folder} not found")
        return
    
    pdf_files = [f for f in os.listdir(data_folder) if f.endswith('.pdf')]
    if not pdf_files:
        print("No PDFs found")
        return
    
    all_docs = []
    
    for pdf_file in pdf_files:
        pdf_path = os.path.join(data_folder, pdf_file)
        docs = process_pdf(pdf_path)
        
        if docs:
            all_docs.extend(docs)
            print(f"{pdf_file}: {len(docs)} pages")
        else:
            print(f"{pdf_file}: Failed")
    
    if all_docs:
        print(f"Total: {len(all_docs)} documents")
        vectorstore = build_vectorstore(all_docs)
        
        if vectorstore:
            print("Ready with Enhanced Tesseract OCR!")
        else:
            print("Failed :(")
    else:
        print("No documents processed")

if __name__ == "__main__":
    main()