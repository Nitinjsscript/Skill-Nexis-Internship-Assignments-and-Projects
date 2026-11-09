import os
import re
import chromadb
import pdfplumber
from pdf2image import convert_from_path
import pytesseract
from langchain_text_splitters import RecursiveCharacterTextSplitter

def extract_pdf_with_ocr_fallback(file_path):
    """Extracts digital text layers or falls back to page-by-page OCR for historic scans."""
    extracted_pages = []
    
    with pdfplumber.open(file_path) as pdf:
        for page_idx, page in enumerate(pdf.pages):
            text = page.extract_text()
            
            # Historical scan check: empty layer or faint typography artifact boundary
            if not text or len(text.strip()) < 20:
                try:
                    # Convert only the single page to an image to protect CPU memory limits
                    images = convert_from_path(file_path, first_page=page_idx+1, last_page=page_idx+1)
                    if images:
                        ocr_text = pytesseract.image_to_string(images[0])
                        extracted_pages.append((page_idx + 1, ocr_text))
                except Exception as ocr_err:
                    print(f"⚠️ OCR engine failed on {os.path.basename(file_path)} Page {page_idx+1}: {ocr_err}")
                    extracted_pages.append((page_idx + 1, ""))
            else:
                extracted_pages.append((page_idx + 1, text))
                
    return extracted_pages

def run_historical_ingestion():
    DATA_FOLDER = "./data"
    DB_FOLDER = "./chroma_db"
    
    if not os.path.exists(DATA_FOLDER):
        os.makedirs(DATA_FOLDER)
        print(f"📁 Core data directory initialized. Drop GovInfo files into '{DATA_FOLDER}' and rerun.")
        return

    print("🤖 Initializing persistent local vector data storage disk matrix...")
    client = chromadb.PersistentClient(path=DB_FOLDER)
    collection = client.get_or_create_collection(
        name="knowledge_base", 
        metadata={"hnsw:space": "cosine"}
    )
    
    # Custom character boundaries to keep historical data tables grouped safely
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=700, 
        chunk_overlap=70,
        separators=["\n\n", "\n", " ", ""]
    )
    
    documents, metadatas, ids = [], [], []
    chunk_index = 0
    
    supported_files = [f for f in os.listdir(DATA_FOLDER) if f.endswith(('.pdf', '.txt', '.md'))]
    if not supported_files:
        print(f"⚠️ Empty vector target source files parameters inside '{DATA_FOLDER}'. Place GovInfo documents here.")
        return

    print(f"📚 Found {len(supported_files)} items. Starting ingestion mapping sequence...")
    
    for filename in supported_files:
        file_path = os.path.join(DATA_FOLDER, filename)
        print(f"🔍 Compiling: {filename}")
        
        # Regex metadata filter: Extract the 4-digit fiscal year from GovInfo filename structures
        year_match = re.search(r'(19\d{2})', filename)
        fiscal_year = int(year_match.group(1)) if year_match else 0
        
        try:
            if filename.endswith('.pdf'):
                pages_data = extract_pdf_with_ocr_fallback(file_path)
                for page_num, page_text in pages_data:
                    if not page_text.strip():
                        continue
                    chunks = text_splitter.split_text(page_text)
                    for chunk in chunks:
                        documents.append(chunk)
                        metadatas.append({
                            "source": filename, 
                            "page": page_num, 
                            "file_type": "pdf", 
                            "fiscal_year": fiscal_year
                        })
                        ids.append(f"id_{filename.replace('.', '_')}_p{page_num}_ch{chunk_index}")
                        chunk_index += 1
                        
            elif filename.endswith(('.txt', '.md')):
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    chunks = text_splitter.split_text(f.read())
                    for chunk in chunks:
                        documents.append(chunk)
                        metadatas.append({
                            "source": filename, 
                            "page": 1, 
                            "file_type": "text", 
                            "fiscal_year": fiscal_year
                        })
                        ids.append(f"id_{filename.replace('.', '_')}_ch{chunk_index}")
                        chunk_index += 1
                        
        except Exception as err:
            print(f"❌ Critical error parsing file asset {filename}: {err}")

    # Safe Batch database upload execution engine
    if documents:
        print(f"🚀 Batching {len(documents)} structural text vectors into persistent storage space...")
        BATCH_SIZE = 100
        for i in range(0, len(documents), BATCH_SIZE):
            collection.add(
                documents=documents[i:i+BATCH_SIZE],
                metadatas=metadatas[i:i+BATCH_SIZE],
                ids=ids[i:i+BATCH_SIZE]
            )
        print("✅ Success! Local knowledge base vectors compiled safely onto your hard disk drive.")
    else:
        print("⚠️ Ingestion completed, but zero indexable text parameters were found.")

if __name__ == "__main__":
    run_historical_ingestion()
