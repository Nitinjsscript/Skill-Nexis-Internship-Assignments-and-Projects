import os
import yaml
import chromadb
import pdfplumber
from pdf2image import convert_from_path
import pytesseract
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter

def extract_pdf_with_ocr_fallback(file_path):
    """Parses digital text layout layers or automatically falls back to page-by-page OCR."""
    extracted_pages = []
    
    with pdfplumber.open(file_path) as pdf:
        for page_idx, page in enumerate(pdf.pages):
            text = page.extract_text()
            
            # SCENARIO: Empty layer implies scanned file, image-only document, or broken encoding
            if not text or len(text.strip()) < 15:
                try:
                    # Render single page as image to conserve host RAM allocation memory space
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

def parse_markdown_with_frontmatter(file_path):
    """Extracts raw content and separates YAML frontmatter blocks to keep metadata clean."""
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
        
    frontmatter = {}
    markdown_body = content
    
    # Check if file begins with a structural YAML boundary block
    if content.startswith('---'):
        parts = content.split('---', 2)
        if len(parts) >= 3:
            try:
                frontmatter = yaml.safe_load(parts[1]) or {}
                markdown_body = parts[2]
            except Exception as yaml_err:
                print(f"⚠️ YAML parsing error in frontmatter for {os.path.basename(file_path)}: {yaml_err}")
                
    return markdown_body, frontmatter

def process_markdown_chunks(file_path, general_text_splitter):
    """Splits markdown elements by keeping architectural headers and code blocks bound together."""
    body, metadata = parse_markdown_with_frontmatter(file_path)
    
    # Establish structural text breaking vectors based on markdown syntax
    headers_to_split_on = [
        ("#", "Header_1"),
        ("##", "Header_2"),
        ("###", "Header_3"),
    ]
    md_header_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    header_splits = md_header_splitter.split_text(body)
    
    final_chunks = []
    for split in header_splits:
        # Prevent markdown configuration lines or multi-line tables from snapping mid-sentence
        sub_chunks = general_text_splitter.split_text(split.page_content)
        for sub_chunk in sub_chunks:
            chunk_meta = split.metadata.copy()
            
            # Flatten dictionary metadata properties to guarantee storage validation compliance
            for k, v in metadata.items():
                if isinstance(v, (list, dict)):
                    chunk_meta[f"frontmatter_{k}"] = str(v)
                else:
                    chunk_meta[f"frontmatter_{k}"] = v
                    
            final_chunks.append({"text": sub_chunk, "metadata": chunk_meta})
            
    return final_chunks

def run_production_ingestion():
    DATA_FOLDER = "./data"
    DB_FOLDER = "./chroma_db"
    
    if not os.path.exists(DATA_FOLDER):
        os.makedirs(DATA_FOLDER)
        print(f"📁 Root data source folder initialized. Drop your files inside '{DATA_FOLDER}' and rerun.")
        return

    print("🤖 Instantiating local vector store client execution loops...")
    client = chromadb.PersistentClient(path=DB_FOLDER)
    collection = client.get_or_create_collection(
        name="knowledge_base", 
        metadata={"hnsw:space": "cosine"}
    )
    
    # Custom structural character boundary splitter patterns
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600, 
        chunk_overlap=60,
        separators=["\n\n", "\n", " ", ""]
    )
    
    documents, metadatas, ids = [], [], []
    chunk_index = 0
    
    supported_files = [f for f in os.listdir(DATA_FOLDER) if f.endswith(('.pdf', '.txt', '.md'))]
    if not supported_files:
        print(f"⚠️ Empty source vector target target parameters inside '{DATA_FOLDER}'. Drop your PDFs and Markdown files here.")
        return

    print(f"📚 Found {len(supported_files)} configuration items. Initiating ingestion matrix calculations...")
    
    for filename in supported_files:
        file_path = os.path.join(DATA_FOLDER, filename)
        print(f"🔎 Ingesting file properties: {filename}")
        
        try:
            if filename.endswith('.pdf'):
                pages_data = extract_pdf_with_ocr_fallback(file_path)
                for page_num, page_text in pages_data:
                    if not page_text.strip():
                        continue
                    chunks = text_splitter.split_text(page_text)
                    for chunk in chunks:
                        documents.append(chunk)
                        metadatas.append({"source": filename, "page": page_num, "file_type": "pdf"})
                        ids.append(f"id_{filename.replace('.', '_')}_p{page_num}_ch{chunk_index}")
                        chunk_index += 1
                        
            elif filename.endswith('.md'):
                md_chunks = process_markdown_chunks(file_path, text_splitter)
                for chunk_item in md_chunks:
                    documents.append(chunk_item["text"])
                    meta = chunk_item["metadata"]
                    meta.update({"source": filename, "page": 1, "file_type": "markdown"})
                    metadatas.append(meta)
                    ids.append(f"id_{filename.replace('.', '_')}_ch{chunk_index}")
                    chunk_index += 1
                    
            elif filename.endswith('.txt'):
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    chunks = text_splitter.split_text(f.read())
                    for chunk in chunks:
                        documents.append(chunk)
                        metadatas.append({"source": filename, "page": 1, "file_type": "text"})
                        ids.append(f"id_{filename.replace('.', '_')}_ch{chunk_index}")
                        chunk_index += 1
                        
        except Exception as err:
            print(f"❌ Failed processing sequence on file item {filename}. Exception: {err}")

    # Step 5: Safe Batch commit chunk routines
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
        print("⚠️ Execution loop completed, but zero indexable text parameters were found.")

if __name__ == "__main__":
    run_production_ingestion()
