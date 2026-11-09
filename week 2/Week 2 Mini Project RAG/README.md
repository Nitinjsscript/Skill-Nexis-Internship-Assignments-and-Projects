# 🏛️ GovInfo Historical Data Analysis Workspace

A production-grade, privacy-first **Local Knowledge Base Conversational Search Engine (RAG)** built using LangChain, ChromaDB, and Ollama. This system runs **100% locally** on your hardware, requiring zero external API keys, zero subscription costs, and ensuring complete data protection.

This specific project is optimized to ingest, structuralize, and analyze **historical US Government data from GovInfo**, specifically the *Annual Reports of the Chief of the Children's Bureau to the Secretary of Labor* spanning from **1916 to 1932**. It features explicit string validation layers to prevent raw vector matrix data leakage and coordinates document retrieval with an adaptive conversational memory system.

---

## 🏗️ Project Architecture Layout

```text
Week 2 Mini Project RAG/
├── data/                 📁 Drop your GovInfo PDF files here (e.g., report_1924.pdf)
├── chroma_db/            📁 Persistent local vector database directory (built by ingestion)
├── requirements.txt      📄 Python dependency index 
├── data ingestion.py     🐍 Multi-format parsing & metadata scraping ingestion pipeline
└── app.py                💻 Streamlit Web UI implementing Buffer, Window, & Summary memory
```

---

## 🛠️ Step 1: System-Level & Environment Prerequisites

### 1. Install System Binary Drivers
Because historical archive logs can contain un-digitized pages or faint typewriter scanned layers, your underlying operating system needs the following background processing utilities for OCR image conversion:

*   **Windows installation:**
    1.  Download and run the **Tesseract OCR** installer. Add `C:\Program Files\Tesseract-OCR` to your Windows System Environment **`Path`**.
    2.  Download **Poppler for Windows**, extract the folder, and append its `bin` directory path (e.g., `C:\poppler\bin`) to your Windows environment **`Path`**.
*   **macOS installation:** `brew install tesseract poppler`
*   **Linux installation:** `sudo apt install tesseract-ocr poppler-utils`

### 2. Install Project Dependencies
Open your project terminal panel and execute the package index collection script:
```bash
pip install -r requirements.txt
```

### 3. Spin Up Your Local AI Engine (Ollama)
1. Download and run the setup file from the official source: [ollama.com](https://ollama.com).
2. Open a fresh terminal panel and initialize the lightweight reasoning model:
   ```bash
   ollama run llama3.2
   ```
   *(Once the setup finishes downloading, type `/exit` to return to your normal command prompt).*

---

## 🚀 Step 2: System Execution Workflow & Commands

### 1. Build Your Local Vector Index
Place your downloaded GovInfo reports inside the `./data` folder. **Ensure every PDF filename contains its 4-digit fiscal year** (e.g., `Childrens_Bureau_Report_1928.pdf`) so the regex engine can append it to the metadata properties. Run the ingestion pipeline script:

```bash
python "data ingestion.py"
```
*Wait until the script prints out:* `✅ Success! Local knowledge base vectors compiled safely onto your hard disk drive.`

### 2. Launch the Conversational Memory Web Hub
Executes an integrated **Streamlit web dashboard** featuring advanced **Buffer, Window, and Summary history routing matrix profiles**. It locks the vector store to a custom, local ONNX embedding wrapper class to guarantee zero raw floating-point text array leakage.

```bash
streamlit run app.py
```

#### ❓ Sequential Multi-Turn Context Questions to Test Memory:
*   **Turn 1 (Establish Entity Data):** `"What were the primary core child labor regulations or allocations discussed inside the Fourteenth Annual Report?"`
*   **Turn 2 (Context Follow-Up relying entirely on Window Memory):** `"Show me a summary list of the specific funding amounts or personnel counts allocated to support it."` *(The system will automatically resolve that "it" refers back to the regulations in Turn 1).*

---

## 🛡️ Key System Capabilities & Engineering Standards
*   **Zero Cloud Data Leakage:** Fully functional offline. Your scanned documents and prompts are processed strictly inside your machine's processor and RAM layouts.
*   **Metadata Lineage Tracing:** The app automatically traces and parses the source code properties, printing a clean **"Verified GovInfo Sources Utilized for Context Verification"** citation footer beneath every generated answer block.
*   **Token Streaming:** Implements a token word-by-word streaming wrapper (`llm.stream()`) matching production-grade standards.
