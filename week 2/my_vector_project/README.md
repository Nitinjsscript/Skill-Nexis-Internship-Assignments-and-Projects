# 🗂️ Local Document Vector DB & Intelligent Agent Workspace

A production-grade, privacy-first **Local Knowledge Base Search Engine and Autonomous AI Agent** built using LangChain, ChromaDB, and Ollama. This project runs **100% locally** on your hardware, requiring zero API keys, zero subscription costs, and ensuring complete data protection.

---

## 🏗️ Project Architecture Layout

```text
my_vector_project/
├── data/                 📁 Drop your 50+ source files here (PDFs, Markdown, etc.)
├── chroma_db/            📁 Persistent local vector database directory (built by ingest.py)
├── requirements.txt      📄 Python dependency index
├── ingest.py             🐍 Multi-format parsing & OCR fallback ingestion pipeline
├── app.py                💻 Streamlit Web search interface targeting local vector indices
├── rag_pipeline.py       🐍 Standalone terminal-based single-turn RAG execution script
├── rag_with_memory.py    🐍 Standalone terminal-based multi-turn conversational RAG script
└── agent_explorer.py     💻 Streamlit UI Autonomous AI Agent featuring tool routing
```

---

## 🛠️ Step 1: Environment & Tool Prerequisites

### 1. Install System-Level Binary Drivers
For the advanced multi-scenario PDF engine to process image-scanned documents, your underlying OS needs the following dependencies:
* **macOS:** `brew install tesseract poppler`
* **Linux:** `sudo apt install tesseract-ocr poppler-utils`
* **Windows:** Download the binary installers for **Tesseract OCR** and **Poppler**, and append their executable paths to your Windows System environment `Path` variable.

### 2. Install Project Dependencies
Run the installation command in your terminal panel:
```bash
pip install -r requirements.txt
```

### 3. Spin Up Your Local AI Engine (Ollama)
1. Download and finish the wizard installation from [ollama.com](https://ollama.com).
2. Open a clean terminal panel and download the reasoning model:
   ```bash
   ollama run llama3.2
   ```
   *(Once verified, type `/exit` to return to your normal command prompt).*

### 4. Build Your Vector Index Data Store
Drop all your `.pdf`, `.md`, and `.txt` files into the `./data` directory, then execute the assembly ingestion pipeline once:
```bash
python ingest.py
```

---

## 🚀 Step 2: System Execution Commands & Test Scenarios

### Module 1: Semantic Document Base Viewer (`app.py`)
Runs an interactive, local Streamlit web search dashboard to explore indexed vector strings directly.

```bash
streamlit run app.py
```
#### ❓ Targeted Verification Test Queries:
* `What is HuSpaCy?`
* `Explain the structural metadata requirements inside a YAML frontmatter block for a skill.`

---

### Module 2: Standalone Local RAG Pipeline (`rag_pipeline.py`)
Hooks your disk-backed vectors directly into your local Llama 3.2 engine to generate structured, single-turn answers directly in your terminal console.

```bash
python rag_pipeline.py
```
#### ❓ Targeted Verification Test Queries:
* `What advancements does HuSpaCy introduce to Hungarian text processing frameworks?`
* `Detail the specific metrics or advantages of using Dice Loss for data-imbalanced NLP tasks.`

---

### Module 3: Conversational Memory Assistant Loop (`rag_with_memory.py`)
Executes a **pure terminal-based interactive chat interface** implementing dynamic **Window Memory** to track contextual pronouns across multi-turn sessions.

```bash
python rag_with_memory.py
```
#### ❓ Sequential Multi-Turn Context Questions:
1. **Turn 1 (Establish Entity):** `Explain the core concept behind Dice Loss and why it is useful for data-imbalanced NLP tasks.`
2. **Turn 2 (Follow up using a Pronoun):** `How does its mathematical formulation compare directly to traditional cross-entropy loss metrics?`
• Watch for: The assistant will read the phrase "its mathematical formulation" and automatically connect it back to "Dice Loss" from your first turn using your active short-term Window memory state arrays.


---

### Module 4: Autonomous Reasoning AI Agent Workspace (`agent_explorer.py`)
Launches an advanced, browser-driven Streamlit chat environment where your AI dynamically routes user questions to distinct sub-tools (Web, Math, Docs).

```bash
streamlit run agent_explorer.py
```
#### ❓ Cross-Domain Tool Verification Questions:
* **Targeting Document Retrieval Tool:** `What are the core technical syntax parameters required inside a skill file configuration?`
* **Targeting Calculator Tool:** `What is the exact mathematical evaluation result of 54932 * (105 / 5)?`
* **Targeting Live Web Search Tool:** `Who is the current Prime Minister of the United Kingdom right now?`

---

## 🛡️ Key Project Features
* **Zero Cloud Data Leakage:** All computation, indexing, and generative modeling runs completely offline.
* **Smart Chunking:** Structural text division that prevents breaking code structures, frontmatter headers, or tables.
* **Token Streaming:** Handled cleanly to show live text rendering word-by-word without leaky debug array prints.
