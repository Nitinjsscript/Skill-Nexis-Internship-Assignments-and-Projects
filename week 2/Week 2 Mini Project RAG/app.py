import os
import streamlit as st
import chromadb
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# 1. Framework Workspace Web Canvas Configuration
st.set_page_config(page_title="GovInfo Historical Analysis Hub", layout="wide")
st.title("🏛️ GovInfo Historical Data Analysis Workspace")
st.markdown("Chat interactively with historical Children's Bureau Annual Reports (1916–1932) using local vectors and adaptive memory.")

# 2. ⚙️ Sidebar System Controllers
st.sidebar.header("⚙️ Engine Architecture Controllers")
MEMORY_MODE = st.sidebar.selectbox("Active Memory Profile", ["window", "buffer", "summary"])
WINDOW_SIZE = st.sidebar.slider("Window Boundary Slicing (Messages)", min_value=2, max_value=10, value=4, step=2)
RETRIEVAL_K = st.sidebar.slider("Historical Document Chunks (k)", min_value=1, max_value=10, value=4)

# 3. 🔋 Optimize Resource Connections via Cache Wrappers
@st.cache_resource
def connect_historical_vector_store(k_chunks):
    DB_FOLDER = "./chroma_db"
    if not os.path.exists(DB_FOLDER):
        return None
        
    # Connect directly to the persistent client on your disk drive
    persistent_client = chromadb.PersistentClient(path=DB_FOLDER)

    print("🧠 Forcing vector coordinate model binding...")
    # This directly targets the local all-MiniLM-L6-v2 ONNX model that data ingestion.py compiled
    from chromadb.utils.embedding_functions.onnx_mini_lm_l6_v2 import ONNXMiniLM_L6_V2
    onnx_embedding_function = ONNXMiniLM_L6_V2()
    
    # Custom wrapper class to make Chroma's native embedding function look exactly like a LangChain embedding class
    class LocalChromadbEmbeddings:
        def embed_documents(self, texts):
            return onnx_embedding_function(texts)
        def embed_query(self, text):
            return onnx_embedding_function([text])[0]

    # Ground the LangChain wrapper with the custom embedding object class to stop array data leakage
    vectorstore = Chroma(
        client=persistent_client,
        collection_name="knowledge_base",
        embedding_function=LocalChromadbEmbeddings()
    )
    return vectorstore.as_retriever(search_kwargs={"k": k_chunks})

@st.cache_resource
def initialize_local_llm():
    return OllamaLLM(model="llama3.2", base_url="http://localhost:11434", temperature=0.1)

retriever = connect_historical_vector_store(RETRIEVAL_K)
llm = initialize_local_llm()

# 4. Initialize Persistent Browser State Maps to Protect Chat History
if "ui_chat_history" not in st.session_state:
    st.session_state.ui_chat_history = []  # For painting the front-end chat box layout bubbles
if "model_chat_history" not in st.session_state:
    st.session_state.model_chat_history = []  # LangChain Message state tracking objects

if retriever is None:
    st.error("🚨 Local vector storage target folder `./chroma_db` not discovered. Execute your ingestion pipeline script completely first.")
else:
    # 5. 📝 Conversational Memory Strategy State Router
    def run_memory_routing(chat_history):
        if MEMORY_MODE == "buffer" or len(chat_history) < 2:
            return chat_history
            
        elif MEMORY_MODE == "window":
            return chat_history[-WINDOW_SIZE:]
            
        elif MEMORY_MODE == "summary":
            with st.spinner("📝 Summarizing historical message context..."):
                old_messages = chat_history[:-2]
                formatted_history = "\n".join([f"{'User' if isinstance(m, HumanMessage) else 'AI'}: {m.content}" for m in old_messages])
                
                summary_prompt = f"""Summarize the following historical conversation between a researcher and an AI assistant into a single concise paragraph. 
Focus strictly on the core legislative bills, fiscal budget values, and dates discussed. Do not lose historical specific acronyms.
                
Conversation History:
{formatted_history}
                
Concise Summary Paragraph:"""
                running_summary = llm.invoke(summary_prompt)
                return [SystemMessage(content=f"Summary of previous conversation turns: {running_summary}")] + chat_history[-2:]
                
        return chat_history

    # 6. 📥 Redraw previous conversation elements upon user page scroll refresh triggers
    for msg in st.session_state.ui_chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # 7. ❓ Accept incoming natural language historical user queries
    if user_query := st.chat_input("Ask about fiscal allocations, infant mortality data, or child welfare history..."):
        # Append and draw user question bubble instantly
        st.session_state.ui_chat_history.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.write(user_query)

        # Process the active memory list based on the chosen strategy layout
        processed_history = run_memory_routing(st.session_state.model_chat_history)

        # 8. Extract document chunks matching user query coordinates
        with st.spinner("🔍 Extracting historical vector matching records..."):
            matched_docs = retriever.invoke(user_query)
        
        # Assemble document text context strings along with tracking properties
        context_chunks = []
        sources_used = set()
        
        for d in matched_docs:
            page_text = str(d.page_content).strip()
            
            # Additional safety filter layer: Skip corrupted float arrays if any anomalies slip through
            if "e-0" in page_text and "," in page_text:
                continue
                
            source_file = d.metadata.get("source", "GovInfo Resource")
            page_num = d.metadata.get("page", 1)
            fiscal_yr = d.metadata.get("fiscal_year", 0)
            
            year_tag = f" (Fiscal Year: {fiscal_yr})" if fiscal_yr > 0 else ""
            context_chunks.append(f"[Source: {source_file} Page {page_num}{year_tag}]\nContent: {page_text}")
            sources_used.add(f"📄 `{source_file}` | Page {page_num}{year_tag}")
            
        context_string = "\n\n".join(context_chunks)

        # 9. Formulate prompt topology
        contextual_prompt = ChatPromptTemplate.from_messages([
            ("system", f"""You are a precise, technical AI archivist specialized in 20th-century US Government history. 
Answer the user's latest question using *only* the provided historical text chunks as your factual context.
If the context maps do not contain the answer, state clearly that the information is missing from the database.
Do not output raw floating-point vector lists, comma-separated mathematical coordinates, or database arrays.

---
HISTORICAL REPOSITORY TEXT CONTEXT:
{context_string}
---"""),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}")
        ])

        # 10. Generate answer output streaming containers
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response = ""
            
            try:
                formatted_messages = contextual_prompt.format_messages(
                    history=processed_history,
                    question=user_query
                )
                
                # Stream the engine tokens word-by-word into the interface canvas live
                for chunk in llm.stream(formatted_messages):
                    full_response += chunk
                    response_placeholder.markdown(full_response + "▌")
                
                # Append verified data lineage trace tags to the bottom of the answer block
                if sources_used:
                    full_response += "\n\n**Verified GovInfo Sources Utilized for Context Verification:**\n" + "\n".join([f"➔ {s}" for s in sources_used])
                response_placeholder.markdown(full_response)
                
                # 💾 Commit results into the session states to maintain historical consistency
                st.session_state.ui_chat_history.append({"role": "assistant", "content": full_response})
                st.session_state.model_chat_history.append(HumanMessage(content=user_query))
                st.session_state.model_chat_history.append(AIMessage(content=full_response))
                
            except Exception as chain_err:
                st.error(f"❌ Execution failure within LangChain pipeline sequence: {str(chain_err)}")
