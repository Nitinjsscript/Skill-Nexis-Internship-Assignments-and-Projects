import streamlit as st
import chromadb

st.set_page_config(page_title="Advanced Semantic Document Base", layout="wide")
st.title("🗂️ Intelligent Vector Knowledge Base Explorer")
st.markdown("Query system documentation, technical specifications, and raw research datasets concurrently.")

@st.cache_resource
def load_db_instance():
    try:
        client = chromadb.PersistentClient(path="./chroma_db")
        return client.get_collection(name="knowledge_base")
    except Exception:
        return None

collection = load_db_instance()

if collection is None:
    st.error("🚨 Vector index files targets not compiled. Create a `/data` folder, drop your items inside, and run `python ingest.py` first.")
else:
    # Sidebar configurations
    st.sidebar.header("⚙️ Search Parameter Rules")
    min_confidence = st.sidebar.slider("Minimum Relevance Matching Score Threshold (%)", min_value=0, max_value=100, value=30)
    max_results = st.sidebar.number_input("Max Result Extractions Count", min_value=1, max_value=10, value=4)

    search_query = st.text_input("Enter search phrase or functional conceptual query:", placeholder="e.g., how to declare frontmatter configurations or NLP parsing toolkits")

    if search_query:
        with st.spinner("Executing mathematical distance mapping vectors calculation..."):
            query_results = collection.query(
                query_texts=[search_query],
                n_results=int(max_results)
            )
            
            if query_results and query_results['documents'] and query_results['documents'][0]:
                st.subheader("🎯 High-Relevance Extracts Discovered")
                any_match_shown = False
                
                for idx, (content, meta, distance) in enumerate(zip(
                    query_results['documents'][0], 
                    query_results['metadatas'][0], 
                    query_results['distances'][0]
                )):
                    # Compute percentage accuracy allocation inverse mapping values metrics
                    score = (1.0 - max(0.0, min(distance, 1.0))) * 100
                    
                    if score < min_confidence:
                        continue
                        
                    any_match_shown = True
                    
                    # Formulate structured location tracking display labels
                    breadcrumb = f"📄 File: {meta.get('source', 'Unknown')}"
                    if meta.get('file_type') == 'pdf':
                        breadcrumb += f" | Page {int(meta.get('page', 1))}"
                        
                    # Structural Markdown heading tracker layout traces
                    headings = [meta[h] for h in ["Header_1", "Header_2", "Header_3"] if h in meta]
                    if headings:
                        breadcrumb += f" ➔ {' ➔ '.join(headings)}"
                        
                    # Display associated frontmatter details if pulled out
                    frontmatter_info = ""
                    if "frontmatter_name" in meta:
                        frontmatter_info = f" [Skill Target: {meta['frontmatter_name']}]"

                    with st.expander(f"Match #{idx + 1} | {breadcrumb}{frontmatter_info} | Confidence score: {score:.1f}%"):
                        # If structural code blocks exist inside markdown extract, render cleanly
                        if meta.get("file_type") == "markdown" and ("```" in content or "    " in content):
                            st.markdown("**Structured Source Code Fragment Block:**")
                            st.code(content, language="markdown")
                        else:
                            st.markdown("**Extracted Text Content:**")
                            st.info(content)
                            
                if not any_match_shown:
                    st.warning("⚠️ Matches found, but their scores fell below your slider minimum confidence threshold setting.")
            else:
                st.info("🔍 No relevant elements discovered inside the collection vector database cluster matching those criteria parameters.")
