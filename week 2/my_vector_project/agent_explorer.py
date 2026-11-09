import os
import streamlit as st
import chromadb
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool

# 1. Framework Workspace Visual Setup
st.set_page_config(page_title="AI Agentic Assistant", layout="wide")
st.title("🤖 Autonomous AI Agent Workspace")
st.markdown("This agent dynamically decides whether to search local documents, calculate math, or scan the live web.")

# 2. Cache Local Resources (Database & Llama 3.2 Model Engine)
@st.cache_resource
def initialize_agent_resources():
    DB_FOLDER = "./chroma_db"
    if not os.path.exists(DB_FOLDER):
        return None, None
        
    persistent_client = chromadb.PersistentClient(path=DB_FOLDER)
    from chromadb.utils.embedding_functions.onnx_mini_lm_l6_v2 import ONNXMiniLM_L6_V2
    onnx_embedding_function = ONNXMiniLM_L6_V2()
    
    vectorstore = Chroma(
        client=persistent_client,
        collection_name="knowledge_base",
        embedding_function=onnx_embedding_function
    )
    
    # Agents require deterministic reasoning models; temperature 0 forces logical decisions
    llm = OllamaLLM(model="llama3.2", base_url="http://localhost:11434", temperature=0.0)
    return vectorstore.as_retriever(search_kwargs={"k": 3}), llm

retriever, llm = initialize_agent_resources()

if retriever is None:
    st.error("🚨 Local vector storage target folder `./chroma_db` not discovered. Execute your `python ingest.py` script completely first.")
else:
    # 3. 🛠️ Define Custom Agent Tools Using LangChain @tool Decorators
    
    @tool
    def document_retrieval_tool(query: str) -> str:
        """Useful when you need to answer questions about specific technical documentation files, 
        internal operational policies, coding guidelines, or specific research files like HuSpaCy or NLP papers."""
        # Ensure query is stripped of unwanted characters and remains a pure text string
        clean_query = str(query).strip().replace("[", "").replace("]", "")
        matched_chunks = retriever.invoke(clean_query)
        return "\n\n".join([f"[File: {d.metadata.get('source')} (Page {d.metadata.get('page', 1)})]\nContent: {d.page_content}" for d in matched_chunks])

    @tool
    def calculator_tool(expression: str) -> str:
        """Useful when you need to answer precise mathematical calculation questions, counting days, 
        evaluating formulas, or balancing numerical values. Input must be a valid mathematical text string."""
        import numexpr
        try:
            # Clean expression string from accidental wrapper quotes passed by the LLM
            clean_expr = str(expression).replace("'", "").replace('"', "")
            result = numexpr.evaluate(clean_expr).item()
            return f"The mathematical calculation result is: {result}"
        except Exception as e:
            return f"Failed to calculate. Make sure input expressions are purely numeric strings. Error: {str(e)}"

    @tool
    def live_web_search_tool(query: str) -> str:
        """Useful when you need to answer questions about current events, movie release dates, 
        up-to-date facts, locations, public organizations, or topics not contained in local document repositories."""
        from langchain_community.tools import DuckDuckGoSearchRun
        try:
            search_engine = DuckDuckGoSearchRun()
            return search_engine.invoke(str(query))
        except Exception as e:
            return f"Web search currently throttled or unavailable. Error: {str(e)}"

    # Compile tool routing array dictionary lookups
    tools_list = [document_retrieval_tool, calculator_tool, live_web_search_tool]
    tools_map = {t.name: t for t in tools_list}

    # 4. Initialize Streamlit Browser State Tracking Variables
    if "agent_messages" not in st.session_state:
        st.session_state.agent_messages = []

    # Render previous conversation logs
    for msg in st.session_state.agent_messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # 5. 🚀 Execution Agent Core Reasoning Engine Loop
    if user_prompt := st.chat_input("Ask me a cross-domain question..."):
        st.session_state.agent_messages.append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.write(user_prompt)

        with st.chat_message("assistant"):
            status_container = st.empty()
            response_container = st.empty()
            
            with st.spinner("🤖 Evaluating tool-routing decision trees..."):
                # Define a strict system constraint prompt to prevent the model from outputting arrays
                routing_prompt = f"""You are a master router for an AI agent workspace. 
                Evaluate this exact user question: '{user_prompt}'
                Select the single best tool name from this exact list: [document_retrieval_tool, calculator_tool, live_web_search_tool, none]
                Do not output any numbers, multi-dimensional coordinate maps, or code arrays. Respond with exactly one plain text word."""
                
                tool_decision = llm.invoke(routing_prompt).strip().lower()
                # Clean up any surrounding punctuation or formatting text artifacts
                tool_decision = "".join(c for c in tool_decision if c.isalnum() or c == "_")
                
                if tool_decision in tools_map:
                    status_container.info(f"🔄 **Agent Action:** Routing execution context to `{tool_decision}`...")
                    
                    # Execute tool extraction sequences passing a pure string input query
                    active_tool = tools_map[tool_decision]
                    tool_output = active_tool.invoke(user_prompt)
                    
                    # Synthesize structural output results
                    synthesis_prompt = f"""Synthesize a complete professional text response for this question: '{user_prompt}'
                    Using this extracted context returned by the tool:
                    ---
                    {tool_output}
                    ---
                    State clearly in your answer that you used the {tool_decision} to fetch this validated data."""
                    
                    final_answer = llm.invoke(synthesis_prompt)
                    response_container.markdown(final_answer)
                else:
                    status_container.warning("🧠 **Agent Action:** Question handled using native internal LLM knowledge weights.")
                    final_answer = llm.invoke(user_prompt)
                    response_container.markdown(final_answer)
                    
                # Save status histories
                st.session_state.agent_messages.append({"role": "assistant", "content": final_answer})
