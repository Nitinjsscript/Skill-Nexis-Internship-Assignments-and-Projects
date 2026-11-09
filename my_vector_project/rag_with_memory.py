import os
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# Choose your memory strategy profile here: "buffer", "window", or "summary"
MEMORY_MODE = "window" 
WINDOW_SIZE = 4 # Only utilized if MEMORY_MODE == "window" (keeps last 4 messages / 2 turns)

def manage_memory_state(chat_history, latest_question, llm):
    """Processes memory history arrays according to the configured state type profile."""
    if MEMORY_MODE == "buffer":
        return chat_history
        
    elif MEMORY_MODE == "window":
        # Keep only the last K messages to prevent context window explosion
        return chat_history[-WINDOW_SIZE:] if len(chat_history) > WINDOW_SIZE else chat_history
        
    elif MEMORY_MODE == "summary":
        if len(chat_history) < 2:
            return chat_history
        
        # Take the oldest turns, ask the local LLM to summarize them, and keep only the summary
        print("📝 Condensing long-term chat turns into a running summary narrative...")
        old_messages = chat_history[:-2]
        formatted_history = "\n".join([f"{'User' if isinstance(m, HumanMessage) else 'AI'}: {m.content}" for m in old_messages])
        
        summary_prompt = f"""Summarize the following conversation history between a User and an AI assistant into a single concise, paragraph. 
Focus strictly on the technical entities and questions discussed.
        
Conversation History:
{formatted_history}
        
Concise Summary Paragraph:"""
        
        running_summary = llm.invoke(summary_prompt)
        
        # Return the summary formatted as a SystemMessage directive + the most recent active turn
        return [SystemMessage(content=f"Summary of previous conversation turns: {running_summary}")] + chat_history[-2:]
        
    return chat_history

def run_conversational_rag():
    DB_FOLDER = "./chroma_db"
    if not os.path.exists(DB_FOLDER):
        print("🚨 Chroma storage vectors folder not found. Run ingest.py first.")
        return

    print(f"🧠 Connecting to local vector index database (Active Memory Mode: {MEMORY_MODE.upper()})...")
    vectorstore = Chroma(persist_directory=DB_FOLDER, collection_name="knowledge_base")
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    llm = OllamaLLM(model="llama3.2", base_url="http://localhost:11434", temperature=0.2)

    # 1. System Prompt detailing how to leverage history context maps
    contextual_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a precise, technical AI engineering assistant. 
Answer the user's latest question using *only* the provided document chunks as context.
If the conversation history or document context does not provide the answer, state that it's missing from the database.

---
DOCUMENT CONTEXT ENCLOSURE MAP:
{context}
---"""),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}")
    ])

    def format_docs(docs):
        return "\n\n".join([f"[Source: {d.metadata.get('source')}]\nContent: {d.page_content}" for d in docs])

    # 2. Modern declarative execution chain sequence
    rag_chain = (
        {
            "context": lambda x: format_docs(retriever.invoke(x["question"])),
            "history": lambda x: x["history"],
            "question": lambda x: x["question"]
        }
        | contextual_prompt
        | llm
        | StrOutputParser()
    )

    # In-memory list tracking the real message state objects
    master_chat_history = []

    print("\n✅ Conversational Memory RAG Pipeline Linked!")
    print("(Type 'exit' to terminate the app session)\n")

    while True:
        user_input = input("❓ Question: ")
        if user_input.lower() in ['exit', 'quit']:
            break
        if not user_input.strip():
            continue

        # Process the active memory list based on the chosen strategy layout
        processed_history = manage_memory_state(master_chat_history, user_input, llm)

        print("🔍 Extracting vector matching records and synthesizing memory stream...")
        try:
            response = rag_chain.invoke({
                "question": user_input,
                "history": processed_history
            })
            
            print(f"\n🤖 Answer:\n{response}\n")
            print("─" * 50)
            
            # Append active turns into the raw master state tracking history arrays
            master_chat_history.append(HumanMessage(content=user_input))
            master_chat_history.append(AIMessage(content=response))
            
        except Exception as e:
            print(f"❌ Execution chain block error: {str(e)}\n")

if __name__ == "__main__":
    run_conversational_rag()
