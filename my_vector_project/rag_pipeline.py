import os
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

def run_local_rag():
    DB_FOLDER = "./chroma_db"
    
    if not os.path.exists(DB_FOLDER):
        print("🚨 Local vector database folder not found! Run your 'python ingest.py' script completely first.")
        return

    print("🧠 Connecting to your local ChromaDB vector index...")
    # By omitting the 'embedding' parameter, LangChain cleanly defaults to Chroma's native model
    vectorstore = Chroma(
        persist_directory=DB_FOLDER,
        collection_name="knowledge_base",
    )
    
    # Configure semantic search retriever parameters to pull top 3 matching chunks
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    print("🤖 Initializing local Llama 3.2 engine integration via Ollama...")
    # Map directly to your background running Ollama local server port
    llm = OllamaLLM(model="llama3.2", base_url="http://localhost:11434", temperature=0.2)

    prompt_template = """You are a highly precise, technical AI engineering assistant. 
Answer the question below using *only* the provided document text chunks as factual context. 
If the context does not contain the answer, state clearly that the information is missing from the database.

---
DOCUMENT CONTEXT ENCLOSURE MAP:
{context}
---

QUESTION: {question}

CONCISE FACTUAL ANSWER:"""

    prompt = ChatPromptTemplate.from_template(prompt_template)

    def format_docs(docs):
        formatted_chunks = []
        for d in docs:
            source = d.metadata.get("source", "Unknown Resource")
            page = d.metadata.get("page", 1)
            formatted_chunks.append(f"[Source: {source} (Page {page})]\nContent: {d.page_content}")
        return "\n\n".join(formatted_chunks)

    # Establish the production execution pipeline layout chain
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    print("\n✅ RAG pipeline linked! Ask questions from your 65 documents.")
    print("(Type 'exit' to quit the application session)\n")

    while True:
        user_question = input("❓ Question: ")
        if user_question.lower() in ['exit', 'quit']:
            break
        if not user_question.strip():
            continue
            
        print("🔍 Searching vectors and generating answer...")
        try:
            response = rag_chain.invoke(user_question)
            print(f"\n🤖 Answer:\n{response}\n")
            print("─" * 40)
        except Exception as e:
            print(f"❌ Execution error: {str(e)}\n")

if __name__ == "__main__":
    run_local_rag()
