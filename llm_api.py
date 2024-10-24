from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
from contextlib import asynccontextmanager
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.schema import HumanMessage, AIMessage

PDF_PATH = "diseaseInfo.pdf"
DB_DIR = "chroma_db"
LOCAL_MODEL = "llama3.2"

chat_histories: Dict[str, List] = {}
vector_db = None

def initialize_rag_system():
    global vector_db
    
    embedding = OllamaEmbeddings(model=LOCAL_MODEL)
    persist_directory = os.path.join(os.getcwd(), DB_DIR)
    
    if os.path.exists(persist_directory):
        print("Loading existing ChromaDB embeddings...")
        vector_db = Chroma(
            persist_directory=persist_directory,
            embedding_function=embedding,
            collection_name="local-rag"
        )
    else:
        print("No existing embeddings found. Creating new embeddings...")
        loader = UnstructuredPDFLoader(file_path=PDF_PATH)
        data = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = text_splitter.split_documents(data)
        
        vector_db = Chroma.from_documents(
            documents=chunks, 
            embedding=embedding,
            collection_name="local-rag",
            persist_directory=persist_directory
        )

@asynccontextmanager
async def lifespan(app: FastAPI):
    initialize_rag_system()
    yield
    if vector_db:
        vector_db.persist()

app = FastAPI(title="RAG System API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    question: str
    session_id: str

class ChatResponse(BaseModel):
    answer: str
    sources: List[str]

def get_qa_chain(session_id: str):
    """Create QA chain with session-specific memory"""
    global vector_db
    
    if vector_db is None:
        initialize_rag_system()
    
    llm = ChatOllama(model=LOCAL_MODEL)
    
    if session_id not in chat_histories:
        chat_histories[session_id] = []
    
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
        k=10
    )
    
    for msg in chat_histories[session_id]:
        if isinstance(msg, HumanMessage):
            memory.chat_memory.add_user_message(msg.content)
        elif isinstance(msg, AIMessage):
            memory.chat_memory.add_ai_message(msg.content)

    custom_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful AI assistant. Use both the document context AND the chat history 
        to provide informed and contextual responses. When responding:
        1. Consider previous exchanges in the chat history
        2. Reference relevant previous discussions when appropriate
        3. Maintain continuity of any topics or themes discussed
        4. If the current question relates to previous discussion, acknowledge that connection
        
        For document-specific questions, prioritize information from the provided context."""),
        ("human", "Previous Discussion Context: {chat_history}"),
        ("human", "Document Context: {context}"),
        ("human", "Current Question: {question}"),
        ("human", "AI: "),
    ])

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_db.as_retriever(search_kwargs={"k": 4}),
        memory=memory,
        combine_docs_chain_kwargs={
            "prompt": custom_prompt,
            "document_separator": "\n---\n",
        },
        return_source_documents=True,
        verbose=False,
        chain_type="stuff",
    )

# API Endpoints
@app.get("/")
async def root():
    return {"message": "plant LLM"}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Process a chat message and return response"""
    try:
        # Get QA chain with session-specific memory
        qa_chain = get_qa_chain(request.session_id)
        
        # Process the question
        result = qa_chain({"question": request.question})
        
        # Update chat history
        if request.session_id not in chat_histories:
            chat_histories[request.session_id] = []
        
        chat_histories[request.session_id].extend([
            HumanMessage(content=request.question),
            AIMessage(content=result['answer'])
        ])
        
        # Prepare sources
        sources = [doc.page_content[:100] + "..." for doc in result['source_documents']]
        
        return ChatResponse(
            answer=result['answer'],
            sources=sources
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/clear-history/{session_id}")
async def clear_history(session_id: str):
    """Clear chat history for a specific session"""
    if session_id in chat_histories:
        chat_histories.pop(session_id)
        return {"message": f"Chat history cleared for session {session_id}"}
    raise HTTPException(status_code=404, detail="Session not found")

@app.post("/reinitialize")
async def reinitialize():
    """Force reinitialization of the RAG system"""
    try:
        initialize_rag_system()
        return {"message": "RAG system reinitialized successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("llm_api:app", host="0.0.0.0", port=8000, reload=True)