from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.document_loaders import OnlinePDFLoader
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.schema import HumanMessage, AIMessage
import os

local_path = "diseaseInfo.pdf"
embedding = OllamaEmbeddings(model="llama3.2")
if local_path:
    loader = UnstructuredPDFLoader(file_path=local_path)
    data = loader.load()
else:
    print("Upload a PDF file")

persist_directory = os.path.join(os.getcwd(), "chroma_db")

if os.path.exists(persist_directory):
    print("Loading existing ChromaDB embeddings...")
    vector_db = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding,
        collection_name="local-rag"
    )
else:
    print("No existing embeddings found. Creating new embeddings...")
    loader = UnstructuredPDFLoader(file_path=local_path)
    data = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(data)
    
    vector_db = Chroma.from_documents(
        documents=chunks, 
        embedding=OllamaEmbeddings(model="llama3.2"),
        collection_name="local-rag",
        persist_directory=persist_directory
    )

local_model = "llama3.2"
llm = ChatOllama(model=local_model)
QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""You are an AI language model assistant. Your task is to generate five
    different versions of the given user question to retrieve relevant documents from
    a vector database. By generating multiple perspectives on the user question, your
    goal is to help the user overcome some of the limitations of the distance-based
    similarity search. Provide these alternative questions separated by newlines.
    Original question: {question}""",
)

retriever = MultiQueryRetriever.from_llm(
    vector_db.as_retriever(), 
    llm,
    prompt=QUERY_PROMPT
)

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    output_key="answer",
    k=10  
)

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

qa_chain = ConversationalRetrievalChain.from_llm(
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
chat_history = []
while True:
    query = input("Human: ")
    if query.lower() in ['exit', 'quit', 'bye']:
        print("AI: Goodbye!")
        break
    
    result = qa_chain({"question": query})
    ai_response = result['answer']
    print("AI:", ai_response)
    
    print("\nRetrieved contexts:")
    for doc in result['source_documents']:
        print(doc.page_content[:50])

    # print("\nSources:")
    # for doc in result['source_documents']:
    #     print(doc.page_content[:100] + "...")
    
    # Update chat history
    chat_history.append(HumanMessage(content=query))
    chat_history.append(AIMessage(content=ai_response))
# Delete all collections in the db
# vector_db.delete_collection()