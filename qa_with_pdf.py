# ----------------- Imports -----------------
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import BaseOutputParser
from dotenv import load_dotenv
import os

# ----------------- Load Documents -----------------
def load_documents(file_path):
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith(".docx"):
        loader = Docx2txtLoader(file_path)
    else:
        raise ValueError("Unsupported file format. Use PDF or DOCX.")
    
    return loader.load()

# ----------------- Split Documents -----------------
def split_documents(documents, chunk_size=500, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(documents)

# ----------------- Vector Store -----------------
def create_vector_store(splits, collection_name="vibrag_ai_docs", persist_dir="chroma_store"):
    embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = Chroma.from_documents(
        documents=splits,
        embedding=embedding_function,
        collection_name=collection_name,
        persist_directory=persist_dir
    )
    return vector_store

# ----------------- Doc2String -----------------
def docs_to_string(docs):
    return "\n".join([doc.page_content for doc in docs])

# ----------------- Output Parser -----------------
class StringOutputParser(BaseOutputParser):
    def parse(self, output: str) -> str:
        return output.strip()

# ----------------- Prompt Template -----------------
prompt_template = """
Answer the question based on the following context and previous conversation:

Conversation History:
{history}

Document Context:
{context}

Question:
{question}

Answer:
"""
prompt = PromptTemplate(input_variables=["history", "context", "question"], template=prompt_template)

# ----------------- Modular RAG Chain -----------------
def rag_chain(question, vector_store, chat_model, chat_history, top_k=3):
    # 1. Retriever
    retriever = vector_store.as_retriever(search_kwargs={"k": top_k})
    docs = retriever.get_relevant_documents(question)

    # 2. Doc2String
    context = docs_to_string(docs)

    # 3. Build prompt with history
    history_text = "\n".join([f"Q: {q}\nA: {a}" for q, a in chat_history])
    formatted_prompt = prompt.format(history=history_text, context=context, question=question)

    # 4. LLM call (runnable passthrough)
    llm_output = chat_model.invoke(formatted_prompt).content

    # 5. Parse output
    answer = StringOutputParser().parse(llm_output)
    
    # 6. Update conversation history
    chat_history.append((question, answer))
    
    return answer

# ----------------- Main Application -----------------
if __name__ == "__main__":
    load_dotenv()  # Load Gemini API keys
    
    # Initialize Gemini chat model
    chat_model = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        temperature=0.2,
        max_output_tokens=500
    )
    
    # Path to your document
    file_path = r"assets\Internship Report.pdf"
    if not os.path.exists(file_path):
        print("File not found. Exiting...")
        exit()
    
    print("Loading document...")
    docs = load_documents(file_path)
    
    print("Splitting document into chunks...")
    splits = split_documents(docs)
    
    print("Creating vector store...")
    vector_store = create_vector_store(splits)
    
    print(f"Document loaded and indexed with {len(splits)} chunks.")
    
    # Initialize conversation history
    chat_history = []

    print("\nYou can now ask questions. Type 'exit' to quit.")
    while True:
        question = input("\nYour Question: ").strip()
        if question.lower() == "exit":
            print("Exiting RAG application.")
            break
        
        answer = rag_chain(question, vector_store, chat_model, chat_history)
        print(f"\nAnswer: {answer}")
