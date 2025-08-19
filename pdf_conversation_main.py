# ---------------- Imports ----------------
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv


# ---------------- Load Documents ----------------
def load_documents(file_path):
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith(".docx"):
        loader = Docx2txtLoader(file_path)
    else:
        raise ValueError("Unsupported file format. Use PDF or DOCX.")
    return loader.load()


# ---------------- Split Documents ----------------
def split_documents(documents, chunk_size=500, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(documents)


# ---------------- Create Vector Store ----------------
def create_vector_store(splits, collection_name="vibrag_ai_docs"):
    embedding_function = embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = Chroma.from_documents(
        documents=splits,
        embedding=embedding_function,
        collection_name=collection_name,
        persist_directory="chroma_store"
    )
    return vector_store


# ---------------- Model ----------------
load_dotenv()
chat_model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0.2,
    max_output_tokens=512,
)


# ---------------- Prompt ----------------
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that answers questions based only on the provided context and chat history."),
    ("system", "Chat history:\n{chat_history}"),
    ("human", "{input}")
])


# ---------------- Chatbot Function ----------------
chat_history = []  # keeps track of conversation

def chain_rag(question, vector_store, k=5, show_docs=True):
    # 1. Always force semantic search on each question
    retriever = vector_store.as_retriever(search_kwargs={"k": k})
    docs = retriever.invoke(question)


    # 2. Build context string from retrieved docs
    context = "\n\n".join([doc.page_content for doc in docs])

    # 3. Build chat history string
    history_str = "\n".join(
        [f"Human: {m.content}" if isinstance(m, HumanMessage) else f"AI: {m.content}" for m in chat_history]
    )

    # 4. Format prompt
    prompt = qa_prompt.format(context=context, chat_history=history_str, input=question)

    # 5. Get LLM response
    response = chat_model.invoke(prompt)
    answer = response.content

    # 6. Update chat history
    chat_history.append(HumanMessage(content=question))
    chat_history.append(AIMessage(content=answer))

    return answer



# ---------------- Usage ----------------
if __name__ == "__main__":
    file_path = r"assets\Internship Report.pdf"  # your PDF file

    # Step 1: Load and split
    docs = load_documents(file_path)
    splits = split_documents(docs)

    # Step 2: Create vector store
    vector_store = create_vector_store(splits)

    # Step 3: Ask questions
    result = chain_rag("is there any optimization technique used?", vector_store)
    print("Q1:", result)

    result = chain_rag("what are the technique used?.", vector_store)
    print("Q2:", result)

    result = chain_rag("What is the minimum cost", vector_store)
    print("Q3:", result)
    result = chain_rag("Which optimization technique give  minimum cost", vector_store)
    print("Q4:", result)
