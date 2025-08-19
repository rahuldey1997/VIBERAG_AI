from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain.docstore.document import Document
from urllib.parse import urlparse
from dotenv import load_dotenv
import requests
from bs4 import BeautifulSoup

# ---------------- Validate URL ----------------
def is_valid_url(url: str) -> bool:
    parsed = urlparse(url)
    return all([parsed.scheme in ["http", "https"], parsed.netloc])

# ---------------- Web Scraper ----------------
def scrape_webpage(url):
    if not is_valid_url(url):
        raise ValueError(f"Invalid URL: {url}")
    response = requests.get(url)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    texts = soup.stripped_strings
    full_text = " ".join(texts)
    return [Document(page_content=full_text, metadata={"source": url})]

# ---------------- Split Documents ----------------
def split_documents(documents, chunk_size=500, chunk_overlap=100):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(documents)

# ---------------- Create Vector Store ----------------
def create_vector_store(splits, collection_name="vibrag_ai_docs"):
    embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
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
    ("system", "Context:\n{context}"),
    ("system", "Chat history:\n{chat_history}"),
    ("human", "{input}")
])

# ---------------- Chat History ----------------
chat_history = []

# ---------------- RAG + URL Fallback ----------------
def chain_rag(question, vector_store, url=None, k=3, show_docs=True):
    retriever = vector_store.as_retriever(search_kwargs={"k": k})
    docs = retriever.invoke(question)

    # Check if any docs are found
    if not docs or all(len(doc.page_content.strip()) == 0 for doc in docs):
        if url:
            # Scrape the URL if no relevant docs found
            scraped_docs = scrape_webpage(url)
            splits = split_documents(scraped_docs)
            vector_store = create_vector_store(splits)  # update vector store
            docs = splits  # use new splits for context

    # Build context
    context = "\n\n".join([doc.page_content for doc in docs]) if docs else "No relevant information found."

    # Build chat history string
    history_str = "\n".join(
        [f"Human: {m.content}" if isinstance(m, HumanMessage) else f"AI: {m.content}" for m in chat_history]
    )

    # Format prompt
    prompt = qa_prompt.format(context=context, chat_history=history_str, input=question)

    # Get LLM response
    response = chat_model.invoke(prompt)
    answer = response.content

    # Update chat history
    chat_history.append(HumanMessage(content=question))
    chat_history.append(AIMessage(content=answer))

    return answer
