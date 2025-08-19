# ---------------- Imports ----------------
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader

def load_documents(file_path):
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith(".docx"):  # Only accept modern Word format
        loader = Docx2txtLoader(file_path)
    else:
        raise ValueError("Unsupported file format. Please upload PDF or DOCX.")
    return loader.load()



# ---------------- Split Documents ----------------
def split_documents(documents, chunk_size=1000, chunk_overlap=100):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(documents)


# ---------------- Model ----------------
load_dotenv()
summaries_model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0.3,
    max_output_tokens=1024,
)


# ---------------- Summarizer Prompt ----------------
summarizer_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that summarizes documents."),
    ("human", "Summarize the following text into a {length} summary:\n\n{text}")
])


# ---------------- Summarizer Function ----------------
def summarize_pdf(file_path, length):
    # 1. Load and split
    docs = load_documents(file_path)
    splits = split_documents(docs)

    chunk_summaries = []

    # 2. Summarize each chunk
    for chunk in splits:
        prompt = summarizer_prompt.format(text=chunk.page_content, length=length)
        response = summaries_model.invoke(prompt)
        chunk_summaries.append(response.content)

    # 3. Combine all chunk summaries into a final summary
    final_prompt = summarizer_prompt.format(
        text="\n\n".join(chunk_summaries),
        length=length
    )
    final_response = summaries_model.invoke(final_prompt)

    return final_response.content

