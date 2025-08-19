import streamlit as st
import tempfile
from utils.pdf_utils import create_vector_store, chain_rag, load_documents, split_documents

def pdf_qa():
    st.title("📄 PDF/DOCX Q&A Chatbot")

    pdf_file = st.file_uploader("Upload your PDF or DOCX", type=["pdf", "docx"])
    status_placeholder = st.empty()

    if pdf_file is not None:
        # Save uploaded file to a temporary path
        with tempfile.NamedTemporaryFile(delete=False, suffix=pdf_file.name) as tmp_file:
            tmp_file.write(pdf_file.read())
            file_path = tmp_file.name

        # File type feedback
        if pdf_file.type == "application/pdf":
            status_placeholder.success("✅ PDF uploaded successfully!")
        elif pdf_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            status_placeholder.success("✅ DOCX uploaded successfully!")
        else:
            st.error("❌ Unsupported file format", icon="🚫")
            return

        # Load, split, and create vector store
        with st.spinner("🔍 Processing document..."):
            docs = load_documents(file_path)
            splits = split_documents(docs)
            vector_store = create_vector_store(splits)

        # Ask questions
        question = st.text_input("💬 Ask a question:")
        if question:
            with st.spinner("⏳ Generating answer..."):
                answer = chain_rag(question, vector_store)
            st.write("🤖 Answer:", answer)


# Run the app
if __name__ == "__main__":
    pdf_qa()
