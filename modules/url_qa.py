import streamlit as st
from utils.url_utils import scrape_webpage, split_documents, create_vector_store, chain_rag

def url_qa():
    st.title("🌐 Webpage Q&A Chatbot")

    # Input URL
    url = st.text_input("🔗 Enter a webpage URL:")
    status_placeholder = st.empty()

    # Initialize session state
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    if "last_url" not in st.session_state:
        st.session_state.last_url = ""

    # Scrape only if a new URL is entered
    if url and url != st.session_state.last_url:
        with st.spinner("🔍 Scraping and processing webpage..."):
            try:
                docs = scrape_webpage(url)
                splits = split_documents(docs)
                st.session_state.vector_store = create_vector_store(splits)
                st.session_state.last_url = url
                status_placeholder.success("✅ Webpage processed successfully!")
            except ValueError as e:
                st.error(str(e), icon="🚫")
                return

    # Ask question only if vector_store is available
    if st.session_state.vector_store:
        question = st.text_input("💬 Ask a question about the webpage:")
        if question:
            with st.spinner("⏳ Generating answer..."):
                answer = chain_rag(question, st.session_state.vector_store,url)
            st.write("🤖 Answer:", answer)


if __name__ == "__main__":
    url_qa()
