import streamlit as st
import tempfile
from utils.chat_utils import chain_rag
import copy

def chat_with_me():

    # Input field with session state
    if "question" not in st.session_state:
        st.session_state.question = ""

    def submit():
        st.session_state.submitted_question = st.session_state.question
        st.session_state.question = ""  # clear input (freeze effect)

    # Input box (clears after submit)
    st.text_input("💬 Ask me:", key="question", on_change=submit)

    # Show answer only after a question is submitted
    if "submitted_question" in st.session_state and st.session_state.submitted_question:
        with st.spinner("⏳ Generating answer..."):  
            # replace with your RAG chain
            answer = chain_rag(st.session_state.submitted_question)
        st.write("🙋 You asked:", st.session_state.submitted_question)
        st.write("🤖 Answer:", answer)



# Run the app
if __name__ == "__main__":
    chat_with_me()
