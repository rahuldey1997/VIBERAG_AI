import streamlit as st
from PIL import Image
import base64
from langchain_google_genai import ChatGoogleGenerativeAI,GoogleGenerativeAIEmbeddings
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

# Import page functions from modules folder
from modules.chat_with_me import chat_with_me
from modules.pdf_qa import pdf_qa
from modules.pdf_summarizer import pdf_summarizer
from modules.url_qa import url_qa


def main():
    # Sidebar Navigation
    logo = Image.open("assets\VIBERAG-Photoroom.png")
    st.sidebar.image(logo, width=120)
    st.sidebar.title("VIBERAG_AI 🤖")

    choice = st.sidebar.radio(
        "Go to:",
        ["Introduction", "Chat with me", "PDF Q&A", "PDF Summarizer", "Web Q&A"]
    )
    # Top Header (logo + orange VIBERAG_AI text side by side)
    st.markdown(
        f"""
        <style>
            .app_header {{
                position: relative;
                top: 10px;
                display: flex;
                align-items: center;
                justify-content: center;
                gap: 12px;
                padding: 0;
                margin: 0;
            }}
            .app_header h1 {{
                margin: 0;
                font-size: 42px;
                color: #FF6F00;
            }}
        </style>

        <div class="app_header">
            <h1>VIBERAG_AI</h1>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.write("---")

    # Pages
    if choice == "Introduction":
        st.subheader("🚀 Welcome to VIBERAG_AI")
        st.write("This is an AI-powered app. Choose from the left sidebar to explore:")
        st.markdown(
            """
            - 💬 **Chat with me**  
            - 📄 **Ask Questions from a PDF**  
            - ✨ **Summarize a PDF**
            - 🌐 **Web Q&A** 
            """
        )

    elif choice == "Chat with me":
        st.subheader("💬 Chat with VIBERAG_AI")
        chat_with_me()

    elif choice == "PDF Q&A":
        pdf_qa()

    elif choice == "PDF Summarizer":

        pdf_summarizer()
    elif choice == "Web Q&A":
        #st.subheader("🌐 Web Q&A")
        #st.write("This feature is under development. Stay tuned!")
        url_qa()


if __name__ == "__main__":
    main()
