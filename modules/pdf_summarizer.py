import streamlit as st
import tempfile
from utils.pdf_summarize_utils import summarize_pdf
def pdf_summarizer():
    st.title("📄 PDF/DOCX Summarizer")

    pdf_file = st.file_uploader("Upload your PDF or DOCX", type=["pdf", "docx"])
    status_placeholder = st.empty()

    if pdf_file is not None:
        # Save uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix=pdf_file.name) as tmp_file:
            tmp_file.write(pdf_file.read())
            file_path = tmp_file.name

        # Feedback
        if pdf_file.type == "application/pdf":
            status_placeholder.success("✅ PDF uploaded successfully!")
        elif pdf_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            status_placeholder.success("✅ DOCX uploaded successfully!")
        else:
            st.error("❌ Unsupported file format", icon="🚫")
            return

        # Length slider
        length = st.slider("📏 Select summary length (words)", 100, 500, 200, step=20)

        # Summarize button
        if st.button("✨ Generate Summary"):
            with st.spinner("⏳ Summarizing document..."):
                summary = summarize_pdf(file_path, length)
            st.subheader("📌 Summary")
            st.write(summary)
            # ✅ Show summary with copy button
            st.code(summary, language="markdown")  # adds copy-to-clipboard button automatically



# Run the app
if __name__ == "__main__":
    pdf_summarizer()