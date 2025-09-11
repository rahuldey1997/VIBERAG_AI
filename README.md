# Context-Aware Q\&A & Summarization AI

An AI application built with **LangChain** and **Streamlit** for context-aware Q\&A and document summarization from PDFs, Docx files, and web content.

## Features

* **Context-Aware Q\&A**: Ask questions and get answers grounded in the content of uploaded documents or web pages.
* **Summarization**: Generate concise summaries for large documents.
* **Multi-Source Support**: Works with PDFs, Docx files, and web pages.
* **Semantic Search**: Quickly retrieve relevant sections using vector-based search.

## Tech Stack

* **LangChain**: Orchestrates LLM workflows and RAG pipeline.
* **Streamlit**: Interactive web app interface.
* **LLM**: Integrated **Gemini-2.5-Flash-Lite** for natural language understanding and generation.
* **Embeddings**: Used **all-MiniLM-L6-v2** for semantic vectorization.
* **Vector Store**: **Chroma** for efficient retrieval in semantic search.

## Implementation Highlights

* Built an **end-to-end RAG (Retrieval-Augmented Generation) pipeline** for document ingestion, processing, and semantic search.
* Seamlessly integrates document embeddings and retrieval for **contextualized Q\&A**.
* Interactive **Streamlit UI** for document upload, question input, and real-time summarization.

