from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings

#from langchain.text_splitter import RecursiveCharacterTextSplitter

#text = """Artificial Intelligence (AI) is transforming industries.
#It enables automation, decision-making, and personalization.
#Large Language Models (LLMs) like GPT and Gemini help with text understanding and generation."""

#splitter = RecursiveCharacterTextSplitter(
#    chunk_size=50,      # max chars per chunk
#    chunk_overlap=10    # overlap between chunks
#)

#chunks = splitter.split_text(text)

#for i, chunk in enumerate(chunks):
#    print(f"Chunk {i+1}: {chunk}")


# Function to load documents (pdf/docx)
def load_documents(file_path):
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith(".docx"):
        loader = Docx2txtLoader(file_path)
    else:
        raise ValueError("Unsupported file format. Use PDF or DOCX.")
    
    documents = loader.load()
    return documents

# Function to split documents
def split_documents(documents, chunk_size=500, chunk_overlap=50):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    splits = text_splitter.split_documents(documents)
    return splits

# Function to generate embeddings
def get_document_embeddings(splits):
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    document_embeddings = embeddings.embed_documents([split.page_content for split in splits])
    return document_embeddings


# ---------------- USAGE ----------------
if __name__ == "__main__":
    # Correct path
    file_path = r"assets\Internship Report.pdf"   # ✅ raw string

    docs = load_documents(file_path)

    # Split into chunks
    splits = split_documents(docs)

    # Get embeddings
    doc_embeddings = get_document_embeddings(splits)

    print(f"Number of chunks: {len(splits)}")
    print(f"Embedding dimension: {len(doc_embeddings[0])}")
    print(doc_embeddings[:2])  # preview first 2 embeddings

    #Creating and persist  Chroma vector store
        #Creating and persist Chroma vector store
    from langchain_community.vectorstores import Chroma  
    embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    collection_name = "vibrag_ai_docs"

    vector_store = Chroma.from_documents(
        documents=splits,
        embedding=embedding_function,
        collection_name=collection_name,
        persist_directory="chroma_store"   # persist to disk
    )

    print(f"Vector store created with {len(splits)} documents.")
    
    #Perform  similarity search
    query = "What is the main topic of the report?"
    # Perform similarity search (top 3 most similar chunks)
    results = vector_store.similarity_search(query, k=3)

    # Print results
    for i, doc in enumerate(results, 1):
        source = doc.metadata.get("source", "unknown")  # fallback if source not present
        print(f"--- Result {i} ---")
        print(f"Source: {source}")
        print(f"Content: {doc.page_content}\n")
    
    #Now we have the chunks and their embeddings stored in the vector store.
    #We cant put vector store in invoke
    #So we need to convert the vector store to a retriever
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    retriever_results = retriever.invoke(query)
    #used for question answering
    from langchain.prompts import PromptTemplate
    from langchain_core.prompts import ChatPromptTemplate 
    template= """
    Answer the question only on the following context: {context}
    Question: {question}
    Answer:"""
    prompt = ChatPromptTemplate.from_template(template)
    from langchain_core.runnables import RunnablePassthrough
    #Rannable passthrough is used to pass the question directly that is what ever i am passing is the question
    rag_chain=({"context": retriever,"question": RunnablePassthrough()}|prompt )
    #Now we can use the rag_chain to answer questions
    question = "What is the main topic of the report?" 
    print(f"chain invoked: {rag_chain.invoke(question)}")
    #We are passing the document object  message=[HumanMessage(content="...")]  we want to pass page_content
    def doc2str(docs):
        return "\n\n".join([doc.page_content for doc in docs]) #output string with \n\n between each doc
    rag_chain=({"context": retriever|doc2str,"question": RunnablePassthrough()}|prompt )#See img 1 in asset
    print(f"chain invoked: {rag_chain.invoke(question)}")
    from langchain_google_genai import ChatGoogleGenerativeAI
    from dotenv import load_dotenv
    load_dotenv()
    pdf_chat_model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0.2,
    max_output_tokens=216,
    )
    rag_chain=(
        {"context": retriever|doc2str,"question": RunnablePassthrough()} | 
        prompt | #prompt is used to format the question and context
        pdf_chat_model|
        StrOutputParser()  # Parse the output to string
    )
    response=rag_chain.invoke(question)
    print(f"Response: {response}")

    #introduce a chat history
    from langchain_core.messages import HumanMessage, AIMessage
    from langchain_core.output_parsers import StrOutputParser
    chat_history = []
    chat_history.extend([
        HumanMessage(content=question),
        AIMessage(content=response)
    ])

    from langchain_core.prompts import MessagePlaceholder
    chat_prompt = (
        "Give a chat history and the latest user question, "
        "which might reference context in the chat history,"
        "formulate a standalone question which can be understood,"
        "without the chat history.Dont answer the question,"
        "just reformulate it if needed and otherwise return it as it is.")
    chat_prompt_template = ChatPromptTemplate.from_messages([
        ("system", chat_prompt),
        MessagePlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])
    chat_chain = chat_prompt_template | pdf_chat_model | StrOutputParser()
    chat_chain.invoke({"input": "What is the main topic of the report?",
                       "chat_history": chat_history})
    #history aware
    from langchain_chains import create_history_aware_retriever
    history_aware_retriever = create_history_aware_retriever(
        pdf_chat_model,retriever,chat_prompt_template
    )
    history_aware_retriever.invoke({
        "input": "What is the main topic of the report?",
        "chat_history": chat_history
    })
    from langchain.chains import create_retrival_chain
    from langchain.chains.combine_documents import create_stuff_documents_chain
    qa_prompt=ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant that answers questions based on the provided context."),
        ("system","Context: {context}"),
        MessagePlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])
    question_answering_chain = create_stuff_documents_chain(pdf_chat_model, qa_prompt)
    rag_chain = create_retrival_chain(
        history_aware_retriever,
        question_answering_chain,   
    )
    response = rag_chain.invoke({
        "input": "What is the main topic of the report?",
        "chat_history": chat_history
    })


