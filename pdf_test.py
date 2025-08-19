from pdf_conversation_main import create_vector_store, chain_rag,load_documents, split_documents

if __name__ == "__main__":
    file_path = r"assets\Internship Report.pdf"

    # Step 1: Initialize vector store once
    
    # Step 1: Load and split
    docs = load_documents(file_path)
    splits = split_documents(docs)

    # Step 2: Create vector store
    vector_store = create_vector_store(splits)
    

    # Step 2: Ask questions
    q1 = "is there any optimization technique used?"
    print("Q1:", chain_rag(q1, vector_store))

    q2 = "what are the techniques used?"
    print("Q2:", chain_rag(q2, vector_store))

    q3 = "What is the minimum cost?"
    print("Q3:", chain_rag(q3, vector_store))

    q4 = "Which optimization technique gives minimum cost?"
    print("Q4:", chain_rag(q4, vector_store))
    q5 = "What was my second question? "
    print("Q4:", chain_rag(q5, vector_store))

