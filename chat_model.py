from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# Initialize model
chat_model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0.2,
    max_output_tokens=216,
)

# Invoke model
result = chat_model.invoke("What is the capital of France?")

# Print raw result
print("Raw Result:", result)

# Print just content
print("Result Content:", result.content)

# Use parser correctly
output_parser = StrOutputParser()
print(output_parser.invoke(result)) #output : The capital of France is Paris.

#Using chain
chain=chat_model | output_parser
# Invoke chain
result = chain.invoke("What is the capital of France?")

#Embedding
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
# Example usage
document_embeddings=embedding_function.embed_documents([split.page_content for split in splits])