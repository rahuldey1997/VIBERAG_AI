from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser

# ---- Model ----
chat_model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0.2,
    max_output_tokens=512,
)

# ---- Memory ----
memory = ConversationBufferMemory(return_messages=True)

# ---- Prompt ----
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that answers questions based only on the provided context and chat history."),
    ("human", "{input}")
])

# ---- Chain with memory ----
base_chain = LLMChain(
    llm=chat_model,
    prompt=qa_prompt,
    memory=memory
)

# ---- Add parser using pipeline ----
chain = base_chain | StrOutputParser()

# ---- Usage ----
def chain_rag(question: str):
    return chain.invoke({"input": question})
