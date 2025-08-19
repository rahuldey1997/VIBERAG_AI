from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
import random
import string

from dotenv import load_dotenv

load_dotenv()

# Initialize model
chat_model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0.2,
    max_output_tokens=216,
)

# Invoke model
result = chat_model.invoke("What is the capital of France?")
#print("Raw Result:", result)
#print("Result Content:", result.content)
parser = StrOutputParser()
# Use parser correctly
#print("Parsed Result:", parser.invoke(result))  # output: The capital of France is Paris
# Using chain
chain = chat_model | parser
result = chain.invoke("What is the capital of France?")
#print("Chain Result:", result)

#while True:
 #   question = input("Ask a question: ")
  #  if question.lower() == "exit":
   #     break
   # response = chain.invoke([HumanMessage(content=question)])
    #print("Response:", response)
result=chain.invoke(
    [
        HumanMessage(content="Hi my name is rahul?"),
        AIMessage(content="Hi rahul whats up."),
        HumanMessage(content="What is my name? "),
    ]
)
#print("Chat Result:", result)
store={}
def get_session_history(session_id:str)->BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]
chat_model_with_history = RunnableWithMessageHistory(chain,get_session_history)
#result=chat_model_with_history.invoke([HumanMessage(content="What is my name?")], session_id="session_1").content
#or
def generate_session_id():
    rand_str = ''.join(random.choices(string.ascii_lowercase + string.digits, k=5))
    return f"session_{rand_str}"

# Create random session_id
session_id = generate_session_id()
print("Using session_id:", session_id)

config = {"configurable": {"session_id": session_id}}
chat_model_with_history.invoke(
    [HumanMessage(content="My name is Rahul")],
    config=config
)
query = "What is my name?"
result = chat_model_with_history.invoke(
    [HumanMessage(content=query)],
    config=config
)

print("History-aware Result:", result)

