
# ---------------- Imports ----------------

from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
# ---------------- Model ----------------
load_dotenv()
chat_model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0.2,
    max_output_tokens=512,
)
parser= StrOutputParser()
#----------Memory----------------


# ---------------- Prompt ----------------
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that answers questions based only on the provided context and chat history."),
    ("system", "Chat history:\n{chat_history}"),
    ("human", "{input}")
])

# ---------------- Chatbot Function ----------------

chat_history = []  # keeps track of conversation

def chain_rag(question, show_docs=True):
    # 1. Build chat history string
    history_str = "\n".join(
        [f"Human: {m.content}" if isinstance(m, HumanMessage) else f"AI: {m.content}" for m in chat_history]
    )
    

    # 2. Create chain (prompt → model → parser)
    chain = qa_prompt | chat_model | parser

    # 3. Run chain
    answer = chain.invoke({"chat_history": chat_history, "input": question})

    # 4. Update chat history
    chat_history.append(HumanMessage(content=question))
    chat_history.append(AIMessage(content=answer))

    return answer

