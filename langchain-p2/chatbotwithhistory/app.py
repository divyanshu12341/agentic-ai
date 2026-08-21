import os 
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage,AIMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
model = ChatGroq(model="llama-3.1-8b-instant")
result = model.invoke([
    HumanMessage(content = "Hi my name is Divyanshu and i am an AI Engineer"),
    AIMessage(content = "Hello divyanshu.Nice to meet you.What are your responsibilities as an AI Engineer"),
    HumanMessage(content = "Hi what's my name and what i do? ")
])
print(result)

## Message history 
## We use message history to wrap our model and make it stateful. 
##  This will keep track of input and output of our model and store them 
## in the datastore

store = {}
def get_session_history(session_id:str)->BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

with_message_history = RunnableWithMessageHistory(model, get_session_history)
config = {"configurable":{"session_id":"chat1"}}
response = with_message_history.invoke(
    [HumanMessage(content = "Hi my name is divyanshu and i am learning Langchain ")],
    config = config
)
response1 = with_message_history.invoke([
    HumanMessage(content = "What's my name? ")
],
config = config)
config1 = {"configurable":{"session_id":"chat2"}}
response1 = with_message_history.invoke([
    HumanMessage(content = "What's my name? ")
],
config = config1)
print(response1)                          