import os 
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import streamlit as st
load_dotenv()

os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "True"
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")


prompt  = ChatPromptTemplate.from_messages(
    [
        ("system","You are helpful assistant.Please respond to the question asked"),
        ("user","Question:{question}")
    ]
)

# streamlit framework 
st.title("Langchain demo with gemma 2")
input_text = st.text_input("What question you have in mind?")
llm = ChatOllama(model="gemma:2b")
output_parser = StrOutputParser()
chain = prompt|llm|output_parser

if input_text:
    st.write(chain.invoke( {"question":input_text}))

