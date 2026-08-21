from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langserve import add_routes
import os 
from dotenv import load_dotenv
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
model = ChatGroq(model="llama-3.1-8b-instant",groq_api_key = groq_api_key)

system_template = "Translate the following into {language}"
generic_template = "Translate the following into {language}"
prompt = ChatPromptTemplate.from_messages([
    ("system", generic_template),
    ("user", "{text}")
])
parser = StrOutputParser()
chain = prompt | model | parser
app = FastAPI(title = "Langchain server",
              version = "1.0",
              description = "A Simple API server using langchain runnable interfaces")

add_routes(app,
           chain,
           path = "/chain")
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app,host = "localhost",port = 8000)