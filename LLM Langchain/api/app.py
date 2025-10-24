from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langserve import add_routes # to interact with various llm models

import uvicorn
import os

from langchain_community.llms import Ollama # for third party LLMs :

from dotenv import load_dotenv

load_dotenv()



app = FastAPI(
    title="Chatbot API",
    decsription="A simple API server",
    version="1.0.0"
)


llm=Ollama(model="llama2")

prompt= ChatPromptTemplate.from_template("Write me an poem about {topic}")

add_routes(
    app,
    prompt|llm,
    path="/poem"
)


if __name__ == "__main__":
    uvicorn.run(app,host="localhost", port=8000)

