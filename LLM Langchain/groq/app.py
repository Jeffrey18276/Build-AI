import streamlit as st
import os 
from langchain_groq import ChatGroq

from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import OllamaEmbeddings

 
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
import  time

from dotenv import load_dotenv

load_dotenv()


groq_api_key = os.getenv("GROQ_API_KEY")


if "vector" not in st.session_state:
    st.session_state.embeddings =OllamaEmbeddings()
    st.session_state.loader = WebBaseLoader(web_path='https://lilianweng.github.io/posts/2024-11-28-reward-hacking/')
    st.session_state.docs = st.session_state.loader.load()

    st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    st.session_state.final_docs = st.session_state.text_splitter.split_documents(st.session_state.docs)
    st.session_state.vector = FAISS.from_documents(st.session_state.final_docs, st.session_state.embeddings)


st.title('ChatGROQ Chatbot')
llm = ChatGroq(api_key=groq_api_key, model="allam-2-7b", temperature=0.7)

prompt = ChatPromptTemplate.from_template(
    """
    Answer the question based on the context below. If the question can't be answered using the context, say "I don't know".
    Please provide the accurate answer based on the question.
    <context>
    {context}
    </context>
    Question:{input}
"""
)

document_chain = create_stuff_documents_chain(llm=llm, prompt = prompt)
retriever =st.session_state.vector.as_retriever() 
retrieval_chain =create_retrieval_chain(retriever=retriever, combine_docs_chain=document_chain)

prompt = st.text_input("Input your prompt here.")

if prompt:
    starter_time = time.time()
    response = retrieval_chain.invoke({"input":prompt})   
    print("resposne_time", time.time() - starter_time)
    st.write(response['answer'])