import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv

def vectorstore_embeddings():
    if "vector" not in st.session_state or st.session_state.vector is None:
        try:
            st.session_state.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )

            st.session_state.loader = PyPDFDirectoryLoader("./us_census")
            st.session_state.docs = st.session_state.loader.load()
            st.write(f"Loaded {len(st.session_state.docs)} documents")

            st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            st.session_state.final_docs = st.session_state.text_splitter.split_documents(st.session_state.docs)
            
            if st.session_state.final_docs:
                st.session_state.vector = FAISS.from_documents(st.session_state.final_docs, st.session_state.embeddings)
                st.success("Vector Store DB is ready!")
            else:
                st.warning("No documents found to create vector store!")
                st.session_state.vector = None
                
        except Exception as e:
            st.error(f"Error with embeddings: {str(e)}")

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

st.title('ChatGROQ PDF Chatbot - LangChain 1.0+')

llm = ChatGroq(api_key=groq_api_key, model="llama-3.1-8b-instant")

# New LCEL approach for LangChain 1.0+
prompt = ChatPromptTemplate.from_template(
    """Answer the question based on the context below. 
Please provide the accurate answer based on the question.

<context>
{context}   
</context>

Question: {input}"""
)


prompt1 = st.text_input("Enter your question from the docs: ")

if st.button("Document Embeddings"):
    vectorstore_embeddings()

if prompt1 and st.session_state.vector:
    try:
        # NEW LCEL APPROACH for LangChain 1.0+
        retriever = st.session_state.vector.as_retriever()
        
        # Create the new LCEL chain
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        rag_chain = (
            {"context": retriever | format_docs, "input": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        
        response = rag_chain.invoke(prompt1)
        st.write("### Answer:")
        st.write(response)
        
        # Show retrieved documents
        with st.expander("Document Similarity Search"):
            retrieved_docs = retriever.invoke(prompt1)
            for i, doc in enumerate(retrieved_docs):
                st.write(f"**Document {i+1}:**")
                st.write(doc.page_content)
                st.write("--------------------------------")
                
    except Exception as e:
        st.error(f"Error: {str(e)}")