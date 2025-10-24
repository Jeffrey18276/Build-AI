from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables import RunnableParallel
from langchain_core.output_parsers import StrOutputParser
import os

load_dotenv()

def analyze_pros(features):
    pros_template= ChatPromptTemplate.from_messages([ 
        ('system',"You are expert in product reviewer."),
        ('user',"Given these features: {features}, list the pros of these features.")
         ])

    return pros_template | model | StrOutputParser()

def analyze_cons(features):
    cons_template =ChatPromptTemplate.from_messages([ 
        ('system',"You are expert in product reviewer."),
        ('user',"Given these features: {features}, list the cons of these features.")
         ])

    return cons_template | model | StrOutputParser()

def combine_pros_cons(pros,cons):
    return f"Pros:\n{pros} Cons:\n{cons}"



prompt_template= ChatPromptTemplate.from_messages([ 
        ('system',"You are expert in product reviewer."),
        ('user',"List the main features of the product {product_name}.")
         ])

groq_api_key = os.getenv("GROQ_API_KEY")

model = ChatGroq(api_key=groq_api_key, model="llama-3.1-8b-instant")

action_prompt = RunnableLambda(lambda x:prompt_template.format_prompt(**x).to_messages() )

pros_branch_chain =(
    RunnableLambda(lambda x:analyze_pros(x)) | model | StrOutputParser()
)


cons_branch_chain= (
    RunnableLambda(lambda x:analyze_cons(x)  | model | StrOutputParser())
)


chain = (action_prompt | model |StrOutputParser()| 
RunnableParallel({'pros':pros_branch_chain,'cons':cons_branch_chain}) |
RunnableLambda(lambda x: combine_pros_cons(x['pros'],x['cons']))
)

res=chain.invoke({"product_name":'Samsung S25 Ultra'})
print(res)



