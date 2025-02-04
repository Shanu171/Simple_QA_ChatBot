import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser 
from langchain_core.prompts import ChatPromptTemplate
import os 
from dotenv import load_dotenv
load_dotenv()

## LAngsmith tracking 
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"]= "true"
os.environ["LANGCHAIN_PROJECT"] = "QA Chatbot with DEEPSEEK"


## Prompt Template 

prompt = ChatPromptTemplate.from_messages(
    [
        ("system","You are a helpful assistant. Please respond to the user queries"),
        ("user","question :{question}")

    ]
)


def generate_response(question,api_key,engine,temperature):
    llm = ChatGroq(model = engine ,groq_api_key = api_key,temperature = temperature)
    output_parser = StrOutputParser()
    chain = prompt|llm|output_parser 
    answer = chain.invoke({"question":question})
    return answer


## Title of the app 
st.title("Enhanced Q&A Chatbot with DeepSeek")

## Sidebar for settings 
st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter your Groq Api key",type="password")

## Dropdown to select various LLM models 
engine = st.sidebar.selectbox("Select an LLM model",["deepseek-r1-distill-llama-70b","gemma2-9b-it"])

## Adjust response paramter 
temperature = st.sidebar.slider("Temperature",  min_value=0.0,max_value=1.0,value=0.75)



## MAin interface for user input 
st.write("Go ahead  and ask any questions")
user_input = st.text_input("You:") 


if user_input:
    response = generate_response(user_input,api_key,engine,temperature)
    st.write(response)

else:
    st.write("Please the query")


