import streamlit as st
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

def get_conversational_agent():
    api_key = st.secrets["openai"]["api_key"]
    llm = ChatOpenAI(openai_api_key=api_key, temperature=0.7)
    memory = ConversationBufferMemory()
    agent = ConversationChain(llm=llm, memory=memory)
    return agent
