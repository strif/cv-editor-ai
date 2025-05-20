import streamlit as st
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

@st.cache_resource
def get_conversational_agent(model_name="gpt-4.1"):
    api_key = st.secrets["openai"]["api_key"]
    llm = ChatOpenAI(openai_api_key=api_key, model_name=model_name, temperature=0.7)
    memory = ConversationBufferMemory()
    agent = ConversationChain(llm=llm, memory=memory)
    return agent
