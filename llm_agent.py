import streamlit as st
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

@st.cache_resource
def get_conversational_agent(model_name="gpt-4o"):
    api_key = st.secrets["openai"]["api_key"]
    
    # ✅ Pass the model_name to ChatOpenAI
    llm = ChatOpenAI(
        openai_api_key=api_key,
        temperature=0.7,
        model_name=model_name  # <-- this is what was missing
    )
    
    memory = ConversationBufferMemory()
    agent = ConversationChain(llm=llm, memory=memory)
    return agent
