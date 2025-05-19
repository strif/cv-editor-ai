from langchain_community.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

def get_conversational_agent():
    llm = ChatOpenAI(temperature=0.7)
    memory = ConversationBufferMemory()
    agent = ConversationChain(llm=llm, memory=memory)
    return agent
