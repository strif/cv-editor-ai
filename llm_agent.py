from langchain.agents import initialize_agent, Tool
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from settings import MODEL_NAME, TEMPERATURE

def dummy_tool_func(input_data):
    return "This is a placeholder for future tools."

def get_conversational_agent():
    llm = ChatOpenAI(model=MODEL_NAME, temperature=TEMPERATURE)
    memory = ConversationBufferMemory(memory_key="chat_history")

    tools = [
        Tool(name="DummyTool", func=dummy_tool_func, description="Placeholder")
    ]

    return initialize_agent(tools, llm, agent="chat-conversational-react-description", memory=memory, verbose=True)
