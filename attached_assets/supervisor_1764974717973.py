from langchain.agents import create_agent
\
from agents.model import llm

agent = create_agent(
    llm,
    tools=[]
)