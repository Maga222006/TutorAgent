from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
load_dotenv()
llm = init_chat_model("groq:meta-llama/llama-4-maverick-17b-128e-instruct")
