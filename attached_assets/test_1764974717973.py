from langchain.chat_models import init_chat_model

from agents.tools_and_utils import Docs
from dotenv import load_dotenv
load_dotenv()

llm = init_chat_model("google_genai:gemini-flash-lite-latest")

"""docs = Docs("documents/Lecture3.pdf")

chunks = docs.get_diverse_chunks_mmr("Give an overview of the main ideas in this document")
print(len(chunks))   # up to 50

for chunk in chunks:
    print(chunk)"""