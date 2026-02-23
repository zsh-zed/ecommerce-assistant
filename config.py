import os

from dotenv import load_dotenv

load_dotenv()

# DEEPSEEK_API_KEY: str = os.getenv("DEEPSEEK_API_KEY") or ""
GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY") or ""
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY") or ""

MODEL_NAME = "groq:llama-3.1-8b-instant"

EMBEDDING_MODEL = "models/gemini-embedding-001"

VECTORSTORE_PATH = "vectorstore"
