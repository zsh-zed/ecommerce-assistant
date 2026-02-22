import os

from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY") or ""

MODEL_NAME = "google_genai:gemini-2.5-flash"

EMBEDDING_MODEL = "models/gemini-embedding-001"

VECTORSTORE_PATH = "vectorstore"
