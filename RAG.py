#Use google-genai text embedding model
from google import genai
import os

client = genai.Client(api_key = os.getenv("GEMINI_API_KEY"))

client = client.models.embed_content(
    models = "gemini-embedding-001"
    content = content
)

