"""
Course starter: embedding API smoke test.

Purpose:
- Sends one text string to the configured embedding model and prints basic output.

Usage:
- Ensure `.streamlit/secrets.toml` is configured.
- Run: `python context_engineering_course/test_embeddings.py`
"""

from openai import OpenAI
from settings import GPT_EMBEDDINGS_MODEL, GPT_BASE, GPT_KEY, USE_GEMINI, GEMINI_API_KEY, GEMINI_EMBEDDING_MODEL

# Text to embed
text = "Hi, how are you doing today? I hope you're having a great day! This is a test of the embedding function. Let's see how it works."

# Clean the text
text = text.replace("\n", " ")

# Generate embedding
if USE_GEMINI:
    import google.generativeai as genai
    genai.configure(api_key=GEMINI_API_KEY)
    result = genai.embed_content(
        model=GEMINI_EMBEDDING_MODEL,
        content=text,
        task_type="retrieval_document"
    )
    embedding = result["embedding"]
else:
    client = OpenAI(api_key=GPT_KEY, base_url=GPT_BASE)
    result = client.embeddings.create(input=[text], model=GPT_EMBEDDINGS_MODEL)
    embedding = result.data[0].embedding

# Print results
print(f"Text: {text}")
print(f"Embedding dimension: {len(embedding)}")
print(f"First 10 values: {embedding[:10]}")
