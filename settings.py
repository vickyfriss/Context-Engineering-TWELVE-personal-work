import streamlit as st

GPT_BASE = st.secrets.get("GPT_BASE")
GPT_VERSION = st.secrets.get("GPT_VERSION")
GPT_KEY = st.secrets.get("GPT_KEY")
GPT_CHAT_MODEL = st.secrets.get("GPT_CHAT_MODEL")
GPT_EMBEDDINGS_MODEL = st.secrets.get("GPT_EMBEDDINGS_MODEL")


if "gpt-5-mini" in GPT_CHAT_MODEL:
	GPT_SUPPORTS_REASONING = True
	GPT_AVAILABLE_REASONING_EFFORTS = ["minimal", "low", "medium", "high"]
	GPT_SUPPORTS_TEMPERATURE = False
elif "gpt-5-nano" in GPT_CHAT_MODEL:
	GPT_SUPPORTS_REASONING = True
	GPT_AVAILABLE_REASONING_EFFORTS = ["minimal", "low", "medium", "high"]
	GPT_SUPPORTS_TEMPERATURE = False
elif "gpt-4o-mini" in GPT_CHAT_MODEL:
	GPT_SUPPORTS_REASONING = False
	GPT_AVAILABLE_REASONING_EFFORTS = []
	GPT_SUPPORTS_TEMPERATURE = True
else:
	GPT_SUPPORTS_REASONING = False
	GPT_AVAILABLE_REASONING_EFFORTS = []
	GPT_SUPPORTS_TEMPERATURE = True

# Gemini secrets
USE_GEMINI = st.secrets.get("USE_GEMINI", False)
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "")
GEMINI_CHAT_MODEL = st.secrets.get("GEMINI_CHAT_MODEL", "")
GEMINI_EMBEDDING_MODEL = st.secrets.get("GEMINI_EMBEDDING_MODEL", "")

#Local LM Studio secrets
USE_LM_STUDIO = st.secrets.get("USE_LM_STUDIO", False)
LM_STUDIO_API_KEY = st.secrets.get("LM_STUDIO_API_KEY", "")
LM_STUDIO_API_BASE = st.secrets.get("LM_STUDIO_API_BASE", "")
LM_STUDIO_CHAT_MODEL = st.secrets.get("LM_STUDIO_CHAT_MODEL", "")
LM_STUDIO_EMBEDDING_MODEL = st.secrets.get("LM_STUDIO_EMBEDDING_MODEL", "")

