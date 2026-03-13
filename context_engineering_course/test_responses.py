"""
Course starter: Responses API smoke test.

Purpose:
- Makes one simple chat-style request and prints the model response text.

Usage:
- Ensure `.streamlit/secrets.toml` is configured.
- Run: `python context_engineering_course/test_responses.py`
"""

from openai import OpenAI

from settings import (
    GPT_BASE,
    GPT_KEY,
    GPT_CHAT_MODEL,
    GPT_SUPPORTS_REASONING,
    GPT_AVAILABLE_REASONING_EFFORTS,
    GPT_SUPPORTS_TEMPERATURE,
)

client = OpenAI(
    api_key=GPT_KEY,
    base_url=GPT_BASE,
)

reasoning_effort = "minimal"
temperature = 1
# Optional: Add tools for responses endpoint (leave None for no tools)
tools = None
# tools = [{"type": "web_search_preview"}] # Web search requires at least minimal reasoning effort
# tools = [{"type": "code_interpreter"}]
tools_string = ", ".join([tool["type"] for tool in tools]) if tools else "no tools"

if GPT_SUPPORTS_REASONING:
    reasoning_effort = reasoning_effort if reasoning_effort in GPT_AVAILABLE_REASONING_EFFORTS else GPT_AVAILABLE_REASONING_EFFORTS[0]
    print(f"---Testing response with {GPT_CHAT_MODEL} and {reasoning_effort} reasoning effort{' with ' + tools_string if tools else ''} ---\n\n")
    response = client.responses.create(
        model=GPT_CHAT_MODEL,
        input="What is context engineering?",
        reasoning={"effort": reasoning_effort},
        tools=tools,
    )
elif GPT_SUPPORTS_TEMPERATURE:
    print(f"---Testing response with {GPT_CHAT_MODEL} and temperature {temperature}{' with ' + tools_string if tools else ''} ---\n\n")
    response = client.responses.create(
        model=GPT_CHAT_MODEL,
        input="What is context engineering?",
        temperature=temperature,
        tools=tools,
    )
else:
    print(f"---Testing response with {GPT_CHAT_MODEL} with no set reasoning effort or temperature{' with ' + tools_string if tools else ''} ---\n\n")
    response = client.responses.create(
        model=GPT_CHAT_MODEL,
        input="What is context engineering?",
        tools=tools,
    )

print(response.output_text)
