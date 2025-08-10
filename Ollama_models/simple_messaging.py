import requests
import json
import dotenv
import os

dotenv.load_dotenv(".env")

OLLAMA_URL = os.getenv("OLLAMA_URL")
MODEL = os.getenv("MODEL")

# send a simple message
messages = [
    {"role": "system", "content": "You are a concise AI assistant."},
    {"role": "user", "content": "What is the capital of France?"}
]

response = requests.post(
    f"{OLLAMA_URL}/api/chat",
    json={
        "model": MODEL,
        "messages": messages,
        "stream": False,
        "options": {"temperature": 0.2, "num_ctx": 4096}
    },
    timeout=300
)

response.raise_for_status()
print(response.json()["message"]["content"])

# send a simple prompt
prompt = "List five SQL anti-patterns in one sentence each."

response = requests.post(
    f"{OLLAMA_URL}/api/generate",
    json={
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.2, "num_ctx": 4096}
    },
    timeout=300
)

response.raise_for_status()
print(response.json()["response"])

# simple streaming mode
with requests.post(
    f"{OLLAMA_URL}/api/generate",
    json={
        "model": MODEL,
        "prompt": prompt,
        "stream": True,
        "options": {"temperature": 0.2, "num_ctx": 4096}
    },
    timeout=300
) as r:
    r.raise_for_status()
    for line in r.iter_lines():
        if line:
            data = json.loads(line.decode('utf-8'))
            if "response" in data:
                print(data["response"], end="", flush=True)


# prompt function
def send_prompt(model: str, prompt: str, temperature: float = 0.2, ctx: int = 4096) -> str:
    """Send a prompt to Ollama and return the model's response."""
    r = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature, "num_ctx": ctx}
        },
        timeout=300
    )
    r.raise_for_status()
    return r.json()["response"]


send_prompt(MODEL, prompt)