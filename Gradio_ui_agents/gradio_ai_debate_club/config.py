import os
from dotenv import load_dotenv
from langchain_ollama import ChatOllama

load_dotenv(".env")

MODEL = (
    os.getenv("MODEL_LARGE")
    or os.getenv("MODEL_MEDIUM")
    or os.getenv("MODEL_SMALL")
)

# Default temperatures (can be overridden)
TEMP_PRO = float(os.getenv("TEMP_PRO", 0.5))
TEMP_CON = float(os.getenv("TEMP_CON", 0.5))
TEMP_MOD = float(os.getenv("TEMP_MOD", 0.2))


def build_model(temp: float) -> ChatOllama:
    """
    Build a ChatOllama model instance with the specified temperature.

    Args:
        temp (float): The temperature setting for the model.

    Returns:
        ChatOllama: An instance of the ChatOllama model configured with the given temperature.
    """
    return ChatOllama(model=MODEL, temperature=temp)


llm_pro = build_model(TEMP_PRO)
llm_con = build_model(TEMP_CON)
llm_mod = build_model(TEMP_MOD)

# Streaming behaviour
CHUNK_DELAY = float(os.getenv("STREAM_DELAY", 0.06)) 
BATCH_CHARS = int(os.getenv("STREAM_BATCH_CHARS", 18))  

# Debate constraints
MIN_ROUNDS = 2
MAX_ROUNDS = 5
MAX_WORDS_PER_SIDE = 130
