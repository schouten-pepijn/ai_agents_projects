import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv(".env")

@dataclass(frozen=True)
class Config:
    # OM = os.getenv("OM_API_KEY")  # Open Meteo
    TMK: str = os.getenv("TMK_API_KEY")  # Ticket Master
    GEOAPP_KEY: str = os.getenv("GEOAPP_API_KEY")  # Geoapify
    LLM_MODEL: str = (
        os.getenv("MODEL_LARGE")
        or os.getenv("gpt-3.5-turbo")
        or os.getenv("MODEL_SMALL")
    )
    BASE_URL: str = os.getenv("BASE_URL")  # Ollama base URL
    EMBED_MODEL: str = os.getenv("EMBED_MODEL") or "nomic-embed-text"
    TICKETMASTER_ENABLED: bool = False


@dataclass(frozen=True)
class Constants:
    OUTDOOR_HITS = [
        "park",
        "stadium",
        "festival",
        "open air",
        "outdoor",
        "beach",
        "market",
        "zoo",
        "garden",
    ]
    INDOOR_HINTS = [
        "theater",
        "cinema",
        "club",
        "museum",
        "hall",
        "arena",
        "indoor",
        "gallery",
        "exhibition",
    ]
    GEOAPIFY_CATEGORIES = ",".join([
        "entertainment.museum",
        "entertainment.zoo",
        "entertainment.theme_park",
        "entertainment.aquarium",
        "leisure.park",
        "leisure.garden",
        "tourism.sights",
        "commercial.shopping_mall"
    ])

config = Config()
constants = Constants()