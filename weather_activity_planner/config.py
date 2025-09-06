import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv(".env")

@dataclass(frozen=True)
class Config:
    # OM = os.getenv("OM_API_KEY")  # Open Meteo
    TMK = os.getenv("TMK_API_KEY")  # Ticket Master
    GEOAPP_KEY = os.getenv("GEOAPP_API_KEY")  # Geoapify

config = Config()