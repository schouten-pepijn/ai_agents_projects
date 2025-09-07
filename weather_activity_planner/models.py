from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime


class City(BaseModel):
    query: str
    lat: float
    lon: float
    country: Optional[str] = None
    label: Optional[str] = None
    
    
class WeatherSlot(BaseModel):
    dt: datetime
    temp: float
    precip_mm: float
    wind_ms: float
    cloud_pct: int = 0.
    pop: float = 0.  # Probability of precipitation
    description: str = "n/a"
    
    
class Event(BaseModel):
    id: str
    source: str
    name: str
    description: Optional[str]
    category: Optional[str] 
    start: Optional[datetime]
    end: Optional[datetime]
    venue: Optional[str]
    indoor: Optional[bool]
    lat: Optional[float] 
    lon: Optional[float] 
    url: Optional[str]


class Place(BaseModel):
    id: str 
    name: str
    category: str
    lat: float
    lon: float
    address: Optional[str] = None
    url: Optional[str] = None

class PlanItem(BaseModel):
    event_id: str
    suitability: float
    reason: str
    slot_dt: datetime


class Plan(BaseModel):
    city: City
    items: List[PlanItem] = Field(default_factory=list)
    

class Query(BaseModel):
    city_query: str
    date_from: datetime
    date_to: datetime
    preferences: Optional[dict] = None
