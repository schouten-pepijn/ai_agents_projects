from models import Event, WeatherSlot, Plan, PlanItem, City
from config import constants
from typing import List, Optional
from datetime import datetime, timezone


def infer_indoor(event: Event):
    text = " ".join(
        filter(
            None,
            [
                event.name,
                event.description or "",
                event.venue or "",
                event.category or "",
            ]
        )
    ).lower()
    
    if any(w in text for w in constants.OUTDOOR_HITS):
        return False
    
    if any(w in text for w in constants.INDOOR_HINTS):
        return True

    return True

def nearest_weather(slot_df: datetime, wx: List[WeatherSlot]) -> WeatherSlot:
    if slot_df.tzinfo is None:
        slot_df = slot_df.replace(tzinfo=timezone.utc)
    
    return min(
        wx,
        key=lambda w: abs((
            w.dt.replace(tzinfo=timezone.utc) if w.dt.tzinfo is None else w.dt
        ).astimezone(timezone.utc) - slot_df.astimezone(timezone.utc)).total_seconds())
    

def outdoor_suitability(w: WeatherSlot) -> float:
    temp_score = max(0, 1 - (abs(w.temp - 21) / 15))
    rain_score = 1 - min(1, (w.precip_mm / 3) + w.pop)
    wind_score = 1 - min(1, w.wind_ms / 12)
    clouds_penalty = max(0, (w.cloud_pct - 80) / 20 * 0.2)

    return max(
        0.0,
        min(
            1.0,
            0.5 * temp_score + 0.35 * rain_score + 0.15 * wind_score - clouds_penalty
        )
    )

def indoor_suitability(w: WeatherSlot) -> float:
    comfort = max(0, 1 - (abs(w.temp - 22) / 20))
    bad_weather = min(1, (w.precip_mm / 2) + w.pop + (w.wind_ms / 15))
    
    return max(
        0.0,
        min(1.0, 0.4 * comfort + 0.6 * bad_weather))
    
def build_plan(city: City, events: List[Event], wx: List[WeatherSlot]) -> Plan:
    items = []
    
    for e in events:
        if not e.start:
            continue
       
        ind = e.indoor if e.indoor is not None else infer_indoor(e)
        w = nearest_weather(e.start.astimezone(timezone.utc), wx)
        s = indoor_suitability(w) if ind else outdoor_suitability(w)

        reason = f"{'Indoor' if ind else 'Outdoor'} · clouds {w.cloud_pct}%, {round(w.temp)}°C, precip {w.precip_mm:.1f}mm"

        items.append(
            PlanItem(
                event_id=e.id,
                suitability=s,
                reason=reason,
                slot_dt=w.dt
            )
        ) 

    items.sort(key=lambda x: (-x.suitability, x.slot_dt))
    
    return Plan(city=city, items=items)