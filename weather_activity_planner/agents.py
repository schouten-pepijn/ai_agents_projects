import os
from typing import TypedDict, List
from datetime import timezone
from langgraph.graph import StateGraph, END

from langchain_ollama import ChatOllama
from models import Query, City, WeatherSlot, Event, Plan, Place
from tools import (
    geoapify_autocomplete_city, geoapify_forward_geocode,
    open_meteo_forecast, ticketmaster_events,
    geoapify_places, parse_geoapify_place
)
from planner import build_plan, append_pois_when_sparse
from config import config, constants


class State(TypedDict):
    query: Query
    city: City
    weather: List[WeatherSlot]
    events: List[Event]
    places: List[Place]
    plan: Plan
    narrative: str
    

def get_llm():
    return ChatOllama(
        model=config.LLM_MODEL,
        base_url=config.BASE_URL,
        temperature=0.7,
        max_retries=3
    )
    
async def city_node(state: State) -> State:
    q = state["query"]
    feats = await geoapify_autocomplete_city(q.city_query) if os.getenv("GEOAPIFY_KEY") else []
    
    if feats:
        lon, lat = feats[0]["geometry"]["coordinates"]
        props = feats[0]["properties"]
        state["city"] = City(query=q.city_query, lat=float(lat), lon=float(lon), country=props.get("country_code"), label=props.get("formatted"))
    
    else:
        state["city"] = await geoapify_forward_geocode(q.city_query)
    
    return state


async def weather_node(state: State) -> State:
    c = state["city"]
    state["weather"] = await open_meteo_forecast(c.lat, c.lon)
    
    return state

async def events_node(state: State) -> State:
    q = state["query"]
    start_iso = q.date_from.astimezone(timezone.utc).isoformat().replace("+00:00","Z")
    end_iso   = q.date_to.astimezone(timezone.utc).isoformat().replace("+00:00","Z")
    lat, lon = state["city"].lat, state["city"].lon
    
 
    preferences = q.preferences or {}
    radius_km = preferences.get("radius_km", 10)
    activity_types = preferences.get("activity_types", [])
    
    tm = await ticketmaster_events(lat, lon, start_iso, end_iso, radius_km=radius_km)
    
    if activity_types:
        filtered_events = []
        for e in tm:
            # Simple keyword matching for activity types
            event_text = f"{e.name} {e.description or ''} {e.category or ''}".lower()
            
            for activity in activity_types:
                if activity.lower() in event_text or any(keyword in event_text for keyword in _get_activity_keywords(activity)):
                    filtered_events.append(e)
                    break
                
        tm = filtered_events
                                
    seen, merged = set(), []
    
    for e in tm:
        if not e.start:
            continue
        
        key=(e.name.lower(), e.start)
        
        if key in seen:
            continue
        
        seen.add(key)
        merged.append(e)
        
    state["events"] = merged
    
    return state


def _get_activity_keywords(activity_type: str) -> list:
    keywords_map = {
        "outdoor": ["park", "festival", "outdoor", "beach", "garden", "hiking", "cycling"],
        "indoor": ["theater", "cinema", "museum", "gallery", "club", "hall"],
        "cultural": ["museum", "art", "theater", "opera", "cultural", "heritage", "history"],
        "sports": ["stadium", "sports", "football", "basketball", "tennis", "athletic"],
        "music": ["concert", "music", "band", "singer", "orchestra", "festival"],
        "food & drink": ["restaurant", "food", "wine", "beer", "dining", "culinary"],
        "family": ["family", "children", "kids", "playground", "zoo", "aquarium"],
        "nightlife": ["bar", "club", "nightlife", "party", "dance", "pub"]
    }
    return keywords_map.get(activity_type.lower(), [])


async def places_node(state: State) -> State:
    c = state["city"]
    q = state["query"]
    
    preferences = q.preferences or {}
    # Limit radius to reasonable bounds for Geoapify API
    radius_km = min(preferences.get("radius_km", 10), 20)
    
    # Use selected place categories from UI, or default to predefined set
    selected_categories = preferences.get("place_categories", [])
    if selected_categories:
        categories_str = ",".join(selected_categories)
    else:
        categories_str = constants.GEOAPIFY_CATEGORIES

    feats = await geoapify_places(c.lat, c.lon, categories_str, radius_km=radius_km)
    state["places"] = [parse_geoapify_place(f) for f in feats]
    return state


async def plan_node(state: State) -> State:
    
    state["plan"] = build_plan(state["city"], state["events"], state["weather"])
    append_pois_when_sparse(state["plan"], state["places"], state["weather"], k=6)
    
    return state


async def narrative_node(state: State) -> State:
    llm = get_llm()

    id2event = {e.id: e for e in state.get("events", [])}
    id2poi   = {f"poi::{p.id}": p for p in state.get("places", [])}

    # Helper to resolve a human-readable label for either event or poi
    def _label(item) -> str:
        if item.event_id in id2event:
            return id2event[item.event_id].name or "Event"
        if item.event_id in id2poi:
            return id2poi[item.event_id].name or "Place"
        return "Suggestion"

    top = state["plan"].items[:8]

    bullets = "\n".join([f"- {_label(i)} ({i.reason})" for i in top])

    q = state["query"]
    preferences = (q.preferences or {})
    activity_types = preferences.get("activity_types", [])
    additional_prefs = preferences.get("additional_preferences", "")

    prefs_chunks = []
    if activity_types:
        prefs_chunks.append(f"User prefers: {', '.join(activity_types)}.")
        
    if additional_prefs:
        prefs_chunks.append(f"Additional preferences: {additional_prefs}.")
        
    preference_context = " ".join(prefs_chunks)

    prompt = f"""Write a concise activity plan for {state['city'].query}. 6-8 items. One sentence each.

{preference_context}

Recommended activities:
{bullets}

Create a narrative that considers the user's preferences and explains why these activities are suitable."""
   
    out = await llm.ainvoke(prompt)
    state["narrative"] = out.content
    return state


def build_graph():
    g = StateGraph(State)
    g.add_node("city", city_node)
    g.add_node("weather", weather_node)
    g.add_node("events", events_node)
    g.add_node("places", places_node)
    g.add_node("plan", plan_node)
    g.add_node("narrative", narrative_node)
    
    g.set_entry_point("city")
    g.add_edge("city","weather")
    g.add_edge("weather","events")
    g.add_edge("events","places")
    g.add_edge("places","plan")
    g.add_edge("plan","narrative")
    g.add_edge("narrative", END)
    return g.compile()