import os
from typing import TypedDict, List
from datetime import timezone
from langgraph.graph import StateGraph, END

from langchain_ollama import ChatOllama
from models import Query, City, WeatherSlot, Event, Plan
from tools import (
    geoapify_autocomplete_city, geoapify_forward_geocode,
    open_meteo_forecast, ticketmaster_events
)
from planner import build_plan
from config import config


class State(TypedDict):
    query: Query
    city: City
    weather: List[WeatherSlot]
    events: List[Event]
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
    
    tm = await ticketmaster_events(lat, lon, start_iso, end_iso)
                                
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

async def plan_node(state: State) -> State:
    
    state["plan"] = build_plan(state["city"], state["events"], state["weather"])
    
    return state


async def narrative_node(state: State) -> State:
    llm = get_llm()
    top = state["plan"].items[:6]
    id2 = {e.id: e for e in state["events"]}
    
    bullets = "\n".join([f"- {id2[i.event_id].name} ({i.reason})" for i in top])
    
    prompt = f"Write a concise weekend plan for {state['city'].query}. 4-6 items. One sentence each.\n{bullets}"
   
    out = await llm.ainvoke(prompt)
    
    state["narrative"] = out.content
    
    return state

def build_graph():
    g = StateGraph(State)
    g.add_node("city", city_node)
    g.add_node("weather", weather_node)
    g.add_node("events", events_node)
    g.add_node("plan", plan_node)
    g.add_node("narrative", narrative_node)
    
    g.set_entry_point("city")
    g.add_edge("city","weather")
    g.add_edge("weather","events")
    g.add_edge("events","plan")
    g.add_edge("plan","narrative")
    g.add_edge("narrative", END)
    return g.compile()