from fastapi import FastAPI
from models import Query
from agents import build_graph

app = FastAPI(title="Weather + Events Planner")
graph = build_graph()

@app.post("/plan")
async def plan(q: Query):
    
    state = {"query": q}
    result = await graph.ainvoke(state)
    
    return {
        "city": result["city"].model_dump(),
        "events": [e.model_dump() for e in result["events"]],
        "plan": [i.model_dump() for i in result["plan"].items],
        "narrative": result["narrative"],
    }