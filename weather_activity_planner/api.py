from fastapi import FastAPI, HTTPException
from models import Query
from agents import build_graph
import traceback

app = FastAPI(title="Weather + Events Planner")
graph = build_graph()

@app.get("/")
async def root():
    return {"message": "Weather + Events Planner API", "endpoints": ["/plan (POST)"]}

@app.post("/plan")
async def plan(q: Query):
    try:
        state = {"query": q}
        result = await graph.ainvoke(state)
        
        return {
            "city": result["city"].model_dump(),
            "events": [e.model_dump() for e in result["events"]],
            "plan": [i.model_dump() for i in result["plan"].items],
            "narrative": result["narrative"],
        }
    except Exception as e:
        error_detail = f"Error: {str(e)}\n{traceback.format_exc()}"
        print(error_detail)
        
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/plan")
async def plan_get():
    raise HTTPException(
        status_code=405, 
        detail="Method not allowed. Use POST method with JSON payload. Visit /docs for API documentation."
    )