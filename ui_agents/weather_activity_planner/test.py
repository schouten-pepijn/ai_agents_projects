from models import Query
from agents import build_graph
import asyncio
from datetime import datetime, timezone, timedelta


def test_plan():
    graph = build_graph()
    
    city = "Rotterdam"
    date_from = datetime.now(timezone.utc)
    date_to = datetime.now(timezone.utc) + timedelta(days=3)

    q = Query(city_query=city, date_from=date_from, date_to=date_to)
    state = {"query": q}
    
    result =  asyncio.run(graph.ainvoke(state))

    assert "city" in result
    assert "events" in result
    assert "plan" in result
    assert "narrative" in result
    
    assert result["city"].query == "Rotterdam"
    
    assert len(result["events"]) > 0
    assert len(result["plan"].items) > 0
    assert isinstance(result["narrative"], str)
    
    print("Narrative:", result["narrative"])
    
    print("Test passed: plan function works as expected.")


if __name__ == "__main__":
    test_plan()