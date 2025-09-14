from graph import build_graph
from config import settings
import json

if __name__ == "__main__":
    print("Building graph...")
    app = build_graph()

    symbol = "AAPL"
    print(f"Using symbol: {symbol}")

    state = {
        "symbol": symbol,
        "news": [],
        "ta_df_json": "",
        "sentiment": {},
        "fuse_json": "",
    }

    print("Invoking graph...")
    state = app.invoke(state)

    print("Graph execution completed.")
    print(f"Final state keys: {list(state.keys())}")
    print(f"News count: {len(state.get('news', []))}")
    print(f"Sentiment: {state.get('sentiment', {})}")
    print(f"TA data: {state.get('ta_df_json', '{}')[:100]}...")
    print(f"Fuse JSON: {state.get('fuse_json', '{}')}")

    try:
        result = json.loads(state["fuse_json"])
        print("\nParsed JSON result:")
        print(json.dumps(result, indent=2))
    except Exception as e:
        print(f"Error parsing fuse_json: {e}")
        print(f"Raw fuse_json: {state['fuse_json']}")
