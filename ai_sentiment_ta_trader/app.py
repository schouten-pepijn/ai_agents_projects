from graph import build_graph
from config import settings
import json
import logging

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)

if __name__ == "__main__":
    logging.debug("Building graph...")
    app = build_graph()

    symbol = "BTC-USD"
    logging.debug(f"Using symbol: {symbol}")

    state = {
        "symbol": symbol,
        "news": [],
        "ta_df_json": "",
        "sentiment": {},
        "fuse_json": "",
    }

    logging.debug("Invoking graph...")
    state = app.invoke(state)

    logging.debug("Graph execution completed.")
    logging.debug(f"Final state keys: {list(state.keys())}")
    logging.debug(f"News count: {len(state.get('news', []))}")
    logging.debug(f"Sentiment: {state.get('sentiment', {})}")
    logging.debug(f"TA data: {state.get('ta_df_json', '{}')[:100]}...")
    logging.debug(f"Fuse JSON: {state.get('fuse_json', '{}')}")

    try:
        result = json.loads(state["fuse_json"])
        logging.debug("\nParsed JSON result:")

        print(json.dumps(result, indent=2))

    except Exception as e:
        logging.debug(f"Error parsing fuse_json: {e}")
        logging.debug(f"Raw fuse_json: {state['fuse_json']}")

        raise e
