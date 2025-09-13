def summarize(symbol: str, r_rule: dict, r_ml: dict) -> dict:
    return {
        "symbol": symbol,
        "rule": r_rule["stats"],
        "ml": r_ml["stats"],
    }
