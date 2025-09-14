from tools.market import fetch_bars


def test_fetch_bars():
    df = fetch_bars("AAPL", period="1mo", interval="1d")
    assert not df.empty, "DataFrame should not be empty"
    assert set(["date", "open", "high", "low", "close", "volume"]).issubset(df.columns)
    print(df.head())


if __name__ == "__main__":
    test_fetch_bars()
