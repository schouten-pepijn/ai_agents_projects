from tools.market import fetch_bars
from tools.news import fetch_news


def test_fetch_bars():
    df = fetch_bars("AAPL", period="1mo", interval="1d")
    assert not df.empty, "DataFrame should not be empty"
    assert set(["date", "open", "high", "low", "close", "volume"]).issubset(df.columns)
    print(df.head())


def test_fetch_news():
    results = fetch_news("AAPL", max_hits=3)
    assert isinstance(results, list)
    assert len(results) > 0
    for item in results:
        assert "title" in item or "body" in item or "content" in item
    print(results)


if __name__ == "__main__":
    test_fetch_news()
    test_fetch_bars()
