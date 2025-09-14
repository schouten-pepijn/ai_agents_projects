from tools.market import fetch_bars
from tools.news import fetch_news
from tools.sentiment import sentiment_vader
from tools.backtest import run_backtest
import pandas as pd
from tools.ta_tools import compute_ta


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


def test_compute_ta():
    data = {
        "close": [
            100,
            102,
            101,
            105,
            107,
            110,
            108,
            109,
            111,
            112,
            113,
            115,
            117,
            116,
            118,
            120,
            119,
            121,
            123,
            122,
            124,
            126,
            125,
            127,
            129,
            128,
            130,
            132,
            131,
            133,
            135,
            134,
            136,
            138,
            137,
            139,
            141,
            140,
            142,
            144,
            143,
            145,
            147,
            146,
            148,
            150,
            149,
            151,
            153,
            152,
        ],
        "high": [
            101,
            103,
            102,
            106,
            108,
            111,
            109,
            110,
            112,
            113,
            114,
            116,
            118,
            117,
            119,
            121,
            120,
            122,
            124,
            123,
            125,
            127,
            126,
            128,
            130,
            129,
            131,
            133,
            132,
            134,
            136,
            135,
            137,
            139,
            138,
            140,
            142,
            141,
            143,
            145,
            144,
            146,
            148,
            147,
            149,
            151,
            150,
            152,
            154,
            153,
        ],
        "low": [
            99,
            101,
            100,
            104,
            106,
            109,
            107,
            108,
            110,
            111,
            112,
            114,
            116,
            115,
            117,
            119,
            118,
            120,
            122,
            121,
            123,
            125,
            124,
            126,
            128,
            127,
            129,
            131,
            130,
            132,
            134,
            133,
            135,
            137,
            136,
            138,
            140,
            139,
            141,
            143,
            142,
            144,
            146,
            145,
            147,
            149,
            148,
            150,
            152,
            151,
        ],
    }

    df = pd.DataFrame(data)
    result = compute_ta(df)

    assert "rsi14" in result.columns
    assert "ema50" in result.columns
    assert "ema200" in result.columns
    assert "atr14" in result.columns
    assert "trend" in result.columns

    print(result[["close", "rsi14", "ema50", "ema200", "atr14", "trend"]].tail())


def test_sentiment_vader():
    texts = [
        "The stock price soared after the company reported record profits.",
        "Investors are worried about the recent decline in sales.",
        "The outlook for the next quarter is neutral.",
    ]
    result = sentiment_vader(texts)

    assert isinstance(result, dict)
    assert all(k in result for k in ["neg", "neu", "pos", "compound"])

    print(result)


def test_run_backtest():
    data = {
        "date": pd.date_range("2024-01-01", periods=250, freq="B"),
        "open": [100 + i * 0.1 for i in range(250)],
        "high": [101 + i * 0.1 for i in range(250)],
        "low": [99 + i * 0.1 for i in range(250)],
        "close": [100 + i * 0.1 + (i % 5 - 2) * 0.5 for i in range(250)],
        "volume": [1000000 for _ in range(250)],
    }
    df = pd.DataFrame(data)
    result = run_backtest(df)

    assert isinstance(result, dict)
    assert "Equity Final [$]" in result

    print({k: result[k] for k in list(result)[:5]})


if __name__ == "__main__":
    test_fetch_news()
    test_fetch_bars()
    test_compute_ta()
    test_sentiment_vader()
    test_run_backtest()
