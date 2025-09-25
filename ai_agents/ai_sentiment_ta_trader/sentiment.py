from typing import List, Dict

import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

try:
    nltk.data.find("sentiment/vader_lexicon.zip")
except LookupError:
    nltk.download("vader_lexicon")

_vader = SentimentIntensityAnalyzer()


def sentiment_vader(texts: List[str]) -> Dict[str, float]:
    if not texts:
        return {"neg": 0.0, "neu": 0.0, "pos": 0.0, "compound": 0.0}

    scores = [_vader.polarity_scores(text) for text in texts]
    agg = {k: sum(s[k] for s in scores) / len(scores) for k in scores[0].keys()}

    return agg
