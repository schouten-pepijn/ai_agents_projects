from ddgs import DDGS


def fetch_news(query: str, max_hits: int = 8):
    with DDGS() as search:
        return list(
            search.news(
                query=query, region="wt-wt", safesearch="off", max_results=max_hits
            )
        )
