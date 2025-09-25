from graphstate import GraphState
from helpers import _get_user_question, _compact_sources_for_llm
from langchain_core.prompts import ChatPromptTemplate
import math
from typing import List, Dict
from config import CONFIG, llm, tavily
from ddgs import DDGS


def generate_queries(state: GraphState) -> GraphState:
    """
    Generate web search queries based on the user's question.
    This function takes the current state of the graph, extracts the user's question,
    and generates a set of concise web search queries to answer the question. The queries
    are designed to consider different perspectives and adhere to specific formatting rules.
    Args:
        state (GraphState): The current state of the graph, containing information such as
            the user's question and the maximum number of results per query.
    Returns:
        GraphState: The updated state of the graph, including the generated queries and
            an appended action log.
    Rules for query generation:
        - Generate between 2 and 6 queries, depending on the `max_results_per_query` value.
        - Each query should be concise, with a maximum of 8 words.
        - Queries should avoid using quotes and be written in English.
        - Duplicate queries are removed.
    Notes:
        - The function uses a chat prompt template and an LLM (Language Model) to generate
          the queries.
        - If no queries are generated, the user's original question is used as a fallback.
    """
    user_q = _get_user_question(state)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Generate {k} web search queries to answer the users question.\n"
                "Rules:\n- Consider different perspectives\n- Be concise, max 8 words per query\n"
                "-Do not use quotes\n- Answer in English.",
            ),
            ("user", "{question}"),
        ]
    )

    chain = prompt | llm

    k = max(2, min(6, math.ceil(state["max_results_per_query"] / 2)))
    response = chain.invoke({"k": k, "question": user_q})
    raw = response.content.strip().split("\n")

    queries = []
    for line in raw:
        q = line.strip("-• 1234567890. ").strip()
        if q:
            queries.append(q)
    queries = list(dict.fromkeys(queries))

    state["queries"] = queries or [user_q]
    state["actions"].append(f"generate_queries({len(state['queries'])})")

    return state


def web_search(state: GraphState) -> GraphState:
    """
    Perform a web search based on the queries provided in the state and update the state with the search results.
    Args:
        state (GraphState): The current state containing queries, configuration, and other relevant data.
    Returns:
        GraphState: The updated state with search results added to the "hits" key and an action log appended.
    The function performs the following:
    - Iterates over the queries in the state.
    - Depending on the configuration (`CONFIG.use_tavily`), it uses either the Tavily search API or DuckDuckGo Search (DDGS) to fetch results.
    - Each result is processed and appended to the "hits" list in the state.
    - Updates the "actions" list in the state with a log of the number of results fetched.
    Notes:
    - Tavily search is used if `CONFIG.use_tavily` is set to "true" (case-insensitive).
    - DuckDuckGo Search (DDGS) is used as a fallback if Tavily is not enabled.
    - Results include the query, title, URL, and content/snippet.
    """
    results: List[Dict] = state.get("hits", [])
    limit = state["max_results_per_query"]

    for q in state["queries"]:
        if bool(CONFIG.use_tavily) and CONFIG.use_tavily.lower() == "true":
            res = tavily.search(q, max_results=limit)

            for r in res.get("results", []):
                results.append(
                    {
                        "query": q,
                        "title": r.get("title"),
                        "url": r.get("url"),
                        "content": r.get("content") or r.get("snippet") or "",
                    }
                )
        else:
            with DDGS() as ddgs:
                for r in ddgs.text(q, max_results=limit):
                    title = r.get("title") or r.get("name") or r.get("body", "")[:120]
                    url = r.get("href") or r.get("url") or r.get("link") or ""
                    content = r.get("body") or r.get("text") or ""
                    results.append(
                        {"query": q, "title": title, "url": url, "content": content}
                    )

    state["hits"] = results
    state["actions"].append(f"web_search({len(results)})")

    return state


def synthesize(state: GraphState) -> GraphState:
    """
    Synthesizes a concise, factual answer to a user's question based on provided source fragments.
    Args:
        state (GraphState): The current state of the graph containing user question, source hits,
                            and other contextual information.
    Returns:
        GraphState: The updated state with the synthesized answer and an appended action log.
    Functionality:
        - Extracts the user's question from the state.
        - Compacts source fragments to fit within a character limit for processing by the LLM.
        - Constructs a prompt for the language model to generate a response.
        - Invokes the language model to produce a synthesized answer.
        - Updates the state with the generated answer and logs the "synthesize" action.
    """
    user_q = _get_user_question(state)
    sources_view = _compact_sources_for_llm(state["hits"], max_chars=8000)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a pragmatic research synthesizer.\n"
                "Produce a concise, factual answer with bullet-point findings and clear references [n].\n"
                "Mention uncertainties. No opinions. No excessive text.",
            ),
            (
                "human",
                "Question:\n{question}\n\nSource fragments:\n{sources}\n\n"
                "Instruction: First provide a short conclusion, then bullets with justification and [source index].",
            ),
        ]
    )

    chain = prompt | llm

    response = chain.invoke({"question": user_q, "sources": sources_view})

    state["answer"] = response.content
    state["actions"].append("synthesize")

    return state


def router(state: GraphState) -> GraphState:
    """
    Updates the state of a graph by incrementing the round index and appending
    the appropriate action to the state's actions list based on the current round.
    Args:
        state (GraphState): A dictionary-like object representing the current state
            of the graph. It must contain the following keys:
            - "round_idx" (int): The current round index.
            - "max_rounds" (int): The maximum number of rounds.
            - "actions" (list): A list of actions performed in the graph.
    Returns:
        GraphState: The updated state with the incremented round index and the
        appended action.
    """
    nxt = state["round_idx"] + 1

    if nxt < state["max_rounds"]:
        state["actions"].append(f"router(loop_{nxt})")

    else:
        state["actions"].append("router(final)")

    state["round_idx"] = nxt

    return state


def should_continue(state: GraphState) -> str:
    """
    Determines whether the process should continue looping or move to the final stage.
    Args:
        state (GraphState): A dictionary-like object containing the current state of the graph,
                            including the current round index ("round_idx") and the maximum number
                            of rounds ("max_rounds").
    Returns:
        str: Returns "loop" if the current round index is less than the maximum number of rounds.
             Returns "final" otherwise.
    """
    if state["round_idx"] < state["max_rounds"]:
        return "loop"

    return "final"
