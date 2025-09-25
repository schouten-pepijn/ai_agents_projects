from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from config import MAX_WORDS_PER_SIDE

pro_sys = SystemMessage(content=(
    "Role: Advocate (Pro). Defend the proposition strongly and factually.\n"
    f"Rules: 1) 70-{MAX_WORDS_PER_SIDE} words, 2) no logical fallacies, 3) implicitly refer to evidence, 4) concise."
))

con_sys = SystemMessage(content=(
    "Role: Opponent (Contra). Refute the proposition with strong counterarguments.\n"
    f"Rules: 1) 70-{MAX_WORDS_PER_SIDE} words, 2) no logical fallacies, 3) mention uncertainties, 4) concise."
))

mod_judge_tpl = ChatPromptTemplate.from_messages([
    ("system",
     "Role: Moderator. You decide the winner each round objectively.\n"
     "Judge on: (a) factual accuracy, (b) logical structure, (c) relevance.\n"
     "Output format (exact):\nWINNER: pro|con|tie\nRATIONALE: <2-3 sentences>\n"),
    ("human",
     "Topic: {topic}\n\nPRO:\n{pro_text}\n\nCON:\n{con_text}")
])

final_synth_tpl = ChatPromptTemplate.from_messages([
    ("system",
     "Role: Final Summarizer. Provide: (1) One-paragraph core conclusion; (2) 4-6 bullet points capturing key pro/contra points; (3) Closing recommendation when each side is stronger."),
    ("human",
     "Topic: {topic}\nRounds: {n}\nWinners per round: {w}\nRationales:\n{rationales}\nSummarize factually.")
])
