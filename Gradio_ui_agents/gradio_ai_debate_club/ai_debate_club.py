import os
import time
import random
from typing import List, Dict, TypedDict, Optional, Iterator
from dotenv import load_dotenv

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

import gradio as gr


load_dotenv(".env")

# MODEL = os.environ["MODEL_SMALL"]
MODEL = os.environ["MODEL_MEDIUM"]
# MODEL = os.environ["MODEL_LARGE"]

# Models
llm_pro = ChatOllama(
    model=MODEL,
    temperature=0.5,
    # max_tokens=1024
)
llm_con = ChatOllama(
    model=MODEL,
    temperature=0.5,
    # max_tokens=1024
)
llm_mod = ChatOllama(
    model=MODEL,
    temperature=0.2,
    # max_tokens=1024
)

# State
class DebateState(TypedDict):
    topic: str
    round_idx: int
    max_rounds: int
    transcript: List[Dict[str, str]]
    rationale: List[str]
    winner_per_round: List[str]
    final_summary: Optional[str]
    
    
# Prompts
pro_sys = SystemMessage(content=(
    "Role: Advocate (Pro). Goal: defend the proposition strongly and factually.\n"
    "Rules: 1) 100-120 words, 2) no logical fallacies, 3) implicitly refer to evidence, 4) brief and to the point."
))
con_sys = SystemMessage(content=(
    "Role: Opponent (Contra). Goal: refute the proposition with strong counterarguments.\n"
    "Rules: 1) 100-120 words, 2) no logical fallacies, 3) mention uncertainties, 4) brief and to the point."
))
mod_judge_tpl = ChatPromptTemplate.from_messages([
    (
        "system",
        "Role: Moderator. You decide the winner of the round objectively.\n"
        "Assess on: (a) factual accuracy, (b) logical structure, (c) relevance.\n"
        "Output format (exact):\n"
        "WINNER: pro|con|tie\n"
        "RATIONALE: <2-3 sentences>\n"
    ),
    (
        "human",
        "Topic: {topic}\n\n"
        "PRO:\n{pro_text}\n\n"
        "CON:\n{con_text}")
])

final_synth_tpl = ChatPromptTemplate.from_messages([
    (   "system",
        "Role: Final Summarizer. First provide a core conclusion in 1 paragraph. "
        "Then 4-6 bullet points with the key points from both sides. "
        "Conclude with a brief recommendation on when Pro or Contra is stronger."
    ),
    (
        "human",
        "Topic: {topic}\n"
        "Rounds: {n}\n"
        "Winners per round: {w}\n"
        "Rationales:\n{rationales}\n"
        "Summarize and remain factual."
    )
])

def stream_llm_response(llm, messages: List, speaker_name: str, chat_history: List, delay: float = 0.1) -> Iterator[tuple]:
    chat_history.append([speaker_name, ""])
    accumulated_text = ""
    
    try:
        for chunk in llm.stream(messages):
            if hasattr(chunk, 'content') and chunk.content:
                accumulated_text += chunk.content
              
                chat_history[-1][1] = accumulated_text
                yield chat_history.copy(), accumulated_text
                time.sleep(delay)
    except Exception:
        response = llm.invoke(messages)
        accumulated_text = response.content
        chat_history[-1][1] = accumulated_text
        yield chat_history.copy(), accumulated_text


def pro_speaks_stream(state: DebateState, chat_history: List) -> Iterator[tuple]:
    topic = state["topic"]
    messages = [pro_sys, HumanMessage(content=f"Hypothese: {topic}\nReact for round {state['round_idx']+1}.")]
    
    speaker_name = f"🟢 Pro (Round {state['round_idx']+1})"
    
    final_content = ""
    for updated_chat, content in stream_llm_response(llm_pro, messages, speaker_name, chat_history):
        final_content = content
        yield updated_chat, state
    
    state["transcript"].append({"role": "pro", "content": final_content.strip(), "round": state['round_idx']})

def con_speaks_stream(state: DebateState, chat_history: List) -> Iterator[tuple]:
    topic = state["topic"]
    messages = [con_sys, HumanMessage(content=f"Hypothese: {topic}\nReact for round {state['round_idx']+1}.")]
    
    speaker_name = f"🔴 Contra (Round {state['round_idx']+1})"
    
    final_content = ""
    for updated_chat, content in stream_llm_response(llm_con, messages, speaker_name, chat_history):
        final_content = content
        yield updated_chat, state
    
    state["transcript"].append({"role": "con", "content": final_content.strip(), "round": state['round_idx']})


def moderator_judges_stream(state: DebateState, chat_history: List) -> Iterator[tuple]:
    r = state["round_idx"]
    
    pro_text = next((t["content"] for t in reversed(state["transcript"]) if t["role"]=="pro" and t["round"]==r), "")
    con_text = next((t["content"] for t in reversed(state["transcript"]) if t["role"]=="con" and t["round"]==r), "")
    
    messages = mod_judge_tpl.format_messages(topic=state["topic"], pro_text=pro_text, con_text=con_text)
    
    speaker_name = f"Moderator (Round {r+1})"
    
    final_content = ""
    for updated_chat, content in stream_llm_response(llm_mod, messages, speaker_name, chat_history):
        final_content = content
        yield updated_chat, state
    
    judge = final_content.strip()
    winner = "tie"
    rationale = ""
    for line in judge.splitlines():
        s = line.strip()
        if s.upper().startswith("WINNER:"):
            w = s.split(":", 1)[1].strip().lower()
            if w in ("pro", "con", "tie"):
                winner = w
        if s.upper().startswith("RATIONALE:"):
            rationale = s.split(":", 1)[1].strip()
            
    state["winner_per_round"].append(winner)
    state["rationale"].append(rationale or judge)
    state["transcript"].append({"role": "mod", "content": f"Round {r+1} Winner: {winner}. Rationale: {rationale}", "round": r})


def finalize_stream(state: DebateState, chat_history: List) -> Iterator[tuple]:
    n = state["max_rounds"]
    w = ", ".join(state["winner_per_round"]) or "None"
    
    rationales = "\n".join([f"R{idx+1}: {rat}" for idx, rat in enumerate(state["rationale"])])
    
    messages = final_synth_tpl.format_messages(
        topic=state["topic"],
        n=n,
        w=w,
        rationales=rationales
    )
    
    speaker_name = "-- Final Summary"
    
    final_content = ""
    for updated_chat, content in stream_llm_response(llm_mod, messages, speaker_name, chat_history):
        final_content = content
        yield updated_chat, state
    
    state["final_summary"] = final_content.strip()


def run_debate_stream(topic: str, rounds: int):
    topic = (topic or '').strip()
    if not topic:
        yield [], '', ''
        return

    init: DebateState = {
        'topic': topic,
        'round_idx': 0,
        'max_rounds': max(2, int(rounds)),
        'transcript': [],
        'rationale': [],
        'winner_per_round': [],
        'final_summary': None
    }
    chat_history: List[List[str]] = []
    state = init

    starter = random.choice(['pro', 'con'])

    total_rounds = state['max_rounds']
    
    for r in range(total_rounds):
        state['round_idx'] = r
        
        if (r % 2 == 0 and starter == 'pro') or (r % 2 == 1 and starter == 'con'):
            ordering = ['pro', 'con', 'mod']
        else:
            ordering = ['con', 'pro', 'mod']

        for role in ordering:
            display_role = 'Moderator' if role == 'mod' else role.capitalize()
            yield chat_history, f"Round {r+1}/{total_rounds} - {display_role} speaking...", ""
            
            if role == 'pro':
                for updated_chat, _ in pro_speaks_stream(state, chat_history):
                    yield updated_chat, f"Round {r+1}/{total_rounds} - {display_role} speaking...", ""
            elif role == 'con':
                for updated_chat, _ in con_speaks_stream(state, chat_history):
                    yield updated_chat, f"Round {r+1}/{total_rounds} - {display_role} speaking...", ""
            elif role == 'mod':
                for updated_chat, _ in moderator_judges_stream(state, chat_history):
                    yield updated_chat, f"Round {r+1}/{total_rounds} - {display_role} speaking...", ""

    # Execute final summary
    yield chat_history, "Generating final summary...", ""
    for updated_chat, _ in finalize_stream(state, chat_history):
         yield updated_chat, "Generating final summary...", ""

    yield chat_history, "Debate completed!", state['final_summary'] or ""


with gr.Blocks(title="Multi-Agent Debate", theme=gr.themes.Soft()) as debate_app:
    gr.Markdown("""
    # 🎭 Multi-Agent Debate Club
    Watch AI agents debate in real-time! Two agents argue different sides while a moderator judges each round.
    """)
    
    with gr.Row():
        with gr.Column(scale=2):
            topic = gr.Textbox(
                label="💭 Debate Topic", 
                placeholder="Example: 'Artificial intelligence will replace most human jobs within 20 years.'",
                lines=2
            )
        with gr.Column(scale=1):
            rounds = gr.Slider(2, 5, value=2, step=1, label="Number of Rounds")
            run_btn = gr.Button("Start Debate", variant="primary", size="lg")

    with gr.Row():
        with gr.Column(scale=2):
            # Changed from Dataframe to Chatbot for better streaming visualization
            chat = gr.Chatbot(
                label="💬 Live Debate Transcript",
                height=600,
                show_label=True,
                container=True,
                bubble_full_width=False
            )
        with gr.Column(scale=1):
            round_info = gr.Textbox(label="📊 Status", interactive=False, lines=2)
            summary = gr.Markdown(label="Final Summary", height=300)

    run_btn.click(
        run_debate_stream, 
        inputs=[topic, rounds], 
        outputs=[chat, round_info, summary], 
        show_progress=True
    )

if __name__ == "__main__":
    debate_app.launch(share=False, debug=True)