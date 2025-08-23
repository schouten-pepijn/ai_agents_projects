import random
from typing import List
from langchain_core.messages import HumanMessage
from .state import DebateState, Role, Utterance
from .prompts import pro_sys, con_sys, mod_judge_tpl, final_synth_tpl
from .config import MIN_ROUNDS, MAX_ROUNDS
from .config import llm_pro, llm_con, llm_mod
from .streaming import stream_llm


def parse_judgement(text: str):
    """
    Parse the moderator's judgement text to extract the winner and rationale.

    Args:
        text (str): The judgement text from the moderator.

    Returns:
        tuple: A tuple containing the winner (str) and rationale (str).
    """
    winner = "tie"
    rationale = ""
    
    for line in text.splitlines():
        s = line.strip()
        
        if s.upper().startswith("WINNER:"):
            w = s.split(":", 1)[1].strip().lower()
            
            if w in ("pro", "con", "tie"):
                winner = w
                
        elif s.upper().startswith("RATIONALE:"):
            rationale = s.split(":", 1)[1].strip()
            
    return winner, rationale or text.strip()


def speak(role: Role, state: DebateState, chat_history: List[List[str]]):
    """
    Generate a response for the given role (Pro or Contra) and update the chat history.

    Args:
        role (Role): The role of the speaker (Pro or Contra).
        state (DebateState): The current state of the debate.
        chat_history (List[List[str]]): The chat history to update.

    Yields:
        List[List[str]]: Updated chat history after each chunk.
    """
    sys_msg = pro_sys if role == Role.PRO else con_sys
    messages = [sys_msg, HumanMessage(content=f"Hypothese: {state.topic}\nReact for round {state.round_idx+1}.")]
    
    label = "🟢 Pro" if role == Role.PRO else "🔴 Contra"
    chat_history.append([f"{label} (Round {state.round_idx+1})", ""])
    
    llm = llm_pro if role == Role.PRO else llm_con
    
    final = ""
    for partial in stream_llm(llm, messages):
        final = partial
        chat_history[-1][1] = final
        yield chat_history
        
    state.transcript.append(Utterance(role=role.value, content=final.strip(), round=state.round_idx))


def judge_round(state: DebateState, chat_history: List[List[str]]):
    """
    Judge the current round by comparing Pro and Contra arguments.

    Args:
        state (DebateState): The current state of the debate.
        chat_history (List[List[str]]): The chat history to update.

    Yields:
        List[List[str]]: Updated chat history after each chunk.
    """
    pro_txt = next((u.content for u in reversed(state.transcript) if u.role == "pro" and u.round == state.round_idx), "")
    con_txt = next((u.content for u in reversed(state.transcript) if u.role == "con" and u.round == state.round_idx), "")
    
    messages = mod_judge_tpl.format_messages(topic=state.topic, pro_text=pro_txt, con_text=con_txt)
    chat_history.append([f"⚖️ Moderator (Round {state.round_idx+1})", ""])
    
    final = ""
    for partial in stream_llm(llm_mod, messages):
        final = partial
        chat_history[-1][1] = final
        yield chat_history
    
    winner, rationale = parse_judgement(final)
    state.winner_per_round.append(winner)
    state.rationale.append(rationale)
    state.transcript.append(Utterance(role=Role.MOD.value, content=f"Round {state.round_idx+1} Winner: {winner}. Rationale: {rationale}", round=state.round_idx))


def summarize(state: DebateState):
    """
    Generate a final summary for the debate based on the rationale and winners.

    Args:
        state (DebateState): The current state of the debate.

    Yields:
        str: The final summary text after each chunk.
    """
    rationales = "\n".join([f"R{i+1}: {r}" for i, r in enumerate(state.rationale)])
    messages = final_synth_tpl.format_messages(topic=state.topic, n=state.max_rounds, w=", ".join(state.winner_per_round) or "None", rationales=rationales)
    
    final = ""
    for partial in stream_llm(llm_mod, messages):
        final = partial
        yield final
        
    state.final_summary = final.strip()


def debate_stream(topic: str, rounds: int):
    """
    Orchestrate the debate flow, alternating speakers and judging rounds.

    Args:
        topic (str): The debate topic.
        rounds (int): The number of rounds for the debate.

    Yields:
        tuple: Updated chat history, status message, and summary text.
    """
    topic = (topic or "").strip()
    
    if not topic:
        yield [], "", ""
        return
    
    max_rounds = max(MIN_ROUNDS, min(MAX_ROUNDS, int(rounds)))
    state = DebateState(topic=topic, max_rounds=max_rounds)
    chat_history: List[List[str]] = []
    
    starter = random.choice([Role.PRO, Role.CON])
    
    for r in range(state.max_rounds):
        state.round_idx = r
        first = starter if r % 2 == 0 else (Role.CON if starter == Role.PRO else Role.PRO)
        second = Role.CON if first == Role.PRO else Role.PRO
        
        # Speakers
        for role in (first, second):
            yield chat_history, f"Round {r+1}/{state.max_rounds} - {role.value} speaking...", ""
            
            for _ in speak(role, state, chat_history):
                yield chat_history, f"Round {r+1}/{state.max_rounds} - {role.value} speaking...", ""
        
        # Judge
        yield chat_history, f"Round {r+1}/{state.max_rounds} - moderator judging...", ""
        
        for _ in judge_round(state, chat_history):
            yield chat_history, f"Round {r+1}/{state.max_rounds} - moderator judging...", ""
    
    # Summary
    yield chat_history, "Generating final summary...", ""
    
    for partial in summarize(state):
        yield chat_history, "Generating final summary...", partial
        
    yield chat_history, "Debate completed!", state.final_summary or ""
