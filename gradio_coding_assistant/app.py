import os
import sys
import tempfile
import subprocess
import json
import shutil
import ast
import gradio as gr

from typing import TypedDict, Literal, Optional, Dict, Any
from dataclasses import dataclass
from dotenv import load_dotenv

from langchain_ollama import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END, START

load_dotenv(".venv")

# config
@dataclass(frozen=True)
class Config:
    model: str = (
        # os.getenv("MODEL_LARGE")
        # or 
        os.getenv("MODEL_MEDIUM")
        or
        os.getenv("MODEL_SMALL")
    )
    base_url: str = os.getenv("BASE_URL", "").strip()

    llm: ChatOllama = None

    def __post_init__(self):
        if not self.base_url:
            raise RuntimeError("BASE_URL environment variable is required.")
        
        object.__setattr__(self, "llm", ChatOllama(
            model=self.model,
            base_url=self.base_url,
            temperature=0  
        ))

config = Config()

# state
class AgentState(TypedDict):
    user_request: str
    attempt: int
    max_attempts: int
    code: str
    exec_stdout: str
    exec_stderr: str
    exec_returncode: int
    eval_report: str
    decision: Literal["final", "repair"]
    

# prompts
SYSTEM_SPEC = """You write a complete Python module that solves the user's request.

Rules:
- Provide a single, self-contained .py script.
- Include a top-level function `def run():` which executes the solution and prints human-readable results.
- Avoid network calls and file deletion. No dangerous shell usage.
- Restrict to stdlib and throw an error otherwise.
- If extra deps are necessary, implement graceful fallback or pure-stdlib version.
- Put all logic in this single file, no placeholders, no TODOs."""

GEN_PROMPT = ChatPromptTemplate([
    ("system", SYSTEM_SPEC),
    ("human", "User request:\n{request}\n\nReturn ONLY valid Python code for a single module. No backticks.")
])

REPAIR_PROMPT = ChatPromptTemplate([
    ("system", SYSTEM_SPEC),
    ("human", """The previous code failed or was inadequate.
     
User request:
{request}

Previous code:
{code}

Execution return code: 
{rc}

STDOUT (truncated):
{stdout}

STDERR:
{stderr}

Evaluator feedback:
{eval_report}

Produce a fully-corrected single Python module. Return ONLY code, no explanations, no backticks."""),
])

EVAL_PROMPT = ChatPromptTemplate([
    ("system", """You are a strict code evaluator. Output a terse JSON object with keys:
- meets_requirements: boolean
- rationale: string
- must_fix: array of short strings
Judging criteria:
- Does 'run()' exist and execute without errors?
- Does the code produce the expected output?
- Is the solution self-contained and safe (no harmful ops)?
Only return JSON."""),
    ("human", """User request:
{request}

Code summary (first ~300 lines):
{code_head}

Observed:
- returncode={rc}
- stderr_present={stderr_present}
- stdout_excerpt={stdout_excerpt}""")
])


@dataclass
class ExecResult:
    returncode: int
    stdout: str
    stderr: str
    path: str

DENY_LIST_CALLS = {
    "os.remove", "os.rmdir",
    "shutil.rmtree", "subprocess.Popen",
    "subprocess.call", "subprocess.run",
    "sys.exit", "os.system"
}


def _basic_safety_ast_check(code: str) -> Optional[str]:
    code = code.strip()  # minor normalization
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return f"SyntaxError: {e}"
    
    bad_hits = []
    for node in ast.walk(tree):
        
        if isinstance(node, ast.Attribute):
            full = f"{getattr(node.value, 'id', '')}.{node.attr}"
            
            if full in DENY_LIST_CALLS:
                bad_hits.append(full)
    
    if bad_hits:
        return f"Denylisted usage: {', '.join(sorted(set(bad_hits)))}"
    
    return None


def run_in_subprocess(module_code: str, timeout_sec: int = 15) -> ExecResult:
    tmpdir = tempfile.mkdtemp(prefix="codeexec_")
    path = os.path.join(tmpdir, "solution.py")
    
    with open(path, "w", encoding="utf-8") as f:
        f.write(module_code)
        
    cmd = [sys.executable, "-c", "import runpy; m=runpy.run_path('solution.py'); m['run']()"]
    
    try:
        p = subprocess.run(
            cmd,
            cwd=tmpdir,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            env={
                k: v for k, v in os.environ.items()
                if k not in {"http_proxy", "https_proxy","HTTP_PROXY","HTTPS_PROXY"}
            }
        )
        return ExecResult(
            returncode=p.returncode,
            stdout=p.stdout,
            stderr=p.stderr,
            path=path
        )
        
    except subprocess.TimeoutExpired as e:
        return ExecResult(
            returncode=124,
            stdout=e.stdout or "",
            stderr="TimeoutExpired",
            path=path
        )
    
    except Exception as e:
        return ExecResult(
            returncode=1,
            stdout="",
            stderr=str(e),
            path=path
        )
        
    finally:
        try:
            shutil.rmtree(tmpdir)
            
        except Exception:
            pass


# Nodes
def draft_code(state: AgentState) -> AgentState:
    msg = GEN_PROMPT.invoke({
        "request": state["user_request"]
    })
    code = config.llm.invoke(msg.to_messages()).content
    return {**state, "code": code}


def exec_code(state: AgentState) -> AgentState:
    code = state["code"]
    violation = _basic_safety_ast_check(code)
    
    if violation:
        res = ExecResult(
            returncode=2,
            stdout="",
            stderr=f"Safety check failed: {violation}",
            path=""
        )
    else:
        res = run_in_subprocess(code, timeout_sec=20)
    
    return {
        **state,
        "exec_stdout": res.stdout,
        "exec_stderr": res.stderr,
        "exec_returncode": res.returncode
    }
    
def evaluate(state: AgentState) -> AgentState:
    """Run LLM-based evaluation ONLY (decision performed in separate 'decide' node)."""
    stdout_excerpt = state["exec_stdout"][:800]
    code_head = state["code"][:4000]
    
    msg = EVAL_PROMPT.invoke({
        "request": state["user_request"],
        "code_head": code_head,
        "rc": state["exec_returncode"],
        "stderr_present": bool(state["exec_stderr"]),
        "stdout_excerpt": stdout_excerpt
    })
    
    raw = config.llm.invoke(msg.to_messages()).content
    
    try:
        data = json.loads(raw)
        
    except Exception:
        data = {
            "meets_requirements": False,
            "rationale": "Non-JSON evaluator output",
            "must_fix": ["Evaluator failed to parse JSON."]
        }
    
    concise = {
        "meets_requirements": bool(data.get("meets_requirements")),
        "rationale": str(data.get("rationale", ""))[:500],
        "must_fix": [str(x) for x in data.get("must_fix", [])][:8],
    }
    
    return {
        **state,
        "eval_report": json.dumps(concise, ensure_ascii=False)
    }
    
def decide(state: AgentState) -> AgentState:
    # (No change in logic; now actually used in the graph.)
    report = json.loads(state["eval_report"])
    ok = (state["exec_returncode"] == 0) and report.get("meets_requirements", False)
    
    if ok or state["attempt"] >= state["max_attempts"]:
        return {
            **state,
            "decision": "final"
        }
    
    return {
        **state,
        "decision": "repair"
    }
    
def repair(state: AgentState) -> AgentState:
    msg = REPAIR_PROMPT.invoke({
        "request": state["user_request"],
        "code": state["code"],
        "rc": state["exec_returncode"],
        "stdout": state["exec_stdout"][:2000],
        "stderr": state["exec_stderr"][:2000],
        "eval_report": state["eval_report"],
    })
    
    new_code = config.llm.invoke(msg.to_messages()).content
    
    return {
        **state,
        "code": new_code,
        "attempt": state["attempt"] + 1
    }
    
    
#  Graph
graph = StateGraph(AgentState)
graph.add_node("draft_code", draft_code)
graph.add_node("execute_code", exec_code)
graph.add_node("evaluate", evaluate)
graph.add_node("decide", decide)       
graph.add_node("repair", repair)

graph.add_edge(START, "draft_code")
graph.add_edge("draft_code", "execute_code")
graph.add_edge("execute_code", "evaluate")
graph.add_edge("evaluate", "decide")        
graph.add_conditional_edges(                 
    "decide",
    lambda s: s["decision"],
    {"final": END, "repair": "repair"}
)
graph.add_edge("repair", "execute_code")

app_graph = graph.compile()


# Pipeline
def run_pipeline(user_request: str, max_attempts: int = 3) -> Dict[str, Any]:
    state: AgentState = {
        "user_request": user_request.strip(),
        "attempt": 1,
        "max_attempts": max_attempts,
        "code": "",
        "exec_stdout": "",
        "exec_stderr": "",
        "exec_returncode": 0,
        "eval_report": "",
        "decision": "repair",
    }
    last_state: Dict[str, Any] = dict(state)

    for event in app_graph.stream(state):
        for _node_name, partial in event.items():
            last_state.update(partial)

    return last_state

# UI
def gr_run(request: str, attempts: int):
    state: AgentState = {
        "user_request": request.strip(),
        "attempt": 1,
        "max_attempts": attempts,
        "code": "",
        "exec_stdout": "",
        "exec_stderr": "",
        "exec_returncode": 0,
        "eval_report": "",
        "decision": "repair",
    }

    code_snap, out_snap, err_snap, eval_snap, status_snap = "", "", "", "", ""
    
    for update in app_graph.stream(state):

        for node_name, node_state in update.items():
            
            code_snap = node_state.get("code", code_snap)
            out_snap = node_state.get("exec_stdout", out_snap)
            err_snap = node_state.get("exec_stderr", err_snap)
            eval_snap = node_state.get("eval_report", eval_snap)
            status_snap = f"Node: {node_name} | Attempt: {node_state.get('attempt', 1)}/{node_state.get('max_attempts', attempts)} | Decision: {node_state.get('decision', '')}"
            
            yield code_snap, out_snap, err_snap, eval_snap, status_snap

with gr.Blocks(title="Python Coding Assistant") as demo:
    
    gr.Markdown("## Python STDLIB Coding Assistant with Repair Loop")
    
    with gr.Row():
        req = gr.Textbox(label="Your request", lines=5, placeholder="Example: Generate a Python program that creates ASCII art from an image.")
        attempts = gr.Slider(1, 5, value=3, step=1, label="Max Attempts")
        
    run_btn = gr.Button("Generate, Run, and Validate")

    code_out = gr.Code(label="Generated Code", language="python")
    stdout_out = gr.Textbox(label="Execution STDOUT", lines=10)
    stderr_out = gr.Textbox(label="Execution STDERR", lines=10)
    eval_out = gr.Textbox(label="Evaluator Report (JSON)", lines=6)
    status = gr.Textbox(label="Status", lines=1)

    run_btn.click(
        fn=gr_run,
        inputs=[req, attempts],
        outputs=[code_out, stdout_out, stderr_out, eval_out, status]
    )

if __name__ == "__main__":
    demo.launch()