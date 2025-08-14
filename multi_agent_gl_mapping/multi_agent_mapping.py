import os
import json
import time
from typing import Any, Dict, List, Literal, Callable
from dataclasses import dataclass
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.output_parsers import PydanticOutputParser, OutputFixingParser

load_dotenv(".env")

@dataclass(frozen=True)
class LLMSettings:
    model_url: str = os.environ["OLLAMA_URL"]
    model: str = os.environ["MODEL"]
    temperature: float = float(os.environ.get("TEMPERATURE", 0.2))


LLMSETTINGS = LLMSettings()

print(LLMSETTINGS)

llm = ChatOllama(
    base_url=LLMSETTINGS.model_url,
    model=LLMSETTINGS.model,
    temperature=LLMSETTINGS.temperature,
    num_ctx=4096,
)

embeddings = OllamaEmbeddings(
    model=LLMSETTINGS.model,
    base_url=LLMSETTINGS.model_url
)

historical_mappings = [
    {"acc_id": "4001", "desc": "Sales revenue - domestic", "category": "Revenue"},
    {"acc_id": "4002", "desc": "Sales revenue - export", "category": "Revenue"},
    {"acc_id": "5001", "desc": "Cost of goods sold - raw materials", "category": "COGS"},
    {"acc_id": "6100", "desc": "Wages and salaries", "category": "Payroll Expense"},
    {"acc_id": "1110", "desc": "Bank account - checking", "category": "Cash and Cash Equivalents"},
    {"acc_id": "2100", "desc": "Accounts payable - trade", "category": "Trade Payables"},
    {"acc_id": "3000", "desc": "Accumulated depreciation - plant", "category": "Accumulated Depreciation"},
]

docs = [
    Document(
        page_content=f"{r['acc_id']} | {r['desc']} | {r['category']}",
        metadata=r
    ) for r in historical_mappings
]

vectorstore = FAISS.from_documents(docs, embeddings)

# Tools
class ToolResult:
    def __init__(self, success: bool, payload: Any = None, message: str = ""):
        self.success = success
        self.payload = payload
        self.message = message
        

def tool_retrieve_similar(query: str, k: int = 3) -> ToolResult:
    retrieved_docs = vectorstore.similarity_search(query, k=k)
    results = [
        {"acc_id": d.metadata["acc_id"], "desc": d.metadata["desc"], "category": d.metadata["category"]}
        for d in retrieved_docs
    ]
    
    return ToolResult(
        success=True,
        payload=results
    )
    
def tool_summarize_samples(samples: List[Dict[str, str]]) -> ToolResult:
    prompt = PromptTemplate(
        input_variables=["samples_json"],
        template=(
            "You are an assistant that produces a short, focused summary.\n"
            "Given these example mappings (json):\n{samples_json}\n"
            "Return a one-paragraph summary mentioning common patterns and likely categories."
        )
    )
    
    chain = prompt | llm
    summary = chain.invoke(input={"samples_json": json.dumps(samples, ensure_ascii=False)}).content

    return ToolResult(success=True, payload=summary)

AllowedAction = Literal["retrieve_similar", "summarize_samples", "suggest_mapping", "write_mapping"]

class PlanStep(BaseModel):
    id: int = Field(..., description="Step id, strictly increasing")
    action: AllowedAction
    args: Dict[str, Any] = Field(default_factory=dict)
    
    @field_validator("args")
    @classmethod
    def validate_args_keys(cls, v, info):
        if 'action' not in info.data:
            return v
            
        action = info.data['action']
        required_keys = {
            "retrieve_similar": ["query"],
            "summarize_samples": ["samples"],
            "suggest_mapping": ["account_description", "context_summary"],
            "write_mapping": ["acc_id", "original_desc", "mapping"]
        }
        
        if action in required_keys:
            expected_keys = set(required_keys[action])
            actual_keys = set(v.keys())
            
            missing_keys = expected_keys - actual_keys
            if missing_keys:
                raise ValueError(f"Action '{action}' missing required args: {missing_keys}")
                
        return v
    

class Plan(BaseModel):
    task: str
    steps: List[PlanStep]

    @field_validator("steps")
    @classmethod
    def steps_must_have_min_items(cls, v):
        if not v or len(v) < 1:
            raise ValueError("steps must contain at least one item")
        return v

    @field_validator("steps")
    @classmethod
    def ids_must_increase(cls, v):
        ids = [s.id for s in v]
        if ids != sorted(ids):
            raise ValueError("Plan step ids must be strictly non-decreasing order")
        return v


class MappingResult(BaseModel):
    mapped_category: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    rationale: str
    
    
plan_parser = PydanticOutputParser(pydantic_object=Plan)
plan_fixing_parser = OutputFixingParser.from_llm(parser=plan_parser, llm=llm)

mapping_parser = PydanticOutputParser(pydantic_object=MappingResult)
mapping_fixing_parser = OutputFixingParser.from_llm(parser=mapping_parser, llm=llm)

planner_prompt = PromptTemplate(
    input_variables=["task", "notes"],
    partial_variables={"format_instructions": plan_parser.get_format_instructions()},
    template=(
        "You are an AI Planner. Decompose the task into an ordered list of tool steps.\n"
        "Allowed actions and their required argument keys:\n"
        "- retrieve_similar: args={{\"query\": str, \"k\": int}}\n"
        "- summarize_samples: args={{\"samples\": list}}\n"
        "- suggest_mapping: args={{\"account_description\": str, \"context_summary\": str}}\n"
        "- write_mapping: args={{\"acc_id\": str, \"original_desc\": str, \"mapping\": dict}}\n"
        "Each step: {{id:int, action:str, args:object}}.\n"
        "Use the EXACT argument keys specified above for each action.\n"
        "Return a JSON object {{\"task\": str, \"steps\": [ ... ]}}.\n"
        "{format_instructions}\n\n"
        "Task: {task}\nNotes: {notes}"
    ),
)

planner_chain = planner_prompt | llm | plan_fixing_parser


class Planner:
    def create_plan(self, task: str, notes: str = "") -> Plan:
        return planner_chain.invoke(input={"task": task, "notes": notes})
    
    
mapping_prompt = PromptTemplate(
    input_variables=["desc", "summary"],
    partial_variables={"format_instructions": mapping_parser.get_format_instructions()},
    template=(
        "You are a senior accounting mapping assistant.\n"
        "Given account description:\n{desc}\n\n"
        "Context summary:\n{summary}\n\n"
        "Return ONLY the JSON for fields: mapped_category, confidence (0.0-1.0), rationale.\n"
        "{format_instructions}"
    ),
)

mapping_chain = mapping_prompt | llm | mapping_fixing_parser


def tool_suggest_mapping(account_description: str, context_summary: str) -> ToolResult:
    out: MappingResult = mapping_chain.invoke(
        {"desc": account_description, "summary": context_summary}
    )
    return ToolResult(True, payload=out.model_dump())

def tool_write_mapping(acc_id: str, original_desc: str, mapping: Dict[str, Any]) -> ToolResult:
    time.sleep(0.05)
    record = {
        "acc_id": acc_id,
        "original_desc": original_desc,
        "mapped_category": mapping.get("mapped_category"),
        "confidence": mapping.get("confidence"),
        "rationale": mapping.get("rationale"),
    }
    print("PERSISTED:", json.dumps(record, ensure_ascii=False))
    return ToolResult(True, payload=record, message="Persisted to mapping store (mock)")

TOOL_REGISTRY: Dict[str, Callable[..., ToolResult]] = {
    "retrieve_similar": tool_retrieve_similar,
    "summarize_samples": tool_summarize_samples,
    "suggest_mapping": tool_suggest_mapping,
    "write_mapping": tool_write_mapping,
}


class Executor:
    def __init__(self, tool_registry: Dict[str, Callable[..., ToolResult]]):
        self.tools = tool_registry
        self.logs: List[Dict[str, Any]] = []

    def execute_plan(self, plan: Plan, context: Dict[str, Any]) -> Dict[str, Any]:
        steps = plan.steps
        memory: Dict[str, Any] = {"intermediate": {}}

        for step in steps:
            step_id = step.id
            action = step.action
            args_template = step.args

            # Resolve {input.*}/{memory.*} placeholders
            resolved_args: Dict[str, Any] = {}
            for k, v in args_template.items():
                if isinstance(v, str) and v.startswith("{") and v.endswith("}"):
                    ref = v[1:-1]
                    if ref.startswith("input."):
                        key = ref.split(".", 1)[1]
                        resolved_args[k] = context.get(key)
                    elif ref.startswith("memory."):
                        key = ref.split(".", 1)[1]
                        resolved_args[k] = memory["intermediate"].get(key)
                    else:
                        resolved_args[k] = v
                else:
                    resolved_args[k] = v

            tool = self.tools.get(action)
            if not tool:
                raise ValueError(f"Unknown tool/action: {action}")

            print(f"Executor: running step {step_id} -> {action} with args {resolved_args}")
            result = tool(**resolved_args)  # type: ignore
            self.logs.append(
                {"step_id": step_id, "action": action, "args": resolved_args,
                 "result": result.payload if result.success else None, "message": result.message}
            )

            # stash outputs
            if action == "retrieve_similar":
                memory["intermediate"]["retrieved"] = result.payload
            elif action == "summarize_samples":
                memory["intermediate"]["summary"] = result.payload
            elif action == "suggest_mapping":
                memory["intermediate"]["mapping"] = result.payload
            elif action == "write_mapping":
                memory["intermediate"]["persisted"] = result.payload

            if not result.success:
                print(f"Step {step_id} failed: {result.message}. Halting execution.")
                break

        return {"status": "completed", "logs": self.logs, "memory": memory}
    
    
    
if __name__ == "__main__":
    task = "Map GL account '4897 - Discounts given to customers - seasonal promo' to the taxonomy."
    notes = "Use historical mappings to inform category. If unsure, provide rationale and confidence."

    planner = Planner()
    plan: Plan = planner.create_plan(task=task, notes=notes)
    print("PLANNER OUTPUT (typed):", plan.model_dump_json(indent=2))

    # Fallback plan if planner fails (rare with parser, but safe)
    if not plan.steps:
        plan = Plan(
            task=task,
            steps=[
                PlanStep(id=1, action="retrieve_similar", args={"query": "{input.account_description}", "k": 4}),
                PlanStep(id=2, action="summarize_samples", args={"samples": "{memory.retrieved}"}),
                PlanStep(id=3, action="suggest_mapping", args={"account_description": "{input.account_description}",
                                                               "context_summary": "{memory.summary}"}),
                PlanStep(id=4, action="write_mapping", args={"acc_id": "{input.account_id}",
                                                             "original_desc": "{input.account_description}",
                                                             "mapping": "{memory.mapping}"}),
            ],
        )

    executor = Executor(TOOL_REGISTRY)
    context = {"account_id": "4897", "account_description": "Discounts given to customers - seasonal promo"}

    result = executor.execute_plan(plan, context)
    print("EXECUTION RESULT MEMORY:", json.dumps(result["memory"], indent=2, ensure_ascii=False))