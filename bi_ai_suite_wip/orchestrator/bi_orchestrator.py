from typing import Any, Dict
from langchain_ollama import ChatOllama
from langchain_core.messages.human import HumanMessage
from langgraph.graph import StateGraph, START, END
from states.bistate import BiState
from agents.analysis_agent import AnalysisAgent
from agents.data_collection_agent import DataCollectionAgent
from agents.reporting_agent import ReportingAgent
from agents.visualization_agent import VisualizationAgent


class BiOrchestrator:
    def __init__(self, llm_base_url: str, llm_model: str):
        self.llm = ChatOllama(
            base_url=llm_base_url,
            model=llm_model,
            temperature=0.1,
        )

        self.data_agent = DataCollectionAgent(self.llm)
        self.analysis_agent = AnalysisAgent(self.llm)
        self.viz_agent = VisualizationAgent(self.llm)
        self.report_agent = ReportingAgent(self.llm)

        self.workflow = self._create_workflow()

    def _create_workflow(self):
        workflow = StateGraph(BiState)

        workflow.add_node("data_collection", self.data_agent.run)
        workflow.add_node("analysis", self.analysis_agent.run)
        workflow.add_node("visualization", self.viz_agent.run)
        workflow.add_node("reporting", self.report_agent.run)

        workflow.add_edge(START, "data_collection")
        
        workflow.add_edge("data_collection", END)
        
        # workflow.add_edge("data_collection", "analysis")
        # workflow.add_edge("analysis", "visualization")
        # workflow.add_edge("visualization", "reporting")
        # workflow.add_edge("reporting", END)

        return workflow.compile()

    def run_bi_pipeline(
        self, initial_task: str = "Generate business intelligence report"
    ):
        initial_state = BiState(
            messages=[HumanMessage(content=initial_task)],
            data_sources={},
            raw_data={},
            analysis_results={},
            visualizations=[],
            reports=[],
            current_agent="data_collection",
        )
        final_state = self.workflow.invoke(initial_state)

        return final_state

    def get_summary(self, state) -> Dict[str, Any]:
        # Handle both BiState objects and dict representations
        if isinstance(state, dict):
            # If state is a dict, access attributes as dict keys
            return {
                "execution_completed": state.get("task_completed", False),
                "data_sources_collected": len(state.get("data_sources", {})),
                "datasets_processed": len(state.get("raw_data", {})),
                "analysis_insights": len(state.get("analysis_results", {}).get("key_insights", [])),
                "visualizations_created": len(state.get("visualizations", [])),
                "reports_generated": len(state.get("reports", [])),
                "total_messages": len(state.get("messages", [])),
            }
        else:
            # If state is a BiState object, access attributes normally
            return {
                "execution_completed": state.task_completed,
                "data_sources_collected": len(state.data_sources),
                "datasets_processed": len(state.raw_data),
                "analysis_insights": len(state.analysis_results.get("key_insights", [])),
                "visualizations_created": len(state.visualizations),
                "reports_generated": len(state.reports),
                "total_messages": len(state.messages),
            }
