from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from tools.collection_tools import collect_sales_data, collect_kpi_data
from states.bistate import BiState
from data_generators.generate_data import (
    generate_sample_sales_data,
    generate_sample_kpi_data,
)
import logging

logger = logging.getLogger(__name__)


class DataCollectionAgent:
    def __init__(self, llm):
        self.llm = llm
        self.tools = [collect_sales_data, collect_kpi_data]
        self.llm_with_tools = self.llm.bind_tools(self.tools)

    def run(self, state: BiState) -> BiState:
        prompt = ChatPromptTemplate.from_template(
            "You are a data collection agent for a business intelligence system. "
            "Your role is to gather comprehensive business data from available sources. "
            "Focus on collecting sales metrics, KPIs, and operational data. "
            "Be systematic and thorough in your data collection approach. "
            "Current task: {task}\n\n"
            "Please use the available tools to collect both sales data and KPI data. "
            "Call collect_sales_data first, then collect_kpi_data."
        )

        messages = [
            HumanMessage(content=prompt.format(task="Collect comprehensive business data"))
        ]

        # Let the LLM decide which tools to use and when
        response = self.llm_with_tools.invoke(messages)
        
        if tool_calls := response.tool_calls:
            for tool_call in tool_calls:
                tool_name = tool_call['name']
                tool_args = tool_call.get('args', {})
                
                if tool_name == 'collect_sales_data':
                    sales_data = collect_sales_data.invoke(tool_args)
                    state.raw_data['sales'] = sales_data
                    state.data_sources['sales'] = "Simulated Sales Database"
                
                elif tool_name == 'collect_kpi_data':
                    kpi_data = collect_kpi_data.invoke(tool_args)
                    state.raw_data['kpis'] = kpi_data
                    state.data_sources['kpis'] = "Simulated KPI Database"
        
        state.messages.append(AIMessage(content="Data collection completed."))
        state.current_agent = "analysis"

        return state
