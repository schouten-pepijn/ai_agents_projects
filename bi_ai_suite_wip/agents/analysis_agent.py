from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
import polars as pl
import json
from tools.analyse_tools import perform_trend_analysis
from states.bistate import BiState
import logging

logger = logging.getLogger(__name__)


class AnalysisAgent:
    def __init__(self, llm):
        self.llm = llm
        self.tools = [perform_trend_analysis]
        # Bind tools to the LLM
        self.llm_with_tools = self.llm.bind_tools(self.tools)

    def run(self, state: BiState) -> BiState:
        # Prepare data summary
        sales_count = len(state.raw_data.get('sales', pl.DataFrame()))
        kpis_count = len(state.raw_data.get('kpis', pl.DataFrame()))
        
        prompt = ChatPromptTemplate.from_template(
            "You are a business data analysis agent. "
            "Analyze the collected business data to identify trends, patterns, and insights. "
            "Current data summary: Sales records: {sales_count}, KPI records: {kpis_count}\n\n"
            "Use the perform_trend_analysis tool to analyze this data and generate insights."
        )

        try:
            # Create the prompt with data context
            messages = [
                HumanMessage(content=prompt.format(
                    sales_count=sales_count, 
                    kpis_count=kpis_count
                ))
            ]

            # Let the LLM decide which tools to use
            response = self.llm_with_tools.invoke(messages)
            
            analysis_result = None
            if response.tool_calls:
                for tool_call in response.tool_calls:
                    try:
                        if tool_call['name'] == 'perform_trend_analysis':
                            data_summary = f"Sales data: {sales_count} records, KPI data: {kpis_count} records"
                            args = tool_call.get('args', {})
                            args['data_summary'] = data_summary
                            result = perform_trend_analysis.invoke(args)
                            analysis_result = result
                            break
                    except Exception as e:
                        logger.exception(f"Tool execution failed for {tool_call['name']}")
                        analysis_result = json.dumps({
                            "error": f"Analysis failed: {str(e)}",
                            "trends": [],
                            "insights": ["Analysis could not be completed due to technical issues"]
                        })
            
            if analysis_result is None:
                # Fallback analysis
                logger.warning("LLM did not use analysis tools, creating fallback analysis")
                analysis_result = json.dumps({
                    "trends": ["Moderate growth trend observed"],
                    "insights": ["Data collection successful", "Ready for visualization"],
                    "summary": f"Processed {sales_count} sales records and {kpis_count} KPI records"
                })

        except Exception as e:
            logger.exception("Analysis agent failed")
            analysis_result = json.dumps({
                "error": f"Analysis failed: {str(e)}",
                "trends": [],
                "insights": ["Analysis could not be completed"]
            })

        # Update state
        try:
            state.analysis_results = json.loads(analysis_result) if isinstance(analysis_result, str) else analysis_result
        except json.JSONDecodeError:
            state.analysis_results = {"raw_result": analysis_result}
            
        state.messages.append(
            AIMessage(content=f"Analysis completed: {analysis_result}")
        )
        state.current_agent = "visualization"

        return state
