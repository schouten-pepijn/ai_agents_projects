from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
import datetime
from tools.visualization_tools import create_revenue_chart
from states.bistate import BiState
import logging

logger = logging.getLogger(__name__)


class VisualizationAgent:
    def __init__(self, llm):
        self.llm = llm
        self.tools = [create_revenue_chart]
        # Bind tools to the LLM
        self.llm_with_tools = self.llm.bind_tools(self.tools)

    def run(self, state: BiState) -> BiState:
        # Get analysis context
        analysis_summary = str(state.analysis_results) if state.analysis_results else "No analysis results available"
        
        prompt = ChatPromptTemplate.from_template(
            "You are a data visualization agent. "
            "Create compelling visualizations based on the analyzed business data. "
            "Analysis results: {analysis_summary}\n\n"
            "Use the create_revenue_chart tool to generate a revenue visualization."
        )

        try:
            # Create the prompt with analysis context
            messages = [
                HumanMessage(content=prompt.format(analysis_summary=analysis_summary))
            ]

            # Let the LLM decide which tools to use
            response = self.llm_with_tools.invoke(messages)
            
            chart_result = None
            if response.tool_calls:
                for tool_call in response.tool_calls:
                    try:
                        if tool_call['name'] == 'create_revenue_chart':
                            args = tool_call.get('args', {})
                            result = create_revenue_chart.invoke(args)
                            chart_result = result
                            break
                    except Exception as e:
                        logger.exception(f"Tool execution failed for {tool_call['name']}")
                        chart_result = f"Chart creation failed: {str(e)}"
            
            if chart_result is None:
                # Fallback
                logger.warning("LLM did not use visualization tools, creating fallback")
                chart_result = "Visualization placeholder - chart would be generated here"

        except Exception as e:
            logger.exception("Visualization agent failed")
            chart_result = f"Visualization failed: {str(e)}"

        # Create visualization record
        visualization = {
            "type": "revenue_chart",
            "description": "Daily revenue trend visualization",
            "created_at": datetime.datetime.now().isoformat(),
            "result": chart_result,
        }

        state.visualizations.append(visualization)
        state.messages.append(
            AIMessage(content=f"Visualization created: {chart_result}")
        )
        state.current_agent = "reporting"

        return state
