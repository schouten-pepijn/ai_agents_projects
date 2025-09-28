from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
import datetime
import json
from tools.visualization_tools import generate_executive_report
from states.bistate import BiState
import logging

logger = logging.getLogger(__name__)


class ReportingAgent:
    def __init__(self, llm):
        self.llm = llm
        self.tools = [generate_executive_report]
        # Bind tools to the LLM
        self.llm_with_tools = self.llm.bind_tools(self.tools)

    def run(self, state: BiState) -> BiState:
        # Prepare context from previous agents
        analysis_str = json.dumps(state.analysis_results) if state.analysis_results else "{}"
        chart_info = f"Generated {len(state.visualizations)} visualizations"
        
        prompt = ChatPromptTemplate.from_template(
            "You are an executive reporting agent. "
            "Generate a comprehensive executive summary report based on the analysis and visualizations. "
            "Analysis results: {analysis_results}\n"
            "Visualization info: {chart_info}\n\n"
            "Use the generate_executive_report tool to create a professional executive summary."
        )

        try:
            # Create the prompt with context
            messages = [
                HumanMessage(content=prompt.format(
                    analysis_results=analysis_str,
                    chart_info=chart_info
                ))
            ]

            # Let the LLM decide which tools to use
            response = self.llm_with_tools.invoke(messages)
            
            report_result = None
            if response.tool_calls:
                for tool_call in response.tool_calls:
                    try:
                        if tool_call['name'] == 'generate_executive_report':
                            args = tool_call.get('args', {})
                            # Ensure required arguments are provided
                            if 'analysis_results' not in args:
                                args['analysis_results'] = analysis_str
                            if 'chart_info' not in args:
                                args['chart_info'] = chart_info
                            result = generate_executive_report.invoke(args)
                            report_result = result
                            break
                    except Exception as e:
                        logger.exception(f"Tool execution failed for {tool_call['name']}")
                        report_result = json.dumps({
                            "error": f"Report generation failed: {str(e)}",
                            "title": "Executive Summary (Error)",
                            "date": datetime.datetime.now().strftime("%Y-%m-%d")
                        })
            
            if report_result is None:
                # Fallback report
                logger.warning("LLM did not use reporting tools, creating fallback report")
                report_result = json.dumps({
                    "title": "Business Intelligence Executive Summary",
                    "date": datetime.datetime.now().strftime("%Y-%m-%d"),
                    "summary": "Report generated successfully",
                    "analysis_included": bool(state.analysis_results),
                    "visualizations_count": len(state.visualizations)
                })

        except Exception as e:
            logger.exception("Reporting agent failed")
            report_result = json.dumps({
                "error": f"Report generation failed: {str(e)}",
                "title": "Executive Summary (Error)",
                "date": datetime.datetime.now().strftime("%Y-%m-%d")
            })

        # Create report record
        try:
            report_content = json.loads(report_result) if isinstance(report_result, str) else report_result
        except json.JSONDecodeError:
            report_content = {"raw_result": report_result}
            
        report = {
            "content": report_content,
            "created_at": datetime.datetime.now().isoformat(),
            "type": "executive_summary",
        }

        state.reports.append(report)
        state.messages.append(AIMessage(content=f"Report generated: {report_result}"))
        state.current_agent = "complete"

        return state
