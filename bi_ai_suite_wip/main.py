from orchestrator.bi_orchestrator import BiOrchestrator
from config.config import LlmConfig

from langchain_ollama import ChatOllama
from agents.data_collection_agent import DataCollectionAgent
from states.bistate import BiState
from langchain_core.messages import HumanMessage

if __name__ == "__main__":
    config = LlmConfig()

    # bi_system = BiOrchestrator(
    #     llm_base_url=config.llm_base_url,
    #     llm_model=config.llm_model,
    # )
    
    # final_result = bi_system.run_bi_pipeline()
    
    # summary = bi_system.get_summary(final_result)
    
    # print("Final BI Pipeline Summary:")
    # print(summary)
    
    llm = ChatOllama(
            base_url=config.llm_base_url,
            model=config.llm_model,
            temperature=0.1,
        )

    initial_task = "Collect comprehensive business data"

    initial_state = BiState(
            messages=[HumanMessage(content=initial_task)],
            data_sources={},
            raw_data={},
            analysis_results={},
            visualizations=[],
            reports=[],
            current_agent="data_collection",
        )

    data_agent = DataCollectionAgent(llm)
    
    state = data_agent.run(initial_state)
    
    print("=== Testing DataCollectionAgent ===")
    print(f"Initial state - Data sources: {len(initial_state.data_sources)}, Raw data: {len(initial_state.raw_data)}")
    
    final_state = data_agent.run(initial_state)
    
    print(f"Final state - Data sources: {len(final_state.data_sources)}, Raw data: {len(final_state.raw_data)}")
    print(f"Data sources collected: {list(final_state.data_sources.keys())}")
    print(f"Raw data keys: {list(final_state.raw_data.keys())}")
    
    # Print the last message to see what the agent accomplished
    if final_state.messages:
        print(f"Last message: {final_state.messages[-1].content}")
    
    # Check if sales data was collected
    if 'sales' in final_state.raw_data:
        sales_df = final_state.raw_data['sales']
        print(f"Sales data shape: {sales_df.shape}")
        print(f"Sales data columns: {sales_df.columns}")
    
    # Check if KPI data was collected
    if 'kpis' in final_state.raw_data:
        kpi_data = final_state.raw_data['kpis']
        print(f"KPI data: {kpi_data}")
    
    