import gradio as gr
from runner import run_research_assistant

def create_gradio_web_app():
    """
    Creates and configures a Gradio web application for an AI Research Assistant.
    The web app provides the following UI components:
    - A Markdown header describing the assistant.
    - A textbox for users to input research questions.
    - Sliders to select the number of research rounds and results per query.
    - A button to start the research process.
    - A Markdown area to display the synthesized answer.
    - A dataframe to show the sources with their IDs, titles, and URLs.
    - A non-editable textbox to display the audit trail of actions taken.
    When the "Start Research" button is clicked, the app calls `run_research_assistant`
    with the user's question, selected rounds, and max results, and displays the answer,
    sources, and actions.
    Returns:
        gr.Blocks: The configured Gradio web application.
    """
    with gr.Blocks(title="AI Research Assistant") as web_app:
        gr.Markdown("### AI Research Assistant\nEnter a question. The agent searches, synthesizes, and cites.")
        
        with gr.Row():
            question = gr.Textbox(label="Question", placeholder="Example: 'What is the most effective way to learn Python?'")

        with gr.Row():
            rounds = gr.Slider(1, 4, value=2, step=1, label="Research rounds")
            max_results = gr.Slider(2, 10, value=5, step=1, label="Results per query")
            
        run_btn = gr.Button("Start Research")

        answer = gr.Markdown(label="Answer")
        sources = gr.Dataframe(
            headers=["id", "title", "url"],
            datatype=["number", "str", "str"],
            interactive=False,
            label="Web sources"
        )
        actions = gr.Textbox(label="Actions (audit trail)", interactive=False)

        run_btn.click(
            fn=run_research_assistant,
            inputs=[question, rounds, max_results],
            outputs=[answer, sources, actions],
        )
        
    return web_app

if __name__ == "__main__":
    web_app = create_gradio_web_app()
    web_app.launch(
        # share=True
    )