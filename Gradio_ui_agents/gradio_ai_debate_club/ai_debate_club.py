import gradio as gr
from debate import debate_stream
from config import MIN_ROUNDS, MAX_ROUNDS

with gr.Blocks(title="Multi-Agent Debate", theme=gr.themes.Soft()) as debate_app:
    gr.Markdown("""
    # Multi-Agent Debate Club
    Watch AI agents debate in real-time! Two agents argue different sides while a moderator judges each round.
    """)
    
    with gr.Row():
        with gr.Column(scale=2):
            topic = gr.Textbox(label="Debate Topic", placeholder="Example: 'Artificial intelligence will replace most human jobs within 20 years.'", lines=2)
        
        with gr.Column(scale=1):
            rounds = gr.Slider(MIN_ROUNDS, MAX_ROUNDS, value=MIN_ROUNDS, step=1, label="Number of Rounds")
            run_btn = gr.Button("Start Debate", variant="primary", size="lg")
    
    with gr.Row():
        with gr.Column(scale=2):
            chat = gr.Chatbot(label="Live Debate Transcript", height=600, show_label=True, container=True, bubble_full_width=False)
        
        with gr.Column(scale=1):
            round_info = gr.Textbox(label="Status", interactive=False, lines=2)
            summary = gr.Markdown(label="Final Summary", height=300)
            
    run_btn.click(debate_stream, inputs=[topic, rounds], outputs=[chat, round_info, summary], show_progress=True)

if __name__ == "__main__":
    debate_app.launch(share=False, debug=True)