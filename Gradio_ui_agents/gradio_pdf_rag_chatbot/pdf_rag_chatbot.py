"""Main application file for PDF RAG chatbot."""
import logging
import gradio as gr
from retrieval_service import RetrievalService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

class PDFChatbotApp:
    def __init__(self):
        self.retrieval_service = RetrievalService()
    
    def ingest_pdfs(self, file_objects):
        try:
            return self.retrieval_service.ingest_documents(file_objects)
        
        except Exception as e:
            logger.error(f"Error in PDF ingestion: {str(e)}")
            return "", f"Error processing files: {str(e)}"
    
    def ask_question(self, session_id: str, question: str):
        try:
            return self.retrieval_service.answer_question(session_id, question)
        
        except Exception as e:
            logger.error(f"Error answering question: {str(e)}")
            return f"Error: {str(e)}", "<i>Error occurred</i>"
    
    def create_ui(self) -> gr.Blocks:
        with gr.Blocks(
            title="PDF RAG Chatbot",
            theme=gr.themes.Soft(),
            css="""
            .gradio-container {
                max-width: 1200px;
                margin: auto;
            }
            .upload-section {
                background: #f8fafc;
                padding: 20px;
                border-radius: 10px;
                margin-bottom: 20px;
            }
            """
        ) as app:
            
            gr.Markdown("""
            # PDF RAG Chatbot
            Upload PDF documents and ask questions to get precise answers with source citations.
            """)
            
            with gr.Row(elem_classes="upload-section"):
                with gr.Column():
                    file_upload = gr.File(
                        label="Upload PDF Files",
                        file_types=[".pdf"],
                        file_count="multiple",
                        height=120
                    )
                    
                    with gr.Row():
                        ingest_btn = gr.Button(
                            "Index Documents",
                            variant="primary",
                            size="lg"
                        )
            
            with gr.Row():
                with gr.Column(scale=2):
                    session_id = gr.Textbox(
                        label="Session ID",
                        interactive=False,
                        placeholder="No active session"
                    )
                with gr.Column(scale=3):
                    status = gr.Textbox(
                        label="Status",
                        interactive=False,
                        placeholder="Ready to process documents"
                    )
            
            gr.Markdown("---")
            
            with gr.Row():
                with gr.Column(scale=3):
                    question = gr.Textbox(
                        label="Your Question",
                        placeholder="Ask anything about your uploaded documents...",
                        lines=3,
                        max_lines=5
                    )
                with gr.Column(scale=1):
                    ask_btn = gr.Button(
                        "Ask Question",
                        variant="primary",
                        size="lg"
                    )
            
            with gr.Row():
                with gr.Column(scale=1):
                    answer = gr.Markdown(
                        label="Answer",
                        height=300
                    )
                with gr.Column(scale=1):
                    highlights = gr.HTML(label="Source Highlights")
            
            ingest_btn.click(
                fn=self.ingest_pdfs,
                inputs=[file_upload],
                outputs=[session_id, status],
                show_progress="full"
            )
            
            ask_btn.click(
                fn=self.ask_question,
                inputs=[session_id, question],
                outputs=[answer, highlights],
                show_progress="full"
            )

        return app

def create_app() -> gr.Blocks:
    app_instance = PDFChatbotApp()
    return app_instance.create_ui()

if __name__ == "__main__":
    app = create_app()
    app.launch()
