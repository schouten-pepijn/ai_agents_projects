from dotenv import load_dotenv
from typing import Iterator
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, trim_messages
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import gradio as gr
import os
import uuid

# Load environment variables
load_dotenv(".env")

# MODEL = os.environ["MODEL"]
MODEL = os.environ["MODEL_EXT"]
BASE_URL = os.environ["BASE_URL"]

# Initialize the model with streaming capability
model = ChatOllama(
    model=MODEL,
    base_url=BASE_URL,
    temperature=0.1,
    streaming=True
)

# Create the prompt template
prompt_template = ChatPromptTemplate(
    [
       ("system",
        "You are a helpful assistant. "\
            "You speak with an {accent} accent. " \
             "Answer all questions to the best of your ability. " \
       ),
       MessagesPlaceholder(
           variable_name="messages"
       )
    ]
)

# Function to stream responses from the model
def stream_response(message: str, accent: str) -> Iterator[str]:
    """Call the model and yield the tokens as they're generated."""
    # Create messages
    messages = [HumanMessage(content=message)]
    
    # Trim messages if needed
    trimmed_messages = trim_messages(
        messages,
        max_tokens=1024,
        strategy="last",
        token_counter=model,
        include_system=True,
        allow_partial=False,
        start_on="human",
    )
    
    # Create the prompt
    prompt = prompt_template.invoke(
        {
            "messages": trimmed_messages,
            "accent": accent
        }
    )
    
    # Stream the response
    for chunk in model.stream(prompt):
        yield chunk.content

# Dictionary to store conversation state
conversations = {}

# Create Gradio interface
with gr.Blocks(title="Chatbot with accent") as demo:
    gr.Markdown("# Chatbot with accent")
    gr.Markdown("Chat with this AI assistant that speaks with different accents!")

    with gr.Row():
        with gr.Column(scale=4):
            # Chat interface
            chatbot = gr.Chatbot(height=500)
            msg = gr.Textbox(
                placeholder="Type your message here...",
                container=False,
                scale=8,
                show_label=False
            )
            
        with gr.Column(scale=1):
            # Accent selector
            accent = gr.Dropdown(
                choices=[
                    "italian",
                    "british",
                    "australian",
                    "american",
                    "russian",
                    "french",
                    "german",
                    "spanish",
                    "indian",
                    "pirate",
                    "dutch",
                    "drunken"
                ],
                value="drunken",
                label="Select Accent"
            )
            
            # Clear button
            clear = gr.Button("Clear Chat")
    
    # Create unique thread ID for each session
    thread_id = gr.State(str(uuid.uuid4()))
    
    # Event handlers
    def user(user_message, history, thread_id, accent):
        # Update history with user message
        history = history + [(user_message, None)]
        return "", history, thread_id
    
    def bot(history, thread_id, accent):
        # Extract the last user message
        user_message = history[-1][0]
        
        # Store thread_id in conversations if it doesn't exist
        if thread_id not in conversations:
            conversations[thread_id] = []
        
        # Add message to conversation history
        conversations[thread_id].append(user_message)
        
        # Stream the response
        bot_response = ""
        for token in stream_response(user_message, accent):
            bot_response += token
            # Update the last response in the history
            history[-1] = (user_message, bot_response)
            yield history
    
    def reset_thread():
        new_thread_id = str(uuid.uuid4())
        return [], new_thread_id
    
    # Connect the components
    msg.submit(user, [msg, chatbot, thread_id, accent], [msg, chatbot, thread_id]).then(
        bot, [chatbot, thread_id, accent], chatbot
    )
    
    clear.click(fn=reset_thread, outputs=[chatbot, thread_id])


if __name__ == "__main__":
    demo.launch(share=False)