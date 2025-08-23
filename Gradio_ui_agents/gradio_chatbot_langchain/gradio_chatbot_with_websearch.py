from dotenv import load_dotenv
from typing import Iterator, Dict, Any
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, trim_messages
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.tools import DuckDuckGoSearchRun
from ddgs import DDGS
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.tools import Tool
import gradio as gr
import os
import uuid
import re

# Load environment variables
load_dotenv(".env")

MODEL = os.environ["MODEL_SMALL"]
# MODEL = os.environ["MODEL_LARGE"]

BASE_URL = os.environ["BASE_URL"]

# Initialize the model with streaming capability
model = ChatOllama(
    model=MODEL,
    base_url=BASE_URL,
    temperature=0.1,
    streaming=True
)

# Initialize DuckDuckGo search tool
search = DuckDuckGoSearchRun()

def ddgs_search(query: str) -> str:
    """Perform a search using DuckDuckGo Search API."""
    results = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=5):
            results.append(f"{r['title']}: {r['href']}")
    return "\n".join(results) if results else "No results found."

# Create tools list
tools = [
    Tool(
        name="web_search",
        description="Search the internet for current information, recent events, news, or specific factual data that might not be in your knowledge base. Use this when users ask about recent events, current statistics, or when you need up-to-date information.",
        func=ddgs_search,
    )
]

# Create the prompt template for regular chat (without tools)
prompt_template = ChatPromptTemplate(
    [
       ("system",
        "You are a helpful assistant. "
        "You speak in the {language} language and with an {accent} accent. "
        "Answer all questions to the best of your ability. "
        "If you don't have current information or if the user is asking about recent events, "
        "let them know that web search is available and they can enable it for more current information."
       ),
       MessagesPlaceholder(
           variable_name="messages"
       )
    ]
)

# Create the prompt template for agent (with tools)
agent_prompt_template = ChatPromptTemplate(
    [
       ("system",
        "You are a helpful assistant. "
        "You speak in the {language} language and with an {accent} accent. "
        "You have access to web search capabilities. Use the web_search tool when you need current information, "
        "recent events, news, or specific factual data that might not be in your training data. "
        "When you use web search, ALWAYS mention that you used web search and provide sources. "
        "Format your response clearly indicating whether information came from your knowledge or web search."
       ),
       MessagesPlaceholder(variable_name="chat_history", optional=True),
       ("human", "{input}"),
       MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

# Create agent
agent = create_tool_calling_agent(model, tools, agent_prompt_template)
agent_executor = AgentExecutor(
    agent=agent, 
    tools=tools, 
    verbose=True,
    handle_parsing_errors=True,
    return_intermediate_steps=True,
    max_iterations=4,
)

# Function to stream responses
def stream_response(message: str, language: str, accent: str) -> Iterator[str]:
    """Call the model and yield the tokens as they're generated."""
    messages = [HumanMessage(content=message)]
    
    trimmed_messages = trim_messages(
        messages,
        max_tokens=1024,
        strategy="last",
        token_counter=model,
        include_system=True,
        allow_partial=False,
        start_on="human",
    )
    
    prompt = prompt_template.invoke(
        {
            "messages": trimmed_messages,
            "language": language,
            "accent": accent
        }
    )
    
    for chunk in model.stream(prompt):
        yield chunk.content

# Function to get response with web search
def get_web_response(message: str, language: str, accent: str, conversation_history: list = None) -> Dict[str, Any]:
    """Call the agent with web search capability and return response with metadata."""
    try:
        # Prepare chat history if available
        chat_history = []
        if conversation_history:
            for msg in conversation_history[-6:]:  # Keep last 6 messages for context
                chat_history.extend([
                    HumanMessage(content=msg),
                ])
        
        # Execute agent
        result = agent_executor.invoke({
            "input": message,
            "language": language,
            "accent": accent,
            "chat_history": chat_history
        })
        
        sources = []
        used_web_search = False
        
        if "intermediate_steps" in result:
            for step in result["intermediate_steps"]:
                if step[0].tool == "web_search":
                    used_web_search = True
                    # Extract URLs from search results
                    search_result = step[1]
                    urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', search_result)
                    sources.extend(urls[:3])
        
        return {
            "response": result["output"],
            "used_web_search": used_web_search,
            "sources": sources[:5],
            "error": None
        }
    except Exception as e:
        return {
            "response": f"I encountered an error while searching the web: {str(e)}. Let me try to answer from my knowledge instead.",
            "used_web_search": False,
            "sources": [],
            "error": str(e)
        }

# Function to format response with source information
def format_response_with_sources(response: str, used_web_search: bool, sources: list) -> str:
    """Format the response with source information and indicators."""
    formatted_response = response
    
    # Add indicator of information source
    if used_web_search:
        formatted_response += "\n\n**Information source:** Web Search"
        if sources:
            formatted_response += "\n**Sources:**\n"
            for i, source in enumerate(sources, 1):
                # Clean up URL for display
                display_url = source.replace("https://", "").replace("http://", "")
                if len(display_url) > 50:
                    display_url = display_url[:50] + "..."
                formatted_response += f"   {i}. {display_url}\n"
    else:
        formatted_response += "\n\n**Information source:** AI Knowledge Base"

    return formatted_response


conversations = {}
# Gradio interface
with gr.Blocks(title="Chatbot with language and web search") as demo:
    gr.Markdown("# Chatbot with language and web search")
    gr.Markdown("Chat with this AI assistant that speaks in different languages and can search the web!")

    with gr.Row():
        with gr.Column(scale=4):
            # Chat interface
            chatbot = gr.Chatbot(
                height=500,
            )
            msg = gr.Textbox(
                placeholder="Type your message here...",
                container=False,
                scale=8,
                show_label=False
            )
            
        with gr.Column(scale=1):
            # language selector
            language = gr.Dropdown(
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
                    "dutch",
                    "swedish"
                ],
                value="dutch",
                label="Select language"
            )
            # language selector
            accent = gr.Dropdown(
                choices=[
                    "normal",
                    "pirate",
                    "minion",
                    "childish",
                    "robot",
                    "drunken",
                    "stuttering",
                    "yaba talk",
                ],
                value="normal",
                label="Select accent"
            )
            # web search toggle
            enable_web_search = gr.Checkbox(
                label="Enable Web Search",
                value=False,
                info="Allow the assistant to search the internet for current information"
            )
            
            # Clear button
            clear = gr.Button("Clear Chat")
            
            gr.Markdown(
                "**Note:** You can change the language, accent, and web search setting at any time during the conversation."
            )
    
    # Create unique thread ID for each session
    thread_id = gr.State(str(uuid.uuid4()))
    
    # Event handlers
    def user(user_message, history, thread_id, language, accent, enable_search):
        # Update history with user message
        history = history + [(user_message, None)]
        return "", history, thread_id

    def bot(history, thread_id, language, accent, enable_search):
        # Extract the last user message
        user_message = history[-1][0]
        
        if thread_id not in conversations:
            conversations[thread_id] = []
        
        conversations[thread_id].append(user_message)
        
        if enable_search:
            # Show searching indicator
            history[-1] = (user_message, "Searching the web and thinking...")
            yield history
            
            # Get response with web search
            result = get_web_response(user_message, language, accent, conversations[thread_id])
            
            # Format response with sources
            formatted_response = format_response_with_sources(
                result["response"], 
                result["used_web_search"], 
                result["sources"]
            )
            
            # Update the history with final response
            history[-1] = (user_message, formatted_response)
            yield history
        else:
            # Stream the response without web search
            bot_response = ""
            for token in stream_response(user_message, language, accent):
                bot_response += token
                # Update the last response in the history
                history[-1] = (user_message, bot_response)
                yield history
            
            # Add source indicator for knowledge-based response
            final_response = format_response_with_sources(bot_response, False, [])
            history[-1] = (user_message, final_response)
            yield history
    
    def reset_thread():
        new_thread_id = str(uuid.uuid4())
        return [], new_thread_id
    
    # Connect the components
    msg.submit(
        fn=user, 
        inputs=[msg, chatbot, thread_id, language, accent, enable_web_search], 
        outputs=[msg, chatbot, thread_id]
    ).then(
        fn=bot,
        inputs=[chatbot, thread_id, language, accent, enable_web_search],
        outputs=chatbot
    )
    
    clear.click(
        fn=reset_thread,
        outputs=[chatbot, thread_id]
    )


if __name__ == "__main__":
    demo.launch(share=False)