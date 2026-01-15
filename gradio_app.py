#!/usr/bin/env python3
"""
Gradio Web Interface for Oncology ML Agent

A web-based conversational interface for the Oncology ML Agent,
providing an intuitive chat interface similar to ChatGPT for
interactive machine learning analysis.
"""

import gradio as gr
import asyncio
import os
import io
import sys
from pathlib import Path
from contextlib import redirect_stdout
import warnings
import pandas as pd

warnings.filterwarnings('ignore', category=UserWarning, module='pydantic')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.core.config import Config
from src.agents import ConversationalMLAgent

# Global agent instance (persists across conversations in the same session)
agent = None
config = None


async def initialize_agent(api_key, objective, model_choice):
    """Initialize the agent (without dataset)"""
    global agent, config
    
    if not api_key:
        return "❌ Error: Please provide an OpenAI API key"
    
    try:
        # Setup configuration
        os.environ["OPENAI_API_KEY"] = api_key
        config = Config.from_env()
        config.llm.model = model_choice
        
        # Create agent (without dataset yet)
        agent = ConversationalMLAgent(config)
        
        # Store objective for later use
        agent._pending_objective = objective.strip() if objective and objective.strip() else "Machine learning analysis"
        
        return f"""✅ Agent Initialized Successfully!

Configuration:
   • Model: {model_choice}
   • Objective: {agent._pending_objective}

Next Step: Go to the Chat tab and upload your dataset(s) using the "Upload Data" button
"""
    except Exception as e:
        return f"❌ Error initializing agent:\n{str(e)}"


async def handle_file_upload(train_file, test_file):
    """Handle dataset file uploads"""
    global agent
    
    if agent is None:
        return "❌ Please initialize the agent first in the Configuration tab"
    
    if train_file is None:
        return "❌ Please upload at least a training dataset"
    
    try:
        # Get file paths (Gradio uploads to temp directory)
        train_path = train_file.name if hasattr(train_file, 'name') else train_file
        test_path = test_file.name if test_file and hasattr(test_file, 'name') else None
        
        # Validate files can be read
        train_df = pd.read_csv(train_path) if train_path.endswith('.csv') else pd.read_excel(train_path)
        
        if test_path:
            test_df = pd.read_csv(test_path) if test_path.endswith('.csv') else pd.read_excel(test_path)
        
        # Set dataset in agent
        objective = getattr(agent, '_pending_objective', 'Machine learning analysis')
        agent.set_dataset(train_path, test_path, objective)
        
        return f"""✅ Dataset Uploaded Successfully!

Training Dataset:
   • File: {Path(train_path).name}
   • Shape: {train_df.shape[0]} rows × {train_df.shape[1]} columns

{f'''Test Dataset:
   • File: {Path(test_path).name}
   • Shape: {test_df.shape[0]} rows × {test_df.shape[1]} columns
''' if test_path else 'Test Dataset: Not provided (will use train/test split)'}

Objective: {objective}

You can now start chatting! Try: "Give me data insights"
"""
    except Exception as e:
        return f"❌ Error loading dataset:\n{str(e)}\n\nSupported formats: CSV (.csv) and Excel (.xlsx)"


async def chat_with_agent(message, history):
    """Handle chat messages"""
    global agent
    
    if agent is None:
        yield history + [{"role": "assistant", "content": "⚠️ Please initialize the agent first using the Configuration tab"}]
        return
    
    if not message or not message.strip():
        yield history
        return
    
    # Add user message to history
    history.append({"role": "user", "content": message})
    yield history
    
    # Add "thinking" indicator immediately
    history.append({"role": "assistant", "content": "🤔 Analyzing your request..."})
    yield history
    
    try:
        # Handle special commands
        if message.lower().strip() == 'summary':
            # Update thinking message
            history[-1] = {"role": "assistant", "content": "📊 Generating summary..."}
            yield history
            
            # Capture summary output
            f = io.StringIO()
            with redirect_stdout(f):
                agent.print_summary()
            response = f.getvalue()
            
            # Format for better display
            response = "**Session Summary**\n\n" + response
        
        elif message.lower().strip() == 'save':
            history[-1] = {"role": "assistant", "content": "💾 Saving session..."}
            yield history
            
            filepath = agent.save_session()
            response = f"Session saved successfully!\n\nFile: `{filepath}`"
        
        elif message.lower().strip() == 'reset':
            history[-1] = {"role": "assistant", "content": "🔄 Resetting session..."}
            yield history
            
            agent.reset_session()
            response = "Session has been reset. All progress cleared, but dataset configuration is preserved."
        
        else:
            # Regular chat - show working status
            history[-1] = {"role": "assistant", "content": "⚙️ Working on it...\n\n_Calling tools and processing your request..._"}
            yield history
            
            # Get the actual response
            response = await agent.chat(message)
        
        # Replace thinking message with actual response
        history[-1] = {"role": "assistant", "content": response}
        yield history
        
    except Exception as e:
        error_msg = f"❌ Error: {str(e)}\n\nPlease try rephrasing your request or check the console for more details."
        history[-1] = {"role": "assistant", "content": error_msg}
        yield history


def clear_chat():
    """Clear the chat history"""
    return []


# Custom CSS for better styling
custom_css = """
#config-section {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 20px;
    border-radius: 10px;
    color: white;
}

#chat-section {
    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    padding: 20px;
    border-radius: 10px;
}

.message {
    padding: 10px;
    border-radius: 8px;
    margin: 5px 0;
}

.gradio-container {
    max-width: 1200px !important;
}

footer {
    display: none !important;
}
"""

# Build Gradio interface
with gr.Blocks(
    title="OncologyAgent",
    theme=gr.themes.Soft(
        primary_hue="purple",
        secondary_hue="blue",
    ),
    css=custom_css
) as demo:
    
    # Header
    gr.Markdown("""
    # OncologyAgent - Interactive Web Interface
    
    An AI-powered conversational assistant for automated machine learning analysis of oncology datasets.
    Supports classification, regression, and survival analysis with comprehensive interpretability reporting.
    """)
    
    with gr.Tabs() as tabs:
        # Configuration Tab
        with gr.Tab("Configuration", id=0):
            gr.Markdown("""
            ### Agent Setup
            Configure your agent by providing your API key and analysis objective.
            
            **Note:** You'll upload your dataset files in the Chat tab after initialization.
            """)
            
            with gr.Row():
                with gr.Column(scale=2):
                    api_key_input = gr.Textbox(
                        label="OpenAI API Key",
                        type="password",
                        placeholder="sk-...",
                        info="Your OpenAI API key (kept secure in your session)"
                    )
                    
                    objective_input = gr.Textbox(
                        label="Analysis Objective",
                        value="Classification analysis for cancer survival prediction",
                        placeholder="e.g., Survival analysis, Classification, Regression...",
                        info="Brief description of your ML objective",
                        lines=2
                    )
                    
                    model_dropdown = gr.Dropdown(
                        choices=["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-4"],
                        value="gpt-4o-mini",
                        label="Model",
                        info="Select the LLM model to use"
                    )
                
                with gr.Column(scale=1):
                    gr.Markdown("""
                    **Quick Start:**
                    1. Enter your API key
                    2. Describe your objective
                    3. Click Initialize
                    4. Go to Chat tab
                    5. Upload your data files
                    6. Start chatting!
                    
                    **Models:**
                    - `gpt-4o-mini`: Fast & cheap
                    - `gpt-4o`: Balanced
                    - `gpt-4-turbo`: High capability
                    """)
            
            with gr.Row():
                init_btn = gr.Button("Initialize Agent", variant="primary", size="lg")
            
            status_output = gr.Textbox(
                label="Status",
                interactive=False,
                lines=6
            )
            
            init_btn.click(
                fn=initialize_agent,
                inputs=[api_key_input, objective_input, model_dropdown],
                outputs=status_output
            )
        
        # Chat Tab
        with gr.Tab("Chat", id=1):
            gr.Markdown("""
            ### Conversational Interface
            """)
            
            # Upload Data Section
            with gr.Accordion("Upload Data", open=True):
                gr.Markdown("Upload your dataset files (CSV or Excel format)")
                
                with gr.Row():
                    train_file_upload = gr.File(
                        label="Training Dataset",
                        file_types=[".csv", ".xlsx"],
                        type="filepath"
                    )
                    test_file_upload = gr.File(
                        label="Test Dataset (Optional)",
                        file_types=[".csv", ".xlsx"],
                        type="filepath"
                    )
                
                upload_btn = gr.Button("Upload Data", variant="primary")
                upload_status = gr.Textbox(label="Upload Status", interactive=False, lines=4)
                
                upload_btn.click(
                    fn=handle_file_upload,
                    inputs=[train_file_upload, test_file_upload],
                    outputs=upload_status
                )
            
            gr.Markdown("---")
            
            chatbot = gr.Chatbot(
                type="messages",
                label="Agent Chat",
                height=400,
                show_label=False,
                avatar_images=(
                    None,  # User avatar (default)
                    "https://em-content.zobj.net/source/apple/391/robot_1f916.png"  # Bot avatar
                )
            )
            
            with gr.Row():
                msg_input = gr.Textbox(
                    label="Your message",
                    placeholder="Type your message here... (Press Enter to send, Shift+Enter for new line)",
                    lines=2,
                    max_lines=5,
                    scale=4,
                    show_label=False,
                    autofocus=True
                )
                send_btn = gr.Button("Send", variant="primary", scale=1)
            
            with gr.Row():
                clear_btn = gr.Button("Clear Chat", scale=1)
                gr.Button("💾 Save Session (type 'save' in chat)", scale=2, interactive=False)
                gr.Button("📊 View Summary (type 'summary' in chat)", scale=2, interactive=False)
            
            # Example prompts
            gr.Markdown("### Example Prompts")
            
            with gr.Row():
                gr.Examples(
                    examples=[
                        ["Give me comprehensive data insights about this dataset"],
                        ["Implement feature engineering for this dataset"],
                        ["Train a random forest classifier"],
                    ],
                    inputs=msg_input,
                    label="Data Analysis & Training"
                )
                
                gr.Examples(
                    examples=[
                        ["Evaluate the model on the test set"],
                        ["Generate an interpretability report"],
                        ["summary"],
                    ],
                    inputs=msg_input,
                    label="Evaluation & Commands"
                )
            
            gr.Markdown("""
            **Available Commands:**
            - Natural language requests: "analyze the data", "train a model", etc.
            - `summary` - View current progress and state
            - `save` - Save the session to file
            - `reset` - Reset session (keeps dataset config)
            
            **Typical Workflow:**
            1. "Give me data insights" → Get comprehensive data analysis
            2. "Implement feature engineering" → Preprocess and engineer features
            3. "Train a random forest model" → Train specific model
            4. "Evaluate on test set" → Get performance metrics
            5. "Generate interpretability report" → Create PDF report with SHAP values
            
            Or simply say: *"Conduct full classification analysis"* to run the entire pipeline!
            """)
            
            # Chat interaction
            # Handle Enter key (submit) - process message and clear input
            msg_input.submit(
                fn=chat_with_agent,
                inputs=[msg_input, chatbot],
                outputs=chatbot
            ).then(
                fn=lambda: gr.update(value=""),
                outputs=msg_input
            )
            
            # Handle Send button click - process message and clear input
            send_btn.click(
                fn=chat_with_agent,
                inputs=[msg_input, chatbot],
                outputs=chatbot
            ).then(
                fn=lambda: gr.update(value=""),
                outputs=msg_input
            )
            
            clear_btn.click(fn=clear_chat, outputs=chatbot)
        
        # Help Tab
        with gr.Tab("Help", id=2):
            gr.Markdown("""
            ## Getting Started
            
            ### 1. Configuration
            
            First, go to the **Configuration** tab and:
            1. Enter your OpenAI API key
            2. Describe your analysis objective
            3. Select your preferred model (gpt-4o-mini recommended for speed/cost)
            4. Click "Initialize Agent"
            
            ### 2. Upload Data
            
            Once initialized, go to the **Chat** tab and:
            1. Click "Upload Data" section
            2. Upload your training dataset (CSV or Excel)
            3. Optionally upload test dataset
            4. Click "Upload Data" button
            5. Wait for confirmation
            
            ### 3. Chat with Agent
            
            Now you can interact with the agent:
            
            **Step-by-Step Execution:**
            - "Give me data insights" - Get comprehensive data analysis
            - "Implement feature engineering" - Apply preprocessing
            - "Train a [model_name] model" - Train specific model
            - "Evaluate the model" - Test performance
            - "Generate interpretability report" - Create PDF with SHAP values
            
            **Full Pipeline:**
            - "Conduct full classification analysis" - Run entire pipeline
            - "Perform complete survival analysis" - End-to-end survival modeling
            
            **Session Management:**
            - `summary` - View current progress
            - `save` - Save session to file
            - `reset` - Clear progress but keep dataset config
            
            ### 4. Understanding Output
            
            The agent will:
            - Show tool execution status while working
            - Provide summaries and key findings
            - Display model performance metrics
            - Offer insights and recommendations
            - Generate interpretability reports in `outputs/interpretability/`
            
            ### 5. Tips
            
            - **Be specific**: "Train a random forest" vs. "Train a model"
            - **One step at a time**: The agent follows your lead
            - **Review summaries**: Use `summary` to check progress
            - **Save your work**: Sessions are saved to `outputs/chat_sessions/`
            
            ### 6. Supported Tasks
            
            - **Classification**: Binary or multi-class prediction
            - **Regression**: Continuous value prediction
            - **Survival Analysis**: Time-to-event modeling (Cox, RSF)
            
            ### 7. Supported File Formats
            
            - **CSV** (.csv) - Comma-separated values
            - **Excel** (.xlsx) - Excel spreadsheet
            
            Both training and test datasets should have the same format and column structure.
            
            ### 8. Output Files
            
            All outputs are saved to the `outputs/` directory:
            - `outputs/chat_sessions/` - Saved conversation sessions
            - `outputs/interpretability/` - PDF reports with SHAP analysis
            - `outputs/visualizations/` - Generated plots and figures
            - `outputs/models/` - Trained model files
            
            ### 9. Troubleshooting
            
            **Agent not responding?**
            - Check that you've initialized the agent in the Configuration tab
            - Verify your API key is correct
            - Ensure you've uploaded dataset files
            
            **Upload errors?**
            - Ensure files are in CSV (.csv) or Excel (.xlsx) format
            - Check that files have column headers
            - Verify files are not corrupted
            
            **Error messages?**
            - Read the error message carefully
            - Check dataset format (CSV or Excel with headers)
            - Ensure sufficient data (minimum 100 samples recommended)
            
            **Need help?**
            - Review the example prompts in the Chat tab
            - Check the console output for detailed logs
            """)
    
    # Footer
    gr.Markdown("""
    ---
    **Oncology ML Agent** | Built with LangChain, Gradio, and OpenAI
    """)


def main():
    """Launch the Gradio interface"""
    print("="*70)
    print("Oncology ML Agent - Gradio Web Interface")
    print("="*70)
    print("\nStarting web server...")
    print("Once started, open your browser to the URL shown below\n")
    
    demo.launch(
        server_name="0.0.0.0",  # Allows access from network
        server_port=7860,
        share=False,  # Set to True to create a public URL
        show_error=True,
        quiet=False
    )


if __name__ == "__main__":
    main()

