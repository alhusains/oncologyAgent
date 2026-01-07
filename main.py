"""
Oncology ML Agent - Interactive Machine Learning Pipeline

An AI-powered conversational agent for automated machine learning analysis
of oncology datasets. Supports classification, regression, and survival analysis
with comprehensive interpretability reporting.

Includes ACE (Agentic Context Engineering) framework for self-improvement.
"""

import asyncio
import sys
import os
from pathlib import Path
import warnings

warnings.filterwarnings('ignore', category=UserWarning, module='pydantic')

sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.core.config import Config
from src.agents import ConversationalMLAgent

# Try to import ACE agent
try:
    from src.agents.ace_agent import ACEMLAgent
    ACE_AVAILABLE = True
except ImportError:
    ACE_AVAILABLE = False
    ACEMLAgent = None


async def main():
    """Interactive ML agent for oncology data analysis"""
    
    print("=" * 70)
    print("Oncology ML Agent - Interactive Pipeline")
    print("=" * 70)
    print("An AI assistant for automated machine learning analysis.")
    print("Supports: Classification, Regression, and Survival Analysis")
    print("=" * 70)
    
    # Get API key
    api_key = input("\nEnter your OpenAI API key: ").strip()
    if not api_key:
        print("Error: API key required")
        return
    
    # Setup configuration
    os.environ["OPENAI_API_KEY"] = api_key
    config = Config.from_env()
    config.llm.model = "gpt-4o-mini"
    
    print(f"\nConfiguration:")
    print(f"  LLM Model: {config.llm.model}")
    print(f"  Output Directory: {config.output_dir}")
    
    # Choose agent type
    use_ace = config.ace.enabled and ACE_AVAILABLE
    if ACE_AVAILABLE:
        ace_choice = input(f"\nUse ACE framework for self-improvement? (Y/n, default: {'Y' if use_ace else 'n'}): ").strip().lower()
        if ace_choice in ['n', 'no']:
            use_ace = False
        elif ace_choice in ['', 'y', 'yes']:
            use_ace = True
    
    # Initialize agent
    if use_ace:
        print("\nInitializing ACE-enhanced agent (with self-improvement)...")
        agent = ACEMLAgent(config)
    else:
        print("\nInitializing conversational agent (standard mode)...")
        agent = ConversationalMLAgent(config)
    
    # Configure dataset
    print("\n" + "=" * 70)
    print("Dataset Configuration")
    print("=" * 70)
    
    dataset_path = input("Training dataset path: ").strip()
    if not dataset_path:
        print("Error: Training dataset path required")
        return
    
    if not os.path.exists(dataset_path):
        print(f"Error: File not found: {dataset_path}")
        return
    
    testset_path = input("Test dataset path (optional, press Enter to skip): ").strip()
    if testset_path and not os.path.exists(testset_path):
        print(f"Warning: Test set not found: {testset_path}")
        print("Will use automatic train/test split instead")
        testset_path = None
    elif not testset_path:
        testset_path = None
    
    objective = input("Analysis objective (e.g., 'Survival analysis', 'Classification'): ").strip()
    if not objective:
        objective = "Machine learning analysis"
    
    agent.set_dataset(dataset_path, testset_path, objective)
    
    print("\n" + "=" * 70)
    print("Interactive Session Started")
    print("=" * 70)
    print("\nAvailable commands:")
    print("  - Natural language requests (e.g., 'analyze the data', 'train models')")
    print("  - 'summary' - View current progress")
    print("  - 'save' - Save session")
    if use_ace:
        print("  - 'improve' - Run self-improvement loop (ACE)")
        print("  - 'playbook' - View learned knowledge (ACE)")
    print("  - 'exit' - End session")
    print("=" * 70)
    
    # Main interaction loop
    while True:
        print()
        user_input = input("You: ").strip()
        
        if not user_input:
            continue
        
        # Handle commands
        if user_input.lower() in ['exit', 'quit', 'bye']:
            print("\nSaving session...")
            agent.save_session()
            print("Session ended.")
            break
        
        if user_input.lower() == 'summary':
            agent.print_summary()
            continue
        
        if user_input.lower() == 'save':
            agent.save_session()
            continue
        
        if user_input.lower() == 'reset':
            confirm = input("Reset session? This will clear all progress (y/n): ").strip().lower()
            if confirm == 'y':
                agent.reset_session()
                print("Session reset")
            continue
        
        # Process user request
        try:
            print("\nAgent: ", end="", flush=True)
            response = await agent.chat(user_input)
            print(response)
        except KeyboardInterrupt:
            print("\n\nInterrupted. Type 'exit' to end session.")
        except Exception as e:
            print(f"\nError: {str(e)}")
            print("Please try again or type 'exit' to end session.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nSession terminated by user.")
    except Exception as e:
        print(f"\nFatal error: {str(e)}")
        sys.exit(1)
