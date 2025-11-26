"""
Conversational ML Agent

A multi-turn conversational agent for interactive ML pipeline execution.
Supports step-by-step execution with persistent state across conversation turns.
"""

import uuid
import json
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from pathlib import Path

# LangChain imports
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.callbacks import BaseCallbackHandler

# Try LangGraph first
try:
    from langgraph.prebuilt import create_react_agent
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    from langchain.agents import AgentExecutor, create_openai_functions_agent

from ..core.config import Config
from .tools import MLToolkit
from .langchain_tools import create_langchain_tools


class SimpleCallbackHandler(BaseCallbackHandler):
    """Simple callback to show tool execution"""
    
    def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs):
        """Called when tool starts"""
        tool_name = serialized.get("name", "unknown")
        print(f"   🔧 Calling tool: {tool_name}")
    
    def on_tool_end(self, output: str, **kwargs):
        """Called when tool ends"""
        print(f"   ✅ Tool completed")
    
    def on_tool_error(self, error: Exception, **kwargs):
        """Called when tool errors"""
        print(f"   ❌ Tool error: {str(error)}")


class ConversationalMLAgent:
    """
    Conversational ML Agent with multi-turn support and persistent state.
    
    Features:
    - Multi-turn conversation (like ChatGPT)
    - Step-by-step execution control (user decides what to do)
    - Persistent state across conversation turns
    - All ML tools plus new data insights and interpretability tools
    - Session management (save/load conversations)
    
    Usage:
        agent = ConversationalMLAgent(config)
        
        # Set dataset (once at start)
        agent.set_dataset("data/train.csv", "data/test.csv")
        
        # Chat in multiple turns
        response = await agent.chat("Give me data insights")
        response = await agent.chat("Now implement feature engineering")
        response = await agent.chat("Train a random forest model")
        response = await agent.chat("Generate interpretability report")
    """
    
    def __init__(self, config: Config):
        """
        Initialize conversational agent.
        
        Args:
            config: Configuration object with LLM and ML settings
        """
        self.config = config
        self.toolkit = MLToolkit(config)
        
        # Validate API key
        api_key = config.llm.api_key
        if not api_key:
            raise ValueError("OpenAI API key required for ConversationalMLAgent")
        
        # Determine temperature based on model
        if config.llm.model.startswith("gpt-4.1-mini") or config.llm.model.startswith("gpt-5"):
            temperature = 1.0
        else:
            temperature = config.llm.temperature
        
        # Create LangChain LLM
        self.llm = ChatOpenAI(
            model=config.llm.model,
            temperature=temperature,
            api_key=api_key,
            model_kwargs={"stream": False}
        )
        
        # Convert tools to LangChain format
        self.tools = create_langchain_tools(self.toolkit)
        
        # Create system prompt for step-by-step execution
        self.system_message = self._create_system_message()
        
        # Session management
        self.session_id = str(uuid.uuid4())
        self.message_history = []
        self.created_at = datetime.now()
        
        # Agent executor (created fresh for each chat)
        self.executor = None
        
        print(f"✅ Conversational ML Agent initialized")
        print(f"   Session ID: {self.session_id}")
        print(f"   Model: {config.llm.model}")
        print(f"   Tools available: {len(self.tools)}")
        print(f"   LangGraph available: {LANGGRAPH_AVAILABLE}")
    
    def _create_system_message(self) -> str:
        """
        Create system prompt for conversational, step-by-step execution.
        
        Key differences from autonomous agent:
        - Execute ONLY what user requests
        - Don't assume full pipeline execution
        - Wait for user instruction after each step
        - Be conversational and helpful
        """
        return """You are an expert ML assistant for interactive data analysis and model building.

IMPORTANT BEHAVIOR:
- Execute ONLY what the user requests in their message
- DO NOT assume the user wants to complete the full ML pipeline
- After completing a task, provide a summary and WAIT for the user's next instruction
- Be conversational, helpful, and clear in your responses

Available tools:
1. analyze_data - Understand dataset structure and identify target
2. engineer_features - Apply preprocessing and feature engineering
3. select_models - Get model recommendations for the task
4. train_model - Train a specific model
5. evaluate_model - Test model performance on test set
6. analyze_errors - Deep dive into prediction errors
7. get_feature_importance - See which features drive predictions
8. get_current_state - Check what's been done so far
9. get_data_insights - Comprehensive data analysis and statistics
10. generate_interpretability_report - Create PDF report with SHAP values and clinical guidance

EXAMPLE INTERACTIONS:

User: "Give me data insights"
You: Call get_data_insights, provide summary, STOP and wait

User: "Implement feature engineering"
You: Call engineer_features, report results, STOP and wait

User: "Now train a random forest model"
You: Call train_model('random_forest'), report CV score, STOP and wait

User: "Evaluate it on test set"
You: Call evaluate_model, show metrics, STOP and wait

User: "Generate interpretability report"
You: Call generate_interpretability_report, provide report path, STOP

User: "Conduct full classification analysis"
You: Execute full pipeline (analyze → engineer → select → train → evaluate)

GUIDELINES:
1. If user asks for specific step → do ONLY that step
2. If user asks for "full analysis" or "complete pipeline" → do all steps
3. Always provide clear, concise summaries after tool calls
4. For data insights, present key findings in readable format
5. For model training, highlight important metrics
6. For interpretability, emphasize clinical implications
7. Ask clarifying questions if the request is ambiguous

Remember: You're an interactive assistant, not an autonomous agent. Follow the user's lead!"""
    
    def set_dataset(
        self,
        dataset_path: str,
        testset_path: Optional[str] = None,
        objective: Optional[str] = None,
        use_preset_CV: bool = False
    ):
        """
        Set the dataset for this session.
        
        This should be called once at the start before chatting.
        
        Args:
            dataset_path: Path to training dataset
            testset_path: Optional path to test dataset
            objective: Optional ML objective description
            use_preset_CV: Whether to use preset CV column from dataset
        """
        self.toolkit.state["dataset_path"] = dataset_path
        self.toolkit.state["testset_path"] = testset_path
        self.toolkit.state["objective"] = objective or "ML analysis"
        self.config.data.use_preset_CV = use_preset_CV
        
        print(f"\n📂 Dataset configured:")
        print(f"   Training: {dataset_path}")
        if testset_path:
            print(f"   Testing: {testset_path}")
        print(f"   Objective: {objective or 'Not specified'}")
    
    async def chat(self, user_message: str) -> str:
        """
        Process a single user message and return agent response.
        
        State persists across multiple chat() calls in the same session.
        
        Args:
            user_message: User's message/request
            
        Returns:
            Agent's response as a string
        """
        # Add user message to history
        self.message_history.append(HumanMessage(content=user_message))
        
        # Inject dataset context if configured
        dataset_context = ""
        if self.toolkit.state.get("dataset_path"):
            dataset_context = f"\n\nIMPORTANT CONTEXT:\n"
            dataset_context += f"- Training dataset: {self.toolkit.state['dataset_path']}\n"
            if self.toolkit.state.get("testset_path"):
                dataset_context += f"- Test dataset: {self.toolkit.state['testset_path']}\n"
            dataset_context += f"- Objective: {self.toolkit.state.get('objective', 'Not specified')}\n"
            dataset_context += f"\nWhen calling analyze_data, use the training dataset path above."
        
        # Create agent executor (fresh for each turn to avoid state issues)
        if LANGGRAPH_AVAILABLE:
            # LangGraph approach
            self.executor = create_react_agent(
                model=self.llm,
                tools=self.tools
            )
            
            # Prepare messages with system message + dataset context
            system_msg = self.system_message + dataset_context
            messages = [SystemMessage(content=system_msg)] + self.message_history
            
            # Execute
            print("   [Processing your request...]", flush=True)
            
            # Create callback handler to show tool execution
            callback = SimpleCallbackHandler()
            
            result = await self.executor.ainvoke(
                {"messages": messages},
                config={
                    "recursion_limit": 30,
                    "callbacks": [callback]
                }
            )
            
            # Extract response from messages
            # LangGraph returns all messages including tool calls
            response_messages = result.get("messages", [])
            
            if not response_messages:
                response_text = "No response generated"
            else:
                # Debug: print message types (can be removed later)
                # print(f"\n   [DEBUG: Got {len(response_messages)} messages]")
                # for i, msg in enumerate(response_messages[-5:]):  # Last 5 messages
                #     print(f"   [DEBUG: Message {i}: type={type(msg).__name__}, has_content={hasattr(msg, 'content')}]")
                
                # LangGraph returns AIMessage objects
                # We want the LAST AIMessage that has content but NO tool_calls
                response_text = None
                
                # Iterate from the end to find the final response
                for msg in reversed(response_messages):
                    msg_type = type(msg).__name__
                    
                    # Check if it's an AI message (not Human, not System, not Tool)
                    if msg_type == 'AIMessage':
                        # Check if it has tool_calls (if so, it's a tool invocation, not a response)
                        if hasattr(msg, 'tool_calls') and msg.tool_calls:
                            continue  # Skip tool call messages
                        
                        # Check if it has content
                        if hasattr(msg, 'content') and msg.content:
                            content = msg.content
                            if isinstance(content, str) and len(content.strip()) > 0:
                                response_text = content.strip()
                                break
                
                # If still no response, try to find ANY AI message with content (excluding Human messages explicitly)
                if not response_text:
                    for msg in reversed(response_messages):
                        msg_type = type(msg).__name__
                        
                        # Skip human messages explicitly
                        if msg_type in ['HumanMessage', 'SystemMessage', 'ToolMessage']:
                            continue
                        
                        if hasattr(msg, 'content') and msg.content:
                            content = str(msg.content)
                            if len(content.strip()) > 0:
                                response_text = content.strip()
                                break
                
                # Last resort
                if not response_text:
                    response_text = "I processed your request but couldn't generate a text response. The tools may have executed successfully - check above for any tool output."
        
        else:
            # Old LangChain approach
            from langchain.agents import AgentExecutor, create_openai_functions_agent
            
            print("   [Using classic LangChain agent]", flush=True)
            
            # Create prompt with message history + dataset context
            system_msg = self.system_message + dataset_context
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_msg),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}")
            ])
            
            # Create agent
            agent = create_openai_functions_agent(
                llm=self.llm,
                tools=self.tools,
                prompt=prompt
            )
            
            # Create callback handler
            callback = SimpleCallbackHandler()
            
            # Create executor
            executor = AgentExecutor(
                agent=agent,
                tools=self.tools,
                verbose=False,
                max_iterations=20,
                handle_parsing_errors=True
            )
            
            # Execute with history
            print("   [Processing your request...]", flush=True)
            result = await executor.ainvoke(
                {
                    "input": user_message,
                    "chat_history": self.message_history[:-1]  # Exclude current message
                },
                config={"callbacks": [callback]}
            )
            
            response_text = result.get("output", "No response generated")
        
        # Add response to history
        self.message_history.append(AIMessage(content=response_text))
        
        return response_text
    
    def get_conversation_history(self) -> List[Dict[str, str]]:
        """
        Get the conversation history in a readable format.
        
        Returns:
            List of message dictionaries with 'role' and 'content'
        """
        history = []
        for msg in self.message_history:
            if isinstance(msg, HumanMessage):
                history.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                history.append({"role": "assistant", "content": msg.content})
        return history
    
    def get_state_summary(self) -> Dict[str, Any]:
        """
        Get summary of current ML pipeline state.
        
        Returns:
            Dictionary with current progress and results
        """
        state = self.toolkit.state
        
        return {
            "session_id": self.session_id,
            "created_at": self.created_at.isoformat(),
            "dataset_configured": state.get("dataset_path") is not None,
            "data_analyzed": state.get("data_analysis") is not None,
            "features_engineered": state.get("feature_result") is not None,
            "models_trained": list(state.get("trained_models", {}).keys()),
            "best_model": state.get("best_model"),
            "best_score": state.get("best_score", 0.0),
            "conversation_turns": len(self.message_history) // 2
        }
    
    def save_session(self, filepath: Optional[str] = None) -> str:
        """
        Save the current session to a JSON file.
        
        Saves:
        - Conversation history
        - ML pipeline state
        - Session metadata
        
        Args:
            filepath: Path to save file (if None, auto-generated)
            
        Returns:
            Path to saved file
        """
        if filepath is None:
            output_dir = Path("outputs/chat_sessions")
            output_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = str(output_dir / f"session_{self.session_id[:8]}_{timestamp}.json")
        
        session_data = {
            "session_id": self.session_id,
            "created_at": self.created_at.isoformat(),
            "saved_at": datetime.now().isoformat(),
            "conversation_history": self.get_conversation_history(),
            "state_summary": self.get_state_summary(),
            "toolkit_state": {
                "dataset_path": self.toolkit.state.get("dataset_path"),
                "testset_path": self.toolkit.state.get("testset_path"),
                "objective": self.toolkit.state.get("objective"),
                "best_model": self.toolkit.state.get("best_model"),
                "best_score": self.toolkit.state.get("best_score", 0.0),
                "trained_models": list(self.toolkit.state.get("trained_models", {}).keys()),
                "task_type": self.toolkit.state.get("feature_result", {}).get("task_type")
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(session_data, f, indent=2, default=str)
        
        print(f"\n💾 Session saved: {filepath}")
        return filepath
    
    def reset_session(self):
        """
        Reset the conversation and start fresh.
        
        Clears conversation history but keeps dataset configuration.
        """
        dataset_path = self.toolkit.state.get("dataset_path")
        testset_path = self.toolkit.state.get("testset_path")
        objective = self.toolkit.state.get("objective")
        
        # Clear conversation
        self.message_history = []
        self.session_id = str(uuid.uuid4())
        self.created_at = datetime.now()
        
        # Clear ML state but keep dataset config
        self.toolkit.state = {
            "dataset_path": dataset_path,
            "testset_path": testset_path,
            "objective": objective,
            "data_analysis": None,
            "feature_result": None,
            "trained_models": {},
            "evaluation_results": {},
            "error_analyses": {},
            "best_score": 0.0,
            "best_model": None
        }
        
        print(f"\n🔄 Session reset. New session ID: {self.session_id}")
    
    def print_summary(self):
        """Print a nice summary of the current session"""
        summary = self.get_state_summary()
        
        print("\n" + "="*70)
        print("SESSION SUMMARY")
        print("="*70)
        print(f"Session ID: {summary['session_id']}")
        print(f"Started: {summary['created_at']}")
        print(f"Conversation Turns: {summary['conversation_turns']}")
        print()
        print(f"Dataset Configured: {'✅' if summary['dataset_configured'] else '❌'}")
        print(f"Data Analyzed: {'✅' if summary['data_analyzed'] else '❌'}")
        print(f"Features Engineered: {'✅' if summary['features_engineered'] else '❌'}")
        print(f"Models Trained: {len(summary['models_trained'])}")
        
        if summary['models_trained']:
            print(f"   Models: {', '.join(summary['models_trained'])}")
        
        if summary['best_model']:
            print(f"Best Model: {summary['best_model']}")
            print(f"Best Score: {summary['best_score']:.3f}")
        
        print("="*70)

