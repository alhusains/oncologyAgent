"""LangChain-based ReAct ML Agent

This module provides a LangChain implementation of the ReAct ML agent that maintains
100% feature parity with the original implementation while adding LangChain capabilities.

Supports:
- Classification tasks
- Regression tasks  
- Survival analysis tasks
- All 8 ML tools
- Conversation logging
- Reflection variant
- State management
- Async execution
"""

import asyncio
import json
from typing import Dict, Any, List, Optional
from datetime import datetime

# LangChain imports with version compatibility
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.outputs import LLMResult

# Try to import LangGraph (LangChain 1.0+)
try:
    from langgraph.prebuilt import create_react_agent
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    # Try older LangChain agent imports
    try:
        from langchain.agents import AgentExecutor, create_openai_functions_agent
    except ImportError:
        pass

from ..core.config import Config
from .tools import MLToolkit
from .langchain_tools import create_langchain_tools


# ============================================================================
# LangChain Version Compatibility
# ============================================================================

def create_react_agent_compatible(llm, tools, prompt=None, system_message=None):
    """
    Create a ReAct agent compatible with both old and new LangChain versions.
    
    LangChain 1.0+ uses LangGraph's create_react_agent.
    Older versions use create_openai_functions_agent with AgentExecutor.
    
    Returns:
        For LangGraph: The agent executor (graph)
        For old LangChain: Tuple of (agent, needs_executor=True)
    """
    if LANGGRAPH_AVAILABLE:
        # LangChain 1.0+ with LangGraph
        # LangGraph's create_react_agent returns an executable graph (no separate executor needed)
        # Extract system message from prompt if provided
        system_prompt = None
        if system_message is None and prompt is not None:
            # Try to extract system message from prompt template
            if hasattr(prompt, 'messages') and len(prompt.messages) > 0:
                first_msg = prompt.messages[0]
                if hasattr(first_msg, 'prompt') and hasattr(first_msg.prompt, 'template'):
                    system_prompt = first_msg.prompt.template
        else:
            system_prompt = system_message
        
        # Create LangGraph ReAct agent
        # Note: LangGraph's create_react_agent has a simpler API than old LangChain
        # It primarily takes model and tools. System prompts are passed during invocation.
        agent_executor = create_react_agent(
            model=llm,
            tools=tools
        )
        return agent_executor, False  # False = doesn't need separate executor
    else:
        # Old LangChain with AgentExecutor
        try:
            from langchain.agents import create_openai_functions_agent
            agent = create_openai_functions_agent(llm=llm, tools=tools, prompt=prompt)
            return agent, True  # True = needs AgentExecutor wrapper
        except (ImportError, AttributeError) as e:
            raise ImportError(
                f"Could not import LangChain agent components. "
                f"Please upgrade: pip install --upgrade langchain langchain-openai langgraph"
            ) from e


# ============================================================================
# Custom Callbacks for Enhanced Functionality
# ============================================================================

class ConversationLoggerCallback(BaseCallbackHandler):
    """
    Callback to log all conversation steps for later retrieval.
    
    This ensures we can provide the same get_conversation_summary() and
    save_conversation() methods as the original implementation.
    """
    
    def __init__(self):
        self.messages = []
        self.iteration_count = 0
        self.tool_calls = []
        super().__init__()
    
    def on_agent_action(self, action: AgentAction, **kwargs) -> None:
        """Called when agent decides to take an action"""
        self.iteration_count += 1
        
        self.messages.append({
            "role": "assistant",
            "type": "action",
            "iteration": self.iteration_count,
            "tool": action.tool,
            "tool_input": action.tool_input,
            "log": action.log
        })
        
        self.tool_calls.append({
            "tool": action.tool,
            "input": action.tool_input,
            "iteration": self.iteration_count
        })
        
        print(f"\nAgent decided to: {action.tool}")
        print(f"   Arguments: {json.dumps(action.tool_input, indent=2)}")
    
    def on_tool_end(self, output: str, **kwargs) -> None:
        """Called when tool finishes execution"""
        # Convert output to string if needed
        if not isinstance(output, str):
            try:
                output_str = json.dumps(output)
            except:
                output_str = str(output)
        else:
            output_str = output
        
        self.messages.append({
            "role": "tool",
            "type": "observation",
            "content": output_str
        })
        
        # Parse output to show summary
        try:
            # Try to parse as JSON if it's a string
            if isinstance(output, str):
                result = json.loads(output)
            else:
                result = output
            
            if isinstance(result, dict):
                if result.get("success"):
                    print(f"   Success: {result.get('message', 'Success')}")
                else:
                    print(f"   Error: {result.get('error', 'Failed')}")
            else:
                # Show truncated output
                print(f"   Result: {str(output)[:100]}...")
        except Exception as e:
            # Safely handle any output type
            try:
                print(f"   Result: {str(output)[:100]}...")
            except:
                print(f"   Result received")
    
    def on_agent_finish(self, finish: AgentFinish, **kwargs) -> None:
        """Called when agent finishes"""
        self.messages.append({
            "role": "assistant",
            "type": "finish",
            "content": finish.return_values.get("output", ""),
            "log": finish.log
        })
        
        print(f"\nAgent finished!")
        print(f"Final message: {finish.return_values.get('output', '')[:200]}...")


class ReflectionCallback(BaseCallbackHandler):
    """
    Callback to add reflection after model evaluations.
    
    This replicates the behavior of ReActAgentWithReflection by injecting
    a reflection prompt when performance is below threshold.
    """
    
    def __init__(
        self,
        toolkit: MLToolkit,
        reflection_threshold: float = 0.75,
        agent_executor: Optional[Any] = None  # Changed from AgentExecutor for compatibility
    ):
        self.toolkit = toolkit
        self.reflection_threshold = reflection_threshold
        self.agent_executor = agent_executor
        self.last_tool = None
        self.should_reflect = False
        super().__init__()
    
    def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs) -> None:
        """Track which tool is being called"""
        self.last_tool = serialized.get("name", "")
    
    def on_tool_end(self, output: str, **kwargs) -> None:
        """Check if we should inject reflection after evaluation"""
        
        # Only reflect after evaluate_model
        if self.last_tool == "evaluate_model":
            current_state = self.toolkit.state
            best_score = current_state.get("best_score", 0)
            
            if best_score < self.reflection_threshold:
                self.should_reflect = True
                
                # Note: Actual reflection injection happens in the agent
                # This just sets the flag
                print(f"\nPerformance is {best_score:.3f} - considering improvements...")


# ============================================================================
# Main LangChain ReAct ML Agent
# ============================================================================

class LangChainReActMLAgent:
    """
    LangChain-based ReAct ML Agent with full feature parity.
    
    This agent uses LangChain's AgentExecutor internally but maintains
    the exact same interface as the original ReActMLAgent.
    
    Features:
    - Same run() method signature
    - Same return value structure  
    - Same conversation logging methods
    - Supports classification, regression, and survival tasks
    - All 8 tools accessible
    - Async execution
    - State management through toolkit
    """
    
    def __init__(self, config: Config):
        """
        Initialize the LangChain ReAct agent.
        
        Args:
            config: Configuration object with LLM and ML settings
        """
        self.config = config
        self.toolkit = MLToolkit(config)
        
        # Validate API key
        api_key = config.llm.api_key
        if not api_key:
            raise ValueError("OpenAI API key required for ReAct agent")
        
        # Determine temperature based on model
        # GPT-5 only supports default temperature of 1
        if config.llm.model.startswith("gpt-4.1-mini") or config.llm.model.startswith("gpt-5"):
            temperature = 1.0
        else:
            temperature = config.llm.temperature
        
        # Create LangChain LLM
        self.llm = ChatOpenAI(
            model=config.llm.model,
            temperature=temperature,
            api_key=api_key,
            model_kwargs={"stream": False}  # Disable streaming for now
        )
        
        # Convert tools to LangChain format
        self.tools = create_langchain_tools(self.toolkit)
        
        # Create agent prompt
        self.prompt, self.system_message = self._create_prompt()
        
        # Create callbacks
        self.conversation_logger = ConversationLoggerCallback()
        
        # Agent and executor will be created in run()
        self.agent = None
        self.executor = None
        
        # Execution state
        self.max_iterations = 20
        self.iteration_count = 0
    
    def _create_prompt(self) -> ChatPromptTemplate:
        """
        Create the ReAct-style prompt for the agent.
        
        This prompt maintains the same strategy and instructions as the
        original implementation.
        """
        
        system_message = """You are an expert ML engineer with access to tools for building ML models.

Your goal: Build the BEST possible ML model for the user's dataset and objective.

Available tools:
- analyze_data: Understand dataset structure and identify target
- engineer_features: Apply preprocessing and feature transformations
- select_models: Choose which models to train
- train_model: Train a specific model
- evaluate_model: Test model performance
- analyze_errors: Understand which predictions failed and why
- get_feature_importance: See which features matter most
- get_current_state: Check progress

STRATEGY for small datasets (<200 samples):
1. Start simple - use logistic regression or small random forest
2. Focus on feature quality over complex models
3. Use quick_mode=true for faster iterations
4. Analyze errors to guide feature engineering

STRATEGY for larger datasets:
1. Try multiple model types
2. Use error analysis to improve
3. Iterate based on performance

PROCESS:
1. Always start with analyze_data
2. Then engineer_features
3. Select and train models (start with simpler ones for small data)
4. Evaluate on test set
5. Get feature importance for the best model (helps understand what drives predictions)
6. If performance is poor (<0.7), analyze_errors and iterate
7. When satisfied or out of ideas, provide a final summary

Remember:
- Small datasets need simple models
- Always get feature importance to understand model behavior
- Check error analysis if performance needs improvement
- Stop when performance is good (>0.8) or you've tried reasonable approaches
- Be strategic about which models to try based on data size

IMPORTANT: When you're done, just provide a natural language summary. Do not try to call more tools."""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_message),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        return prompt, system_message
    
    async def run(
        self,
        dataset_path: str,
        objective: str,
        testset_path: Optional[str] = None,
        max_iterations: Optional[int] = None,
        use_preset_CV: bool = False
    ) -> Dict[str, Any]:
        """
        Run the ReAct agent to build an ML model.
        
        This method maintains 100% compatibility with the original ReActMLAgent.run()
        
        Args:
            dataset_path: Path to dataset file
            objective: ML objective description
            testset_path: Optional path to pre-split test set. If None, auto-splits
            max_iterations: Maximum iterations (default 20)
            use_preset_CV: If True, use CV column from dataset for cross-validation
            
        Returns:
            Dict with:
                - success: bool
                - iterations: int
                - best_model: str
                - best_score: float
                - trained_models: List[str]
                - conversation_history: List[Dict]
                - final_state: Dict
        """
        
        if max_iterations:
            self.max_iterations = max_iterations
        
        # Set config parameters
        self.config.data.use_preset_CV = use_preset_CV
        
        # Initialize toolkit state
        self.toolkit.state["dataset_path"] = dataset_path
        self.toolkit.state["testset_path"] = testset_path
        self.toolkit.state["objective"] = objective
        
        # Create agent (fresh for each run) - compatible with both LangGraph and old LangChain
        agent_result, needs_executor = create_react_agent_compatible(
            llm=self.llm,
            tools=self.tools,
            prompt=self.prompt
        )
        
        # Store for later use
        self.needs_executor = needs_executor
        
        if needs_executor:
            # Old LangChain: need to wrap agent in AgentExecutor
            from langchain.agents import AgentExecutor
            self.agent = agent_result
            self.executor = AgentExecutor(
                agent=self.agent,
                tools=self.tools,
                verbose=False,  # We handle our own printing
                max_iterations=self.max_iterations,
                max_execution_time=None,  # No time limit
                handle_parsing_errors=True,
                return_intermediate_steps=True,
                callbacks=[self.conversation_logger]
            )
        else:
            # LangGraph: agent_result is already the executor (graph)
            self.executor = agent_result
            self.agent = None  # Not used in LangGraph
        
        print("Starting LangChain ReAct ML Agent")
        print("=" * 70)
        print(f"Dataset: {dataset_path}")
        print(f"Objective: {objective}")
        print(f"Max iterations: {self.max_iterations}")
        print("=" * 70)
        
        # Prepare input
        input_text = f"""Dataset: {dataset_path}
Objective: {objective}
Test Set: {testset_path if testset_path else 'Auto-split from main dataset'}

Please build the best ML model for this task. Start by analyzing the data."""
        
        # Run agent
        try:
            if LANGGRAPH_AVAILABLE and not self.needs_executor:
                # LangGraph execution
                # LangGraph expects messages format with system message first
                messages = [
                    ("system", self.system_message),
                    ("user", input_text)
                ]
                result = await self.executor.ainvoke(
                    {"messages": messages},
                    config={"callbacks": [self.conversation_logger], "recursion_limit": self.max_iterations}
                )
            else:
                # Old LangChain execution
                result = await self.executor.ainvoke(
                    {"input": input_text},
                    config={"callbacks": [self.conversation_logger]}
                )
            
            # Get final state from toolkit
            final_state = self.toolkit.state
            
            # Update iteration count from logger
            self.iteration_count = self.conversation_logger.iteration_count
            
            print(f"\n{'='*70}")
            print("LangChain ReAct Agent Completed")
            print(f"{'='*70}")
            print(f"Total iterations: {self.iteration_count}")
            print(f"Best model: {final_state.get('best_model', 'None')}")
            print(f"Best score: {final_state.get('best_score', 0):.3f}")
            print(f"Models trained: {list(final_state.get('trained_models', {}).keys())}")
            print(f"{'='*70}")
            
            # Return in same format as original
            return {
                "success": True,
                "iterations": self.iteration_count,
                "best_model": final_state.get("best_model"),
                "best_score": final_state.get("best_score", 0.0),
                "trained_models": list(final_state.get("trained_models", {}).keys()),
                "conversation_history": self.conversation_logger.messages,
                "final_state": final_state
            }
            
        except Exception as e:
            print(f"\nError in LangChain ReAct agent: {str(e)}")
            import traceback
            traceback.print_exc()
            
            return {
                "success": False,
                "error": str(e),
                "iterations": self.conversation_logger.iteration_count,
                "final_state": self.toolkit.state
            }
    
    def get_conversation_summary(self) -> List[Dict[str, str]]:
        """
        Get a human-readable summary of the conversation.
        
        Returns same format as original implementation.
        
        Returns:
            List of dicts with 'role' and 'content' keys
        """
        
        summary = []
        
        for msg in self.conversation_logger.messages:
            msg_type = msg.get("type", "")
            
            if msg_type == "action":
                # Agent action
                tool = msg.get("tool", "unknown")
                tool_input = msg.get("tool_input", {})
                summary.append({
                    "role": "Agent",
                    "content": f"Action: {tool} with {json.dumps(tool_input)}"
                })
                
            elif msg_type == "observation":
                # Tool result
                content = msg.get("content", "")
                # Truncate long outputs
                if len(content) > 200:
                    content = content[:200] + "..."
                summary.append({
                    "role": "Tool",
                    "content": content
                })
                
            elif msg_type == "finish":
                # Final output
                content = msg.get("content", "")
                summary.append({
                    "role": "Agent",
                    "content": f"Final: {content}"
                })
        
        return summary
    
    def save_conversation(self, filepath: str):
        """
        Save full conversation to a JSON file.
        
        Same format as original implementation.
        
        Args:
            filepath: Path where to save the conversation
        """
        
        conversation_data = {
            "timestamp": datetime.now().isoformat(),
            "agent_type": "langchain_react",
            "iterations": self.conversation_logger.iteration_count,
            "messages": self.conversation_logger.messages,
            "tool_calls": self.conversation_logger.tool_calls,
            "final_state": {
                "best_model": self.toolkit.state.get("best_model"),
                "best_score": self.toolkit.state.get("best_score", 0.0),
                "trained_models": list(self.toolkit.state.get("trained_models", {}).keys()),
                "task_type": self.toolkit.state.get("feature_result", {}).get("task_type")
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(conversation_data, f, indent=2)
        
        print(f"Conversation saved to {filepath}")


# ============================================================================
# Reflection Variant
# ============================================================================

class LangChainReActAgentWithReflection(LangChainReActMLAgent):
    """
    Enhanced LangChain ReAct agent that reflects on performance after evaluations.
    
    This adds an extra reasoning step to decide on improvements when performance
    is below threshold, maintaining feature parity with the original
    ReActAgentWithReflection.
    """
    
    def __init__(self, config: Config):
        """
        Initialize the reflection agent.
        
        Args:
            config: Configuration object
        """
        super().__init__(config)
        
        # Will be set in run()
        self.reflection_callback = None
    
    async def run(
        self,
        dataset_path: str,
        objective: str,
        testset_path: Optional[str] = None,
        max_iterations: Optional[int] = None,
        use_preset_CV: bool = False
    ) -> Dict[str, Any]:
        """
        Run agent with reflection after evaluations.
        
        When a model evaluation shows score < 0.75, the agent is prompted to
        reflect on potential improvements.
        
        Args:
            Same as parent class
            
        Returns:
            Same as parent class
        """
        
        if max_iterations:
            self.max_iterations = max_iterations
        
        # Set config
        self.config.data.use_preset_CV = use_preset_CV
        self.toolkit.state["dataset_path"] = dataset_path
        self.toolkit.state["testset_path"] = testset_path
        self.toolkit.state["objective"] = objective
        
        # Create reflection callback
        self.reflection_callback = ReflectionCallback(
            toolkit=self.toolkit,
            reflection_threshold=0.75
        )
        
        # Create agent - compatible with both LangGraph and old LangChain
        agent_result, needs_executor = create_react_agent_compatible(
            llm=self.llm,
            tools=self.tools,
            prompt=self.prompt
        )
        
        # Store for later use
        self.needs_executor = needs_executor
        
        if needs_executor:
            # Old LangChain: need to wrap agent in AgentExecutor
            from langchain.agents import AgentExecutor
            self.agent = agent_result
            self.executor = AgentExecutor(
                agent=self.agent,
                tools=self.tools,
                verbose=False,
                max_iterations=self.max_iterations,
                handle_parsing_errors=True,
                return_intermediate_steps=True,
                callbacks=[self.conversation_logger, self.reflection_callback]
            )
        else:
            # LangGraph: agent_result is already the executor (graph)
            self.executor = agent_result
            self.agent = None  # Not used in LangGraph
        
        print("Starting LangChain ReAct ML Agent with Reflection")
        print("=" * 70)
        print(f"Dataset: {dataset_path}")
        print(f"Objective: {objective}")
        print(f"Max iterations: {self.max_iterations}")
        print(f"Reflection enabled (threshold: 0.75)")
        print("=" * 70)
        
        # Prepare input
        input_text = f"""Dataset: {dataset_path}
Objective: {objective}
Test Set: {testset_path if testset_path else 'Auto-split from main dataset'}

Please build the best ML model for this task. Start by analyzing the data."""
        
        # Run agent with reflection support
        try:
            # Initial run
            if LANGGRAPH_AVAILABLE and not self.needs_executor:
                # LangGraph execution with system message
                messages = [
                    ("system", self.system_message),
                    ("user", input_text)
                ]
                result = await self.executor.ainvoke(
                    {"messages": messages},
                    config={"callbacks": [self.conversation_logger, self.reflection_callback], "recursion_limit": self.max_iterations}
                )
            else:
                # Old LangChain execution
                result = await self.executor.ainvoke(
                    {"input": input_text},
                    config={"callbacks": [self.conversation_logger, self.reflection_callback]}
                )
            
            # Check if we should inject reflection
            if self.reflection_callback.should_reflect:
                best_score = self.toolkit.state.get("best_score", 0)
                
                reflection_prompt = f"""The current best score is {best_score:.3f}, which could be improved.

Based on the results so far:
1. Should we analyze errors to understand failures?
2. Should we try a different model?
3. Should we engineer new features?
4. Should we stop and accept this performance?

What's your next strategic move to improve the model?"""
                
                print(f"\n{'='*70}")
                print("REFLECTION PHASE")
                print(f"{'='*70}")
                
                # Continue with reflection
                result = await self.executor.ainvoke(
                    {"input": reflection_prompt},
                    config={"callbacks": [self.conversation_logger, self.reflection_callback]}
                )
            
            # Get final state
            final_state = self.toolkit.state
            self.iteration_count = self.conversation_logger.iteration_count
            
            print(f"\n{'='*70}")
            print("LangChain ReAct Agent with Reflection Completed")
            print(f"{'='*70}")
            print(f"Total iterations: {self.iteration_count}")
            print(f"Best model: {final_state.get('best_model', 'None')}")
            print(f"Best score: {final_state.get('best_score', 0):.3f}")
            print(f"Models trained: {list(final_state.get('trained_models', {}).keys())}")
            print(f"{'='*70}")
            
            return {
                "success": True,
                "iterations": self.iteration_count,
                "best_model": final_state.get("best_model"),
                "best_score": final_state.get("best_score", 0.0),
                "trained_models": list(final_state.get("trained_models", {}).keys()),
                "conversation_history": self.conversation_logger.messages,
                "final_state": final_state
            }
            
        except Exception as e:
            print(f"\nError in LangChain ReAct agent with reflection: {str(e)}")
            import traceback
            traceback.print_exc()
            
            return {
                "success": False,
                "error": str(e),
                "iterations": self.conversation_logger.iteration_count,
                "final_state": self.toolkit.state
            }

