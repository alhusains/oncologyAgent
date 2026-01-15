"""
ACE-Enhanced Conversational ML Agent

Extends the ConversationalMLAgent with ACE (Agentic Context Engineering) capabilities:
- Trajectory tracking for all actions
- Self-improvement loops with ablation testing
- Playbook-informed decision making
- Automatic reflection and learning
"""

import uuid
from typing import Dict, Any, List, Optional
from datetime import datetime

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

try:
    from langgraph.prebuilt import create_react_agent
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False

from ..core.config import Config
from .conversational_agent import ConversationalMLAgent, SimpleCallbackHandler
from .tools import MLToolkit
from .langchain_tools import create_langchain_tools

# ACE imports
from ..ace.generator import TrajectoryGenerator, ImprovementExperimentTracker
from ..ace.reflector import TrajectoryReflector
from ..ace.curator import PlaybookCurator
from ..ace.controller import ImprovementController
from ..ace.schemas import ActionType, Trajectory


class ACEMLAgent(ConversationalMLAgent):
    """
    ACE-Enhanced Conversational ML Agent.
    
    This agent extends ConversationalMLAgent with:
    - Trajectory tracking for learning from experience
    - Self-improvement loop (user can ask agent to improve itself)
    - Playbook of learned strategies that informs decisions
    - Automatic reflection after experiments
    
    Usage:
        agent = ACEMLAgent(config)
        agent.set_dataset("data/train.csv", "data/test.csv")
        
        # Regular chat - with playbook-informed responses
        response = await agent.chat("Analyze the data and train a model")
        
        # Self-improvement - agent iterates to improve
        response = await agent.chat("Try to improve the model performance")
        
        # View learned knowledge
        agent.print_playbook_summary()
    """
    
    def __init__(self, config: Config):
        """Initialize ACE-enhanced agent"""
        # Set ACE flag BEFORE parent init (parent calls _create_system_message which needs this)
        self.ace_enabled = config.ace.enabled
        self.curator = None  # Will be initialized after parent if ACE is enabled
        
        # Initialize base agent (this calls _create_system_message)
        super().__init__(config)
        
        # Now initialize ACE components
        if self.ace_enabled:
            # Create ACE components
            self.trajectory_generator = TrajectoryGenerator()
            self.curator = PlaybookCurator(
                playbook_path=config.ace.playbook_path,
                auto_save=config.ace.auto_save_playbook
            )
            self.reflector = TrajectoryReflector(llm_client=self._create_llm_for_reflection())
            self.improvement_controller = ImprovementController(
                generator=self.trajectory_generator,
                reflector=self.reflector,
                curator=self.curator,
                llm_client=self._create_llm_for_reflection(),
                config={
                    "max_improvement_iterations": config.ace.max_improvement_iterations,
                    "min_improvement_threshold": config.ace.min_improvement_threshold,
                    "max_changes_per_iteration": config.ace.max_changes_per_iteration
                }
            )
            
            # Link trajectory generator to toolkit for tracking
            self.toolkit.trajectory_generator = self.trajectory_generator
            
            # State tracking
            self._current_trajectory: Optional[Trajectory] = None
            self._actions_since_reflection = 0
            self._reflection_threshold = config.ace.reflection_threshold
            self._baseline_established = False
            
            # Regenerate system message now that curator is initialized
            self.system_message = self._create_system_message()
            
            print(f"   ACE Framework: Enabled")
            print(f"   Playbook items: {self.curator.playbook.total_items}")
        else:
            print(f"   ACE Framework: Disabled")
    
    def _create_llm_for_reflection(self):
        """Create a simple LLM client for ACE reflection"""
        class SimpleLLMClient:
            def __init__(self, llm):
                self.llm = llm
            
            async def complete_json(self, prompt: str, system_message: str = None) -> Dict[str, Any]:
                import json
                messages = []
                if system_message:
                    messages.append(SystemMessage(content=system_message))
                messages.append(HumanMessage(content=prompt))
                
                response = await self.llm.ainvoke(messages)
                
                # Extract JSON from response
                content = response.content
                try:
                    # Try to parse JSON directly
                    return json.loads(content)
                except json.JSONDecodeError:
                    # Try to extract JSON from markdown code block
                    import re
                    json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', content)
                    if json_match:
                        return json.loads(json_match.group(1))
                    # Last resort: try to find JSON object
                    json_match = re.search(r'\{[\s\S]*\}', content)
                    if json_match:
                        return json.loads(json_match.group(0))
                    return {}
        
        return SimpleLLMClient(self.llm)
    
    def _create_system_message(self) -> str:
        """Create system message augmented with playbook knowledge"""
        base_message = super()._create_system_message()
        
        # Check if ACE is enabled and curator is initialized
        if not self.ace_enabled or self.curator is None:
            return base_message
        
        # Add playbook context (use simple conditions since this is called at init)
        conditions = self._get_simple_conditions()
        playbook_context = self.curator.get_context_for_prompt(
            conditions,
            max_items=self.config.ace.max_playbook_items_in_prompt
        )
        
        ace_instructions = """

SELF-IMPROVEMENT CAPABILITIES:
- If the user asks to "improve", "optimize", or "make it better", you can run a self-improvement loop
- Use the 'improve_model' tool to automatically iterate and find better configurations
- You learn from each experiment and can apply lessons to future tasks

When suggesting changes or strategies, consider the learned knowledge from previous experiments shown above.
"""
        
        return base_message + playbook_context + ace_instructions
    
    def _get_simple_conditions(self) -> Dict[str, Any]:
        """Get basic conditions without LLM (for sync contexts like init)"""
        state = self.toolkit.state
        data_analysis = state.get("data_analysis") or {}
        
        return {
            "task_type": data_analysis.get("task_type", "classification") if data_analysis else "classification",
            "cancer_type": "unknown",
            "n_samples_range": "medium",
            "n_features_range": "medium",
            "baseline_model": state.get("best_model")
        }
    
    async def _get_current_conditions(self) -> Dict[str, Any]:
        """Get current conditions for playbook lookup using LLM analysis"""
        state = self.toolkit.state
        data_analysis = state.get("data_analysis") or {}
        feature_result = state.get("feature_result") or {}
        
        n_samples = feature_result.get("n_samples_train", 0) if feature_result else 0
        n_features = feature_result.get("n_features", 0) if feature_result else 0
        task_type = data_analysis.get("task_type", "classification") if data_analysis else "classification"
        
        # Use LLM to detect cancer context and categorize dataset
        cancer_context = await self._detect_cancer_context()
        dataset_categories = await self._categorize_dataset_characteristics(n_samples, n_features, task_type)
        
        return {
            "task_type": task_type,
            "cancer_type": cancer_context.get("cancer_type", "unknown"),
            "cancer_site": cancer_context.get("site", ""),
            "cancer_stage_info": cancer_context.get("stage_info", ""),
            "n_samples_range": dataset_categories.get("n_samples", "medium"),
            "n_features_range": dataset_categories.get("n_features", "medium"),
            "baseline_model": state.get("best_model")
        }
    
    async def _detect_cancer_context(self) -> Dict[str, str]:
        """Use LLM to detect cancer context from available information"""
        state = self.toolkit.state
        objective = state.get("objective", "")
        dataset_path = state.get("dataset_path", "")
        feature_names = state.get("feature_names", [])[:20]
        
        # Build context for LLM
        context_parts = []
        if objective:
            context_parts.append(f"Objective: {objective}")
        if dataset_path:
            context_parts.append(f"Dataset: {dataset_path}")
        if feature_names:
            context_parts.append(f"Features: {', '.join(str(f) for f in feature_names)}")
        
        if not context_parts:
            return {"cancer_type": "unknown", "site": "", "stage_info": ""}
        
        context = "\n".join(context_parts)
        
        prompt = f"""Analyze this oncology ML task and extract cancer-related context.

TASK INFORMATION:
{context}

This is an ONCOLOGY dataset. Extract:
1. Cancer type (e.g., breast, lung, prostate, colorectal, melanoma, pancreatic, ovarian, etc.)
2. Cancer site/organ if identifiable
3. Any stage or biomarker information mentioned

Return JSON:
{{
    "cancer_type": "specific cancer type or 'unknown'",
    "site": "anatomical site if known",
    "stage_info": "any stage/grade/biomarker info mentioned",
    "reasoning": "brief explanation"
}}
"""
        
        try:
            response = await self._create_llm_for_reflection().complete_json(prompt)
            return {
                "cancer_type": response.get("cancer_type", "unknown").lower(),
                "site": response.get("site", ""),
                "stage_info": response.get("stage_info", "")
            }
        except Exception as e:
            print(f"LLM cancer context detection failed: {e}")
            return {"cancer_type": "unknown", "site": "", "stage_info": ""}
    
    async def _categorize_dataset_characteristics(self, n_samples: int, n_features: int, task_type: str) -> Dict[str, str]:
        """Use LLM to categorize dataset characteristics for oncology context"""
        if n_samples == 0 or n_features == 0:
            return {"n_samples": "unknown", "n_features": "unknown"}
        
        prompt = f"""Categorize this ONCOLOGY dataset for machine learning.

DATASET:
- Samples: {n_samples} patients
- Features: {n_features} clinical/molecular features
- Task: {task_type}

For oncology clinical data, categorize:
- Sample size: small/medium/large (consider clinical trial sizes, not just ML norms)
- Feature count: low/medium/high (consider clinical vs. genomic data scales)

Return JSON:
{{
    "n_samples": "small|medium|large",
    "n_features": "low|medium|high",
    "reasoning": "brief justification"
}}
"""
        
        try:
            response = await self._create_llm_for_reflection().complete_json(prompt)
            return {
                "n_samples": response.get("n_samples", "medium"),
                "n_features": response.get("n_features", "medium")
            }
        except Exception as e:
            print(f"LLM categorization failed: {e}, using fallback")
            # Fallback only if LLM fails
            return {
                "n_samples": "small" if n_samples < 500 else "medium" if n_samples < 5000 else "large",
                "n_features": "low" if n_features < 20 else "medium" if n_features < 100 else "high"
            }
    
    async def chat(self, user_message: str) -> str:
        """
        Process user message with ACE enhancements.
        
        Handles special commands:
        - "improve" / "optimize" - runs self-improvement loop
        - "playbook" / "knowledge" - shows playbook summary
        """
        # Check for improvement request
        if self.ace_enabled and self._is_improvement_request(user_message):
            return await self._handle_improvement_request(user_message)
        
        # Check for playbook query
        if self.ace_enabled and self._is_playbook_query(user_message):
            return self._handle_playbook_query()
        
        # Start trajectory if this is a new analysis
        if self.ace_enabled and self._should_start_trajectory(user_message):
            await self._start_new_trajectory()
        
        # Regular chat processing
        response = await super().chat(user_message)
        
        # Track action for reflection
        if self.ace_enabled:
            self._actions_since_reflection += 1
            
            # Record step in trajectory
            if self.trajectory_generator.current_trajectory:
                self._record_chat_action(user_message, response)
            
            # Check if we should trigger reflection
            if self._should_trigger_reflection():
                await self._trigger_reflection()
        
        # DEBUG: Show state after chat processing
        print(f"  [DEBUG] End of chat. Trained models: {list(self.toolkit.state['trained_models'].keys())}")
        
        return response
    
    def _is_improvement_request(self, message: str) -> bool:
        """Check if user is asking for self-improvement"""
        message_lower = message.lower()
        improvement_keywords = [
            "improve", "optimize", "make it better", "try to improve",
            "self-improve", "iterate", "find better", "enhance performance",
            "can you improve", "try improving"
        ]
        return any(kw in message_lower for kw in improvement_keywords)
    
    def _is_playbook_query(self, message: str) -> bool:
        """Check if user is asking about playbook"""
        message_lower = message.lower()
        playbook_keywords = ["playbook", "what have you learned", "knowledge", "strategies", "lessons"]
        return any(kw in message_lower for kw in playbook_keywords)
    
    async def _handle_improvement_request(self, user_message: str) -> str:
        """Handle user request for self-improvement"""
        # Check prerequisites
        if self.toolkit.state.get("best_model") is None:
            return "I need to train a baseline model first before I can try to improve it. Please ask me to train a model first."
        
        baseline_score = self.toolkit.state.get("best_score", 0)
        baseline_model = self.toolkit.state.get("best_model", "unknown")
        
        # Extract user suggestions if any
        user_suggestions = self._extract_suggestions(user_message)
        
        # Get or create baseline trajectory ID
        baseline_trajectory_id = "baseline"
        if self.trajectory_generator.current_trajectory:
            baseline_trajectory_id = self.trajectory_generator.current_trajectory.trajectory_id
        
        # Run improvement loop
        print(f"\nStarting self-improvement loop...")
        print(f"Baseline: {baseline_model} with score {baseline_score:.4f}")
        
        try:
            result = await self.improvement_controller.run_improvement_loop(
                toolkit=self.toolkit,
                baseline_score=baseline_score,
                baseline_model=baseline_model,
                baseline_trajectory_id=baseline_trajectory_id,
                session_id=self.session_id,
                user_suggestions=user_suggestions,
                max_iterations=self.config.ace.max_improvement_iterations
            )
            
            # Format response
            response = self._format_improvement_response(result)
            
            # Add to conversation history
            self.message_history.append(HumanMessage(content=user_message))
            self.message_history.append(AIMessage(content=response))
            
            return response
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return f"I encountered an error during the improvement loop: {str(e)}. The baseline model is still available."
    
    def _extract_suggestions(self, message: str) -> Optional[List[str]]:
        """Extract specific suggestions from user message"""
        # Look for specific suggestions after "try" or "like"
        message_lower = message.lower()
        
        suggestions = []
        
        # Check for model suggestions
        models = ["xgboost", "random forest", "logistic", "catboost", "lightgbm"]
        for model in models:
            if model in message_lower:
                suggestions.append(f"Try training {model}")
        
        # Check for feature suggestions
        if "feature" in message_lower or "interaction" in message_lower:
            if "age" in message_lower or "tumor" in message_lower:
                suggestions.append("Create age-tumor interaction features")
        
        return suggestions if suggestions else None
    
    def _format_improvement_response(self, result: Dict[str, Any]) -> str:
        """Format improvement loop results as response"""
        lines = []
        lines.append("## Self-Improvement Results\n")
        
        lines.append(f"**Baseline Score:** {result['baseline_score']:.4f}")
        lines.append(f"**Final Score:** {result['final_score']:.4f}")
        lines.append(f"**Total Improvement:** {result['total_improvement']:+.4f}")
        lines.append(f"**Iterations:** {result['iterations']}")
        lines.append("")
        
        if result['successful_changes']:
            lines.append("### Beneficial Changes Found:")
            for change in result['successful_changes']:
                lines.append(f"- {change['description']} (+{change['improvement']:.4f})")
            lines.append("")
        
        if result['lessons_learned'] > 0:
            lines.append(f"*Learned {result['lessons_learned']} new lessons for future experiments.*")
        
        if result['total_improvement'] > 0.01:
            lines.append("\nThe improved model is now the active best model.")
        elif result['total_improvement'] > 0:
            lines.append("\nSmall improvements found. The updated model is now active.")
        else:
            lines.append("\nNo significant improvements found. The baseline model remains the best.")
        
        return "\n".join(lines)
    
    def _handle_playbook_query(self) -> str:
        """Handle query about playbook/learned knowledge"""
        summary = self.curator.get_summary()
        
        lines = []
        lines.append("## My Learned Knowledge (Playbook)\n")
        lines.append(f"**Version:** {summary['version']}")
        lines.append(f"**Total Strategies:** {summary['total_items']}")
        lines.append(f"**Experiments Processed:** {summary['trajectories_processed']}")
        lines.append(f"**Lessons Extracted:** {summary['lessons_extracted']}")
        lines.append("")
        lines.append("### Knowledge by Domain:")
        
        for domain, info in summary['domains'].items():
            if info['n_items'] > 0:
                lines.append(f"\n**{domain.replace('_', ' ').title()}** ({info['n_items']} items):")
                for item in info['top_items'][:2]:
                    lines.append(f"- {item['title'][:60]}... (confidence: {item['confidence']:.0%})")
        
        lines.append("\n*This knowledge informs my decisions and improves over time.*")
        
        return "\n".join(lines)
    
    def _should_start_trajectory(self, message: str) -> bool:
        """Check if we should start a new trajectory"""
        start_keywords = ["analyze", "start", "load", "train", "begin", "run"]
        message_lower = message.lower()
        
        return (
            self.trajectory_generator.current_trajectory is None and
            any(kw in message_lower for kw in start_keywords) and
            self.toolkit.state.get("dataset_path") is not None
        )
    
    async def _start_new_trajectory(self):
        """Start a new trajectory for tracking"""
        state = self.toolkit.state
        feature_result = state.get("feature_result") or {}
        data_analysis = state.get("data_analysis") or {}
        
        # Get cancer context using LLM
        cancer_context = await self._detect_cancer_context()
        
        self.trajectory_generator.start_trajectory(
            experiment_id=self.session_id,
            dataset_info={
                "path": state.get("dataset_path"),
                "n_samples": feature_result.get("n_samples_train", 0),
                "n_features": feature_result.get("n_features", 0)
            },
            cancer_type=cancer_context.get("cancer_type", "unknown"),
            task_type=data_analysis.get("task_type", "classification")
        )
        
        self._baseline_established = False
        self._actions_since_reflection = 0
    
    def _record_chat_action(self, user_message: str, response: str):
        """Record a chat interaction as an action in trajectory"""
        # Determine action type from message
        message_lower = user_message.lower()
        
        if "analyze" in message_lower or "insight" in message_lower:
            action_type = ActionType.DATA_ANALYSIS
        elif "feature" in message_lower or "engineer" in message_lower:
            action_type = ActionType.FEATURE_ENGINEERING
        elif "ensemble" in message_lower or "combine" in message_lower or "stacking" in message_lower:
            action_type = ActionType.ENSEMBLE_CREATION
        elif "train" in message_lower or "model" in message_lower:
            action_type = ActionType.MODEL_TRAINING
        elif "evaluate" in message_lower or "test" in message_lower:
            action_type = ActionType.MODEL_EVALUATION
        elif "error" in message_lower or "wrong" in message_lower:
            action_type = ActionType.ERROR_ANALYSIS
        else:
            return  # Don't track generic messages
        
        # Record the step
        self.trajectory_generator.record_step(
            action_type=action_type,
            action_name="chat_interaction",
            action_inputs={"user_message": user_message[:200]},
            action_outputs={
                "response_length": len(response),
                "best_score": self.toolkit.state.get("best_score", 0),
                "best_model": self.toolkit.state.get("best_model")
            },
            reasoning=user_message[:100]
        )
        
        # Check if baseline is established (first model trained)
        if not self._baseline_established and self.toolkit.state.get("best_model"):
            self._baseline_established = True
    
    def _should_trigger_reflection(self) -> bool:
        """Check if we should trigger automatic reflection"""
        if not self.config.ace.auto_reflect:
            return False
        
        # Trigger after threshold actions
        if self._actions_since_reflection >= self._reflection_threshold:
            return True
        
        # Trigger after model evaluation
        if self.toolkit.state.get("evaluation_results"):
            return True
        
        return False
    
    async def _trigger_reflection(self):
        """Trigger reflection on current trajectory"""
        if self.trajectory_generator.current_trajectory is None:
            return
        
        # End current trajectory
        trajectory = self.trajectory_generator.end_trajectory(
            final_metrics=self.toolkit.state.get("evaluation_results", {}),
            best_model=self.toolkit.state.get("best_model"),
            best_score=self.toolkit.state.get("best_score", 0)
        )
        
        # Reflect
        print("\n   Reflecting on experiment...")
        
        conditions = await self._get_current_conditions()
        playbook_context = self.curator.get_context_for_prompt(conditions)
        lessons = await self.reflector.reflect_on_trajectory(trajectory, playbook_context)
        
        # Curate lessons
        if lessons:
            result = self.curator.curate_lessons(lessons)
            print(f"   Learned {result['items_created']} new, updated {result['items_merged']} existing strategies")
        
        # Reset counter
        self._actions_since_reflection = 0
        
        # Update playbook stats
        self.curator.playbook.total_trajectories_processed += 1
    
    def save_session(self, filepath: Optional[str] = None) -> str:
        """Save session including playbook state"""
        # Trigger final reflection if trajectory active
        if self.ace_enabled and self.trajectory_generator.current_trajectory:
            import asyncio
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.create_task(self._trigger_reflection())
                else:
                    loop.run_until_complete(self._trigger_reflection())
            except Exception:
                pass  # Don't fail save on reflection error
        
        # Save playbook
        if self.ace_enabled:
            self.curator.save()
        
        return super().save_session(filepath)
    
    def print_playbook_summary(self):
        """Print playbook summary"""
        if self.ace_enabled:
            self.curator.print_summary()
        else:
            print("ACE framework is disabled")
    
    async def get_playbook_strategies(self, domain: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get strategies from playbook for current context"""
        if not self.ace_enabled:
            return []
        
        conditions = await self._get_current_conditions()
        focus_domains = [domain] if domain else None
        
        return self.curator.get_strategies_for_improvement(conditions, focus_domains)


# Convenience function to create the enhanced agent
def create_ace_agent(config: Config) -> ACEMLAgent:
    """Create an ACE-enhanced ML agent"""
    return ACEMLAgent(config)

