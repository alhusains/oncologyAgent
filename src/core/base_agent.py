"""Base agent class for the tabular ML agent framework"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from datetime import datetime
import time

from .state import AgentResult, TaskType, TaskStatus, AgentState
from .config import Config


class BaseAgent(ABC):
    """Abstract base class for all agents in the framework"""
    
    def __init__(self, name: str, config: Config):
        self.name = name
        self.config = config
        self.state = AgentState(agent_name=name)
    
    @abstractmethod
    async def execute(self, inputs: Dict[str, Any]) -> AgentResult:
        """Execute the agent's main task"""
        pass
    
    @abstractmethod
    def get_task_type(self) -> TaskType:
        """Return the task type this agent handles"""
        pass
    
    def create_result(
        self, 
        inputs: Dict[str, Any], 
        outputs: Dict[str, Any] = None,
        status: TaskStatus = TaskStatus.COMPLETED,
        error_message: Optional[str] = None,
        confidence_score: Optional[float] = None,
        suggestions: list = None,
        execution_time: Optional[float] = None
    ) -> AgentResult:
        """Create a standardized agent result"""
        return AgentResult(
            task_type=self.get_task_type(),
            agent_name=self.name,
            status=status,
            inputs=inputs,
            outputs=outputs or {},
            error_message=error_message,
            confidence_score=confidence_score,
            suggestions=suggestions or [],
            execution_time_seconds=execution_time
        )
    
    async def run_with_timing(self, inputs: Dict[str, Any]) -> AgentResult:
        """Execute agent with timing and error handling"""
        start_time = time.time()
        
        try:
            self.state.current_task = f"Executing {self.get_task_type().value}"
            self.state.add_memory(f"Started task: {self.get_task_type().value}")
            
            result = await self.execute(inputs)
            
            end_time = time.time()
            result.execution_time_seconds = end_time - start_time
            
            self.state.last_action = f"Completed {self.get_task_type().value}"
            self.state.add_memory(f"Completed task in {result.execution_time_seconds:.2f}s")
            
            return result
            
        except Exception as e:
            end_time = time.time()
            execution_time = end_time - start_time
            
            error_result = self.create_result(
                inputs=inputs,
                status=TaskStatus.FAILED,
                error_message=str(e),
                execution_time=execution_time
            )
            
            self.state.add_memory(f"Task failed: {str(e)}")
            self.state.current_task = None
            
            return error_result
    
    def validate_inputs(self, inputs: Dict[str, Any], required_keys: list) -> bool:
        """Validate that required input keys are present"""
        missing_keys = [key for key in required_keys if key not in inputs]
        if missing_keys:
            raise ValueError(f"Missing required inputs: {missing_keys}")
        return True
    
    def log(self, message: str, level: str = "INFO") -> None:
        """Log a message (can be extended with proper logging)"""
        timestamp = datetime.now().isoformat()
        print(f"[{timestamp}] {level} - {self.name}: {message}")
        self.state.add_memory(f"{level}: {message}")


class LLMAgent(BaseAgent):
    """Base class for agents that use LLM capabilities"""
    
    def __init__(self, name: str, config: Config):
        super().__init__(name, config)
        from ..llm.client import LLMClient
        self.llm_client = LLMClient(config.llm)
    
    async def query_llm(
        self, 
        prompt: str, 
        system_message: Optional[str] = None,
        temperature: Optional[float] = None
    ) -> str:
        """Query the LLM with a prompt"""
        try:
            response = await self.llm_client.complete(
                prompt=prompt,
                system_message=system_message,
                temperature=temperature
            )
            self.state.add_memory(f"LLM query completed. Response length: {len(response)} chars")
            return response
        except Exception as e:
            self.log(f"LLM query failed: {str(e)}", "ERROR")
            raise
    
    async def analyze_with_llm(
        self, 
        data_description: str, 
        analysis_type: str,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Analyze data using LLM with specific analysis type"""
        context_str = ""
        if context:
            context_str = f"\nContext: {context}"
        
        prompt = f"""
        Analyze the following data for {analysis_type}:
        
        Data Description:
        {data_description}
        {context_str}
        
        Please provide a detailed analysis with specific recommendations.
        """
        
        return await self.query_llm(prompt)


class AnalysisAgent(LLMAgent):
    """Base class for agents that perform data analysis"""
    
    def __init__(self, name: str, config: Config):
        super().__init__(name, config)
    
    async def generate_insights(self, data_summary: Dict[str, Any]) -> list:
        """Generate insights from data summary using LLM"""
        prompt = f"""
        Based on the following data summary, generate key insights and recommendations:
        
        {data_summary}
        
        Provide insights in a structured format with actionable recommendations.
        """
        
        response = await self.query_llm(prompt)
        # Parse response into structured insights
        insights = response.split('\n')
        return [insight.strip() for insight in insights if insight.strip()]


class CritiquingAgent(LLMAgent):
    """Base class for agents that provide critiques and suggestions"""
    
    def __init__(self, name: str, config: Config):
        super().__init__(name, config)
    
    async def critique_results(
        self, 
        results: Dict[str, Any], 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Critique results and provide improvement suggestions"""
        prompt = f"""
        Please critique the following ML results and provide specific improvement suggestions:
        
        Results:
        {results}
        
        Context:
        {context}
        
        Focus on:
        1. Model performance issues
        2. Data quality concerns
        3. Feature engineering opportunities
        4. Model selection improvements
        5. Validation concerns
        
        Provide specific, actionable recommendations.
        """
        
        response = await self.query_llm(prompt)
        
        # Parse response into structured critique
        return {
            "critique": response,
            "improvement_suggestions": self._extract_suggestions(response),
            "confidence": 0.8  # Default confidence
        }
    
    def _extract_suggestions(self, critique_text: str) -> list:
        """Extract actionable suggestions from critique text"""
        lines = critique_text.split('\n')
        suggestions = []
        
        for line in lines:
            line = line.strip()
            if line.startswith(('•', '-', '*', '1.', '2.', '3.', '4.', '5.')):
                suggestions.append(line)
        
        return suggestions
