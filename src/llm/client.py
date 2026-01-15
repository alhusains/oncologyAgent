"""LLM client for GPT-4o integration"""

from typing import Optional, Dict, Any, List
import asyncio
import json
import os

try:
    from openai import AsyncOpenAI
except ImportError:
    AsyncOpenAI = None

from ..core.config import LLMConfig


class LLMClient:
    """Client for interacting with LLM services"""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        if config.provider == "openai":
            if AsyncOpenAI is None:
                raise ImportError("OpenAI package not available. Install with: pip install openai")
            
            # Handle API key
            api_key = config.api_key or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OpenAI API key not provided. Set OPENAI_API_KEY environment variable or config.llm.api_key")
            
            self.client = AsyncOpenAI(
                api_key=api_key,
                timeout=config.timeout
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {config.provider}")
    
    async def complete(
        self, 
        prompt: str,
        system_message: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        response_format: Optional[str] = None
    ) -> str:
        """Generate completion for a prompt"""
        
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})
        
        # GPT-5 only supports default temperature of 1
        effective_temperature = temperature or self.config.temperature
        if self.config.model.startswith("gpt-4.1-mini") or self.config.model.startswith("gpt-5"):
            effective_temperature = 1.0
        
        kwargs = {
            "model": self.config.model,
            "messages": messages,
            "temperature": effective_temperature,
        }
        
        # GPT-5 uses max_completion_tokens (includes reasoning + output tokens combined)
        is_gpt5 = self.config.model.startswith("gpt-5")
        if is_gpt5:
            base_tokens = max_tokens or self.config.max_tokens
            kwargs["max_completion_tokens"] = base_tokens * 2  # Account for reasoning overhead
            kwargs["reasoning_effort"] = self.config.reasoning_effort
            kwargs["verbosity"] = self.config.verbosity
        else:
            kwargs["max_tokens"] = max_tokens or self.config.max_tokens
        
        if response_format == "json" and not is_gpt5:
            kwargs["response_format"] = {"type": "json_object"}
            
        try:
            create_params = {
                "model": kwargs["model"],
                "messages": kwargs["messages"],
                "temperature": kwargs["temperature"],
            }
            
            # Add max tokens parameter
            if "max_tokens" in kwargs:
                create_params["max_tokens"] = kwargs["max_tokens"]
            elif "max_completion_tokens" in kwargs:
                create_params["max_completion_tokens"] = kwargs["max_completion_tokens"]
            
            # Add optional parameters
            if "response_format" in kwargs:
                create_params["response_format"] = kwargs["response_format"]
            if "reasoning_effort" in kwargs:
                create_params["reasoning_effort"] = kwargs["reasoning_effort"]
            if "verbosity" in kwargs:
                create_params["verbosity"] = kwargs["verbosity"]
            
            response = await self.client.chat.completions.create(**create_params)
            return response.choices[0].message.content
        except Exception as e:
            raise RuntimeError(f"LLM completion failed: {str(e)}")
    
    async def complete_json(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        temperature: Optional[float] = None
    ) -> Dict[str, Any]:
        """Generate completion with JSON response format"""
        
        # GPT-5 requires stronger JSON format enforcement via prompting
        if self.config.model.startswith("gpt-5"):
            json_prompt = f"""
            {prompt}
            
            IMPORTANT: You must respond with ONLY valid JSON. Do not include any explanatory text, markdown formatting, 
            or code blocks. Your entire response must be a single parseable JSON object starting with {{ and ending with }}.
            """
        else:
            json_prompt = f"""
            {prompt}
            
            Please respond in valid JSON format.
            """
        
        response = await self.complete(
            prompt=json_prompt,
            system_message=system_message,
            temperature=temperature,
            response_format="json"
        )
        
        try:
            return json.loads(response)
        except json.JSONDecodeError as e:
            # Try to extract JSON from the response
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group())
                except json.JSONDecodeError:
                    pass
            raise ValueError(f"Invalid JSON response from LLM: {e}")
    
    async def analyze_data_schema(
        self, 
        data_info: Dict[str, Any],
        user_objective: str
    ) -> Dict[str, Any]:
        """Analyze data schema and provide recommendations"""
        
        from ..llm.prompts import PromptTemplates
        
        system_message = """You are a data science expert. Analyze the provided dataset schema and user objective to provide recommendations for ML pipeline."""
        
        prompt = PromptTemplates.data_analysis_prompt(data_info, user_objective)
        
        return await self.complete_json(prompt, system_message)
    
    async def suggest_feature_engineering(
        self,
        data_schema: Dict[str, Any],
        target_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Suggest feature engineering techniques"""
        
        # Import the prompt template
        from ..llm.prompts import PromptTemplates
        
        # Use the standardized prompt from prompts.py
        prompt = PromptTemplates.feature_engineering_prompt(
            data_schema=data_schema,
            target_info=target_info,
            domain_context="oncology"  # Default domain
        )
        
        system_message = """You are an expert feature engineer. Analyze the dataset and suggest SPECIFIC, IMPLEMENTABLE feature engineering operations. Follow the format exactly as specified."""
        
        return await self.complete_json(prompt, system_message)
    
    async def suggest_models(
        self,
        task_type: str,
        data_characteristics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Suggest appropriate models for the task"""
        
        system_message = """You are an ML model selection expert. Recommend the best models for the given task and data characteristics."""
        
        prompt = f"""
        Task Type: {task_type}
        
        Data Characteristics:
        {json.dumps(data_characteristics, indent=2)}
        
        Please recommend:
        1. Top 3-5 models to try (with reasoning)
        2. Hyperparameter ranges for each model
        3. Evaluation metrics to focus on
        4. Potential challenges and solutions
        
        Respond in JSON format with model recommendations and rationale.
        """
        
        return await self.complete_json(prompt, system_message)
    
    async def critique_model_performance(
        self,
        performance_results: Dict[str, Any],
        experiment_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Critique model performance and suggest improvements"""
        
        system_message = """You are an ML performance analyst. Critique the model results and provide specific improvement suggestions."""
        
        prompt = f"""
        Model Performance Results:
        {json.dumps(performance_results, indent=2)}
        
        Experiment Context:
        {json.dumps(experiment_context, indent=2)}
        
        Please provide:
        1. Performance assessment
        2. Identification of issues (overfitting, underfitting, bias, etc.)
        3. Specific improvement suggestions
        4. Alternative approaches to try
        5. Data collection recommendations (if applicable)
        
        Respond in JSON format with detailed analysis and actionable suggestions.
        """
        
        return await self.complete_json(prompt, system_message)
    
    async def generate_report_content(
        self,
        experiment_results: Dict[str, Any],
        interpretability_results: Dict[str, Any]
    ) -> str:
        """Generate report content from experiment results"""
        
        system_message = """You are a data science report writer. Create a comprehensive, professional report from the ML experiment results."""
        
        prompt = f"""
        Experiment Results:
        {json.dumps(experiment_results, indent=2)}
        
        Interpretability Results:
        {json.dumps(interpretability_results, indent=2)}
        
        Please create a comprehensive report including:
        1. Executive Summary
        2. Data Analysis Summary
        3. Model Performance Analysis
        4. Feature Importance and Interpretability
        5. Conclusions and Recommendations
        6. Technical Details and Methodology
        
        Format the report in markdown with clear sections and professional language.
        """
        
        return await self.complete(prompt, system_message)
