"""State management for the tabular ML agent framework"""

from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum
import uuid
import json
from pathlib import Path


class TaskStatus(str, Enum):
    """Status of a task or experiment"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskType(str, Enum):
    """Types of tasks in the ML pipeline"""
    DATA_ANALYSIS = "data_analysis"
    FEATURE_ENGINEERING = "feature_engineering"
    MODEL_SELECTION = "model_selection"
    MODEL_TRAINING = "model_training"
    MODEL_EVALUATION = "model_evaluation"
    INTERPRETATION = "interpretation"
    CRITIQUE = "critique"
    REPORTING = "reporting"


class AgentResult(BaseModel):
    """Result from an agent execution"""
    task_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    task_type: TaskType
    status: TaskStatus = TaskStatus.PENDING
    agent_name: str
    timestamp: datetime = Field(default_factory=datetime.now)
    
    # Input/Output
    inputs: Dict[str, Any] = Field(default_factory=dict)
    outputs: Dict[str, Any] = Field(default_factory=dict)
    
    # Metadata
    execution_time_seconds: Optional[float] = None
    error_message: Optional[str] = None
    confidence_score: Optional[float] = None
    suggestions: List[str] = Field(default_factory=list)
    
    # Artifacts
    artifacts: Dict[str, Any] = Field(default_factory=dict)  # plots, models, etc.


class DataSchema(BaseModel):
    """Schema information about the dataset"""
    n_rows: int
    n_cols: int
    column_info: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    target_column: Optional[str] = None
    task_type: Optional[str] = None  # classification, regression
    data_quality_issues: List[str] = Field(default_factory=list)


class MLPipelineState(BaseModel):
    """State of the ML pipeline"""
    model_config = {"protected_namespaces": ()}  # Disable protected namespace warning
    
    # Data information
    data_schema: Optional[DataSchema] = None
    feature_columns: List[str] = Field(default_factory=list)
    target_column: Optional[str] = None
    
    # Processing results
    processed_features: Optional[Dict[str, Any]] = None
    train_test_split_info: Optional[Dict[str, Any]] = None
    
    # Model information
    selected_models: List[str] = Field(default_factory=list)
    trained_models: Dict[str, Any] = Field(default_factory=dict)
    best_model: Optional[str] = None
    
    # Evaluation results
    model_performance: Dict[str, Dict[str, float]] = Field(default_factory=dict)
    
    # Interpretability
    feature_importance: Optional[Dict[str, float]] = None
    explanations: Optional[Dict[str, Any]] = None


class ExperimentState(BaseModel):
    """Complete state of an ML experiment"""
    experiment_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    experiment_name: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    
    # User inputs
    user_objective: Optional[str] = None
    dataset_path: Optional[str] = None
    target_variable: Optional[str] = None
    
    # Pipeline state
    pipeline_state: MLPipelineState = Field(default_factory=MLPipelineState)
    
    # Execution history
    task_history: List[AgentResult] = Field(default_factory=list)
    current_task: Optional[TaskType] = None
    
    # Status
    overall_status: TaskStatus = TaskStatus.PENDING
    progress_percentage: float = 0.0
    
    # Critique and improvement
    critique_history: List[str] = Field(default_factory=list)
    improvement_suggestions: List[str] = Field(default_factory=list)
    iteration_count: int = 0
    max_iterations: int = 3

    def add_task_result(self, result: AgentResult) -> None:
        """Add a task result to the history"""
        self.task_history.append(result)
        self.updated_at = datetime.now()
        
        # Update current task
        if result.status == TaskStatus.IN_PROGRESS:
            self.current_task = result.task_type
        elif result.status in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
            self.current_task = None
    
    def get_latest_result(self, task_type: TaskType) -> Optional[AgentResult]:
        """Get the latest result for a specific task type"""
        results = [r for r in self.task_history if r.task_type == task_type]
        return results[-1] if results else None
    
    def get_completed_tasks(self) -> List[TaskType]:
        """Get list of completed task types"""
        completed = []
        for result in self.task_history:
            if result.status == TaskStatus.COMPLETED and result.task_type not in completed:
                completed.append(result.task_type)
        return completed
    
    def update_progress(self) -> None:
        """Update progress percentage based on completed tasks"""
        total_tasks = len(TaskType)
        completed_tasks = len(self.get_completed_tasks())
        self.progress_percentage = (completed_tasks / total_tasks) * 100
    
    def save_to_file(self, filepath: Union[str, Path]) -> None:
        """Save experiment state to file"""
        with open(filepath, 'w') as f:
            json.dump(self.model_dump(mode='json'), f, indent=2, default=str)
    
    @classmethod
    def load_from_file(cls, filepath: Union[str, Path]) -> "ExperimentState":
        """Load experiment state from file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls(**data)


class AgentState(BaseModel):
    """State for individual agents"""
    agent_name: str
    current_task: Optional[str] = None
    context: Dict[str, Any] = Field(default_factory=dict)
    memory: List[str] = Field(default_factory=list)
    last_action: Optional[str] = None
    
    def add_memory(self, memory_item: str) -> None:
        """Add item to agent memory"""
        self.memory.append(f"{datetime.now().isoformat()}: {memory_item}")
        # Keep only last 50 memory items
        if len(self.memory) > 50:
            self.memory = self.memory[-50:]
    
    def update_context(self, key: str, value: Any) -> None:
        """Update agent context"""
        self.context[key] = value
    
    def get_context_summary(self) -> str:
        """Get a summary of current context"""
        return json.dumps(self.context, indent=2, default=str)
