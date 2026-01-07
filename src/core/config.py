"""Configuration management for the tabular ML agent framework"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from pathlib import Path
import yaml
import os
from dotenv import load_dotenv

load_dotenv()


class LLMConfig(BaseModel):
    """Configuration for LLM settings"""
    provider: str = Field(default="openai", description="LLM provider")
    model: str = Field(default="gpt-4o-mini", description="Model name")
    api_key: Optional[str] = Field(default=None, description="API key")
    temperature: float = Field(default=0.1, description="Temperature for generation")
    max_tokens: int = Field(default=4096, description="Maximum tokens for response")
    timeout: int = Field(default=300, description="Request timeout in seconds")


class DataConfig(BaseModel):
    """Configuration for data processing"""
    max_file_size_mb: int = Field(default=500, description="Maximum file size in MB")
    supported_formats: List[str] = Field(
        default=["csv", "xlsx", "parquet"], description="Supported file formats"
    )
    missing_value_threshold: float = Field(
        default=0.5, description="Threshold for missing values (0-1)"
    )
    categorical_cardinality_threshold: int = Field(
        default=50, description="Max unique values for categorical features"
    )
    test_size: float = Field(default=0.2, description="Test set proportion")
    val_size: float = Field(default=0.2, description="Validation set proportion")
    random_state: int = Field(default=42, description="Random state for reproducibility")
    use_preset_CV: bool = Field(default=False, description="Use preset CV column from dataset for cross-validation groups")


class MLConfig(BaseModel):
    """Configuration for ML pipeline"""
    max_training_time_minutes: int = Field(
        default=60, description="Maximum training time per model"
    )
    cv_folds: int = Field(default=5, description="Cross-validation folds")
    optuna_trials: int = Field(default=100, description="Optuna optimization trials")
    early_stopping_rounds: int = Field(default=50, description="Early stopping rounds")
    min_models_to_train: int = Field(default=3, description="Minimum number of models to train")
    max_models_to_train: int = Field(default=5, description="Maximum number of models to train")
    n_jobs: int = Field(
        default=8, description="Number of parallel jobs/threads for models (set via ML_N_JOBS env var or config)"
    )
    models_to_try: List[str] = Field(
        default=["autogluon", "catboost", "xgboost", "random_forest"],
        description="Models to evaluate (for classification, autogluon trains an ensemble)"
    )
    metrics: Dict[str, List[str]] = Field(
        default={
            "classification": ["accuracy", "f1", "roc_auc", "precision", "recall"],
            "regression": ["mae", "mse", "rmse", "r2"]
        },
        description="Metrics for different task types"
    )


class ACEConfig(BaseModel):
    """Configuration for ACE (Agentic Context Engineering) framework"""
    enabled: bool = Field(default=True, description="Enable ACE framework")
    playbook_path: str = Field(default="knowledge/playbook.json", description="Path to playbook file")
    
    # Self-improvement settings
    max_improvement_iterations: int = Field(
        default=3, description="Maximum iterations for self-improvement loop"
    )
    min_improvement_threshold: float = Field(
        default=0.005, description="Minimum improvement to consider a change beneficial"
    )
    max_changes_per_iteration: int = Field(
        default=3, description="Maximum changes to test per iteration"
    )
    
    # Reflection settings
    auto_reflect: bool = Field(
        default=True, description="Automatically trigger reflection after experiments"
    )
    reflection_threshold: int = Field(
        default=5, description="Number of actions before triggering reflection"
    )
    
    # Playbook settings
    auto_save_playbook: bool = Field(default=True, description="Auto-save playbook after updates")
    merge_threshold: float = Field(
        default=0.75, description="Similarity threshold for merging playbook items (0-1)"
    )
    min_confidence_for_suggestions: float = Field(
        default=0.4, description="Minimum confidence for playbook suggestions"
    )
    max_playbook_items_in_prompt: int = Field(
        default=10, description="Maximum playbook items to include in LLM prompts"
    )


class Config(BaseModel):
    """Main configuration class"""
    llm: LLMConfig = Field(default_factory=LLMConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    ml: MLConfig = Field(default_factory=MLConfig)
    ace: ACEConfig = Field(default_factory=ACEConfig)
    
    # Paths
    data_dir: Path = Field(default=Path("data"), description="Data directory")
    output_dir: Path = Field(default=Path("outputs"), description="Output directory")
    config_dir: Path = Field(default=Path("configs"), description="Config directory")
    knowledge_dir: Path = Field(default=Path("knowledge"), description="Knowledge/playbook directory")
    
    # Logging
    log_level: str = Field(default="INFO", description="Logging level")
    log_to_file: bool = Field(default=True, description="Log to file")
    
    # UI
    gradio_port: int = Field(default=7860, description="Gradio server port")
    gradio_share: bool = Field(default=False, description="Share Gradio interface")

    @classmethod
    def from_yaml(cls, config_path: str) -> "Config":
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)
    
    @classmethod
    def from_env(cls) -> "Config":
        """Load configuration from environment variables"""
        config = cls()
        
        # Override with environment variables
        if os.getenv("OPENAI_API_KEY"):
            config.llm.api_key = os.getenv("OPENAI_API_KEY")
        if os.getenv("LLM_MODEL"):
            config.llm.model = os.getenv("LLM_MODEL")
        if os.getenv("LOG_LEVEL"):
            config.log_level = os.getenv("LOG_LEVEL")
        if os.getenv("ML_N_JOBS"):
            config.ml.n_jobs = int(os.getenv("ML_N_JOBS"))
            
        return config
    
    def save_yaml(self, config_path: str) -> None:
        """Save configuration to YAML file"""
        with open(config_path, 'w') as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False)

    def model_post_init(self, __context: Any) -> None:
        """Create directories if they don't exist"""
        self.data_dir.mkdir(exist_ok=True, parents=True)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.config_dir.mkdir(exist_ok=True, parents=True)
        self.knowledge_dir.mkdir(exist_ok=True, parents=True)
