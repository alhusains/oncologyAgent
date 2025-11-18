"""LangChain-compatible tool definitions for ML toolkit"""

import asyncio
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field
from langchain.tools import StructuredTool

from .tools import MLToolkit


# ============================================================================
# Input Schemas for Each Tool
# ============================================================================

class AnalyzeDataInput(BaseModel):
    """Input schema for analyze_data tool"""
    dataset_path: str = Field(
        description="Path to the dataset file (CSV or Excel)"
    )
    objective: str = Field(
        description="ML objective/goal describing what to predict"
    )


class EngineerFeaturesInput(BaseModel):
    """Input schema for engineer_features tool"""
    target_variable: Optional[str] = Field(
        default=None,
        description="Target variable name. If None, uses the one from data analysis"
    )
    strategy: str = Field(
        default="auto",
        description="Feature engineering strategy: 'auto', 'aggressive', or 'conservative'"
    )


class SelectModelsInput(BaseModel):
    """Input schema for select_models tool"""
    task_type: Optional[str] = Field(
        default=None,
        description="Task type: 'classification', 'regression', or 'survival'. If None, auto-detected"
    )


class TrainModelInput(BaseModel):
    """Input schema for train_model tool"""
    model_name: str = Field(
        description="Name of model to train. Classification: 'autogluon' (recommended - trains ensemble), 'logistic_regression', 'random_forest', 'xgboost', 'lightgbm', 'catboost'. Survival: 'cox_ph', 'random_survival_forest', etc."
    )
    quick_mode: bool = Field(
        default=False,
        description="Use quick mode for faster training with fewer hyperparameter trials"
    )


class EvaluateModelInput(BaseModel):
    """Input schema for evaluate_model tool"""
    model_name: str = Field(
        description="Name of trained model to evaluate on test set"
    )


class AnalyzeErrorsInput(BaseModel):
    """Input schema for analyze_errors tool"""
    model_name: str = Field(
        description="Name of model to analyze errors for"
    )
    n_samples: int = Field(
        default=20,
        description="Number of error samples to analyze in detail"
    )


class GetFeatureImportanceInput(BaseModel):
    """Input schema for get_feature_importance tool"""
    model_name: str = Field(
        description="Name of model to get feature importance from"
    )
    top_k: int = Field(
        default=10,
        description="Number of top features to return"
    )


class GetCurrentStateInput(BaseModel):
    """Input schema for get_current_state tool"""
    pass


# ============================================================================
# Tool Creation Function
# ============================================================================

def create_langchain_tools(toolkit: MLToolkit) -> list:
    """
    Convert MLToolkit to LangChain StructuredTool objects.
    
    This creates LangChain-compatible wrappers around the existing ML toolkit
    while preserving all functionality for classification, regression, and
    survival analysis tasks.
    
    Args:
        toolkit: MLToolkit instance containing the ML pipeline components
        
    Returns:
        List of 8 LangChain StructuredTool objects
    """
    
    tools = []
    
    # ========================================================================
    # Tool 1: Analyze Data
    # ========================================================================
    
    async def analyze_data_async(dataset_path: str, objective: str) -> Dict[str, Any]:
        """
        Analyze dataset structure and identify target variable.
        
        Supports classification, regression, and survival analysis tasks.
        Automatically detects task type from objective and data.
        """
        return await toolkit._analyze_data(dataset_path, objective)
    
    def analyze_data_sync(dataset_path: str, objective: str) -> Dict[str, Any]:
        """Synchronous wrapper for analyze_data"""
        return asyncio.run(analyze_data_async(dataset_path, objective))
    
    tools.append(StructuredTool(
        name="analyze_data",
        description=(
            "Analyze the dataset to understand its structure, identify the target variable, "
            "and determine the task type (classification/regression/survival). "
            "This tool handles all three task types automatically. "
            "ALWAYS call this first before any other tool."
        ),
        func=analyze_data_sync,
        coroutine=analyze_data_async,
        args_schema=AnalyzeDataInput
    ))
    
    # ========================================================================
    # Tool 2: Engineer Features
    # ========================================================================
    
    async def engineer_features_async(
        target_variable: Optional[str] = None,
        strategy: str = "auto"
    ) -> Dict[str, Any]:
        """
        Apply feature engineering including preprocessing, encoding, and scaling.
        
        Handles:
        - Classification: Standard preprocessing + class balancing
        - Regression: Standard preprocessing + target scaling
        - Survival: Risk-stratified splitting + censoring handling
        """
        return await toolkit._engineer_features(target_variable, strategy)
    
    def engineer_features_sync(
        target_variable: Optional[str] = None,
        strategy: str = "auto"
    ) -> Dict[str, Any]:
        """Synchronous wrapper for engineer_features"""
        return asyncio.run(engineer_features_async(target_variable, strategy))
    
    tools.append(StructuredTool(
        name="engineer_features",
        description=(
            "Apply feature engineering including preprocessing, encoding, and scaling. "
            "Automatically handles classification, regression, and survival analysis tasks. "
            "For survival tasks, applies risk-stratified splitting to ensure balanced event distribution. "
            "Call this after analyze_data."
        ),
        func=engineer_features_sync,
        coroutine=engineer_features_async,
        args_schema=EngineerFeaturesInput
    ))
    
    # ========================================================================
    # Tool 3: Select Models
    # ========================================================================
    
    async def select_models_async(task_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Select appropriate models based on task type and data characteristics.
        
        Returns different model sets for:
        - Classification: logistic_regression, random_forest, xgboost, lightgbm
        - Regression: linear_regression, random_forest, xgboost, lightgbm
        - Survival: cox_ph, random_survival_forest, gradient_boosting_survival
        """
        return await toolkit._select_models(task_type)
    
    def select_models_sync(task_type: Optional[str] = None) -> Dict[str, Any]:
        """Synchronous wrapper for select_models"""
        return asyncio.run(select_models_async(task_type))
    
    tools.append(StructuredTool(
        name="select_models",
        description=(
            "Select appropriate ML models based on the task type and dataset characteristics. "
            "Automatically selects the right models for classification, regression, or survival analysis. "
            "For survival: suggests Cox PH, Random Survival Forest, Gradient Boosting Survival. "
            "Call this after engineer_features."
        ),
        func=select_models_sync,
        coroutine=select_models_async,
        args_schema=SelectModelsInput
    ))
    
    # ========================================================================
    # Tool 4: Train Model
    # ========================================================================
    
    async def train_model_async(
        model_name: str,
        quick_mode: bool = False
    ) -> Dict[str, Any]:
        """
        Train a specific model with hyperparameter optimization.
        
        Supports all model types:
        - Classification: autogluon (AutoML ensemble - recommended), logistic_regression, random_forest, xgboost, lightgbm, catboost, tabpfn
        - Regression: linear_regression, ridge, random_forest, xgboost, lightgbm
        - Survival: cox_ph, coxnet, random_survival_forest, gradient_boosting_survival, deepsurv
        """
        return await toolkit._train_model(model_name, quick_mode)
    
    def train_model_sync(model_name: str, quick_mode: bool = False) -> Dict[str, Any]:
        """Synchronous wrapper for train_model"""
        return asyncio.run(train_model_async(model_name, quick_mode))
    
    tools.append(StructuredTool(
        name="train_model",
        description=(
            "Train a specific ML model. For classification, 'autogluon' trains an AutoML ensemble (recommended). "
            "For individual models, supports logistic_regression, xgboost, lightgbm, catboost, random_forest. "
            "For survival analysis: cox_ph, random_survival_forest, gradient_boosting_survival, deepsurv. "
            "Can be called multiple times to train different models. "
            "Use quick_mode=true for faster training on small datasets."
        ),
        func=train_model_sync,
        coroutine=train_model_async,
        args_schema=TrainModelInput
    ))
    
    # ========================================================================
    # Tool 5: Evaluate Model
    # ========================================================================
    
    async def evaluate_model_async(model_name: str) -> Dict[str, Any]:
        """
        Evaluate trained model on held-out test set.
        
        Returns task-specific metrics:
        - Classification: accuracy, precision, recall, F1, ROC-AUC
        - Regression: R², MAE, MSE, RMSE
        - Survival: C-index, Integrated Brier Score, time-dependent AUC
        """
        return await toolkit._evaluate_model(model_name)
    
    def evaluate_model_sync(model_name: str) -> Dict[str, Any]:
        """Synchronous wrapper for evaluate_model"""
        return asyncio.run(evaluate_model_async(model_name))
    
    tools.append(StructuredTool(
        name="evaluate_model",
        description=(
            "Evaluate a trained model's performance on the held-out test set. "
            "Returns appropriate metrics for the task type: "
            "- Classification: accuracy, F1, ROC-AUC "
            "- Regression: R², MAE, RMSE "
            "- Survival: C-index, Integrated Brier Score, time-dependent AUC "
            "Essential for understanding model quality. Call after train_model."
        ),
        func=evaluate_model_sync,
        coroutine=evaluate_model_async,
        args_schema=EvaluateModelInput
    ))
    
    # ========================================================================
    # Tool 6: Analyze Errors
    # ========================================================================
    
    async def analyze_errors_async(
        model_name: str,
        n_samples: int = 20
    ) -> Dict[str, Any]:
        """
        Deep dive into prediction errors to understand model failures.
        
        For all task types, provides:
        - Error patterns and distributions
        - Feature differences in errors vs correct predictions
        - Confidence analysis
        - Actionable improvement suggestions
        """
        return await toolkit._analyze_errors(model_name, n_samples)
    
    def analyze_errors_sync(
        model_name: str,
        n_samples: int = 20
    ) -> Dict[str, Any]:
        """Synchronous wrapper for analyze_errors"""
        return asyncio.run(analyze_errors_async(model_name, n_samples))
    
    tools.append(StructuredTool(
        name="analyze_errors",
        description=(
            "Analyze prediction errors to understand why the model fails. "
            "Works for classification, regression, and survival tasks. "
            "Provides error patterns, feature analysis, and improvement suggestions. "
            "Very useful when model performance is poor (<0.7) to guide improvements."
        ),
        func=analyze_errors_sync,
        coroutine=analyze_errors_async,
        args_schema=AnalyzeErrorsInput
    ))
    
    # ========================================================================
    # Tool 7: Get Feature Importance
    # ========================================================================
    
    async def get_feature_importance_async(
        model_name: str,
        top_k: int = 10
    ) -> Dict[str, Any]:
        """
        Get feature importance rankings for a trained model.
        
        Works for all task types and most model types.
        Helps understand what drives predictions.
        """
        return await toolkit._get_feature_importance(model_name, top_k)
    
    def get_feature_importance_sync(
        model_name: str,
        top_k: int = 10
    ) -> Dict[str, Any]:
        """Synchronous wrapper for get_feature_importance"""
        return asyncio.run(get_feature_importance_async(model_name, top_k))
    
    tools.append(StructuredTool(
        name="get_feature_importance",
        description=(
            "Get the most important features for a trained model. "
            "Works for classification, regression, and survival models. "
            "Helps understand what drives predictions and can guide feature engineering. "
            "Returns ranked list of features with importance scores."
        ),
        func=get_feature_importance_sync,
        coroutine=get_feature_importance_async,
        args_schema=GetFeatureImportanceInput
    ))
    
    # ========================================================================
    # Tool 8: Get Current State
    # ========================================================================
    
    def get_current_state() -> Dict[str, Any]:
        """
        Get current progress and state of the ML pipeline.
        
        Shows what's been completed and current best results.
        Useful for checking progress mid-execution.
        """
        return toolkit._get_current_state()
    
    tools.append(StructuredTool(
        name="get_current_state",
        description=(
            "Check the current progress of the ML pipeline. "
            "Shows what steps have been completed, which models have been trained, "
            "and the current best model and score. "
            "Useful for understanding where you are in the process."
        ),
        func=get_current_state,
        args_schema=GetCurrentStateInput
    ))
    
    return tools

