"""Model selection module with LLM guidance"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

from ..core.base_agent import LLMAgent
from ..core.state import TaskType, AgentResult
from ..core.config import Config


class ModelSelector(LLMAgent):
    """Agent for selecting appropriate models based on data characteristics"""
    
    def __init__(self, config: Config):
        super().__init__("model_selector", config)
        
        # Model configurations for different task types
        self.model_configs = {
            "classification": {
                "autogluon": {
                    "class": "TabularPredictor",
                    "module": "autogluon.tabular",
                    "hyperparams": {},
                    "notes": "AutoML ensemble that trains and combines multiple models automatically. Handles preprocessing, ensembling, and hyperparameter tuning internally. Best for most classification tasks."
                },
                "logistic_regression": {
                    "class": "LogisticRegression",
                    "module": "sklearn.linear_model",
                    "hyperparams": {
                        "C": {"type": "float", "range": [0.001, 100], "log": True},
                        "solver": {"type": "categorical", "values": ["liblinear", "lbfgs", "saga"]},
                        "max_iter": {"type": "int", "value": 1000}
                    }
                },
                "random_forest": {
                    "class": "RandomForestClassifier",
                    "module": "sklearn.ensemble",
                    "hyperparams": {
                        "n_estimators": {"type": "int", "range": [50, 300]},
                        "max_depth": {"type": "int", "range": [3, 20]},
                        "min_samples_split": {"type": "int", "range": [2, 20]},
                        "min_samples_leaf": {"type": "int", "range": [1, 10]}
                    }
                },
                "xgboost": {
                    "class": "XGBClassifier",
                    "module": "xgboost",
                    "hyperparams": {
                        "n_estimators": {"type": "int", "range": [50, 300]},
                        "max_depth": {"type": "int", "range": [3, 10]},
                        "learning_rate": {"type": "float", "range": [0.01, 0.3]},
                        "subsample": {"type": "float", "range": [0.6, 1.0]},
                        "colsample_bytree": {"type": "float", "range": [0.6, 1.0]}
                    }
                },
                "lightgbm": {
                    "class": "LGBMClassifier",
                    "module": "lightgbm",
                    "hyperparams": {
                        "n_estimators": {"type": "int", "range": [50, 300]},
                        "max_depth": {"type": "int", "range": [3, 10]},
                        "learning_rate": {"type": "float", "range": [0.01, 0.3]},
                        "num_leaves": {"type": "int", "range": [20, 100]},
                        "feature_fraction": {"type": "float", "range": [0.6, 1.0]}
                    }
                },
                "catboost": {
                    "class": "CatBoostClassifier",
                    "module": "catboost",
                    "hyperparams": {
                        "iterations": {"type": "int", "range": [50, 300]},
                        "depth": {"type": "int", "range": [3, 10]},
                        "learning_rate": {"type": "float", "range": [0.01, 0.3]},
                        "l2_leaf_reg": {"type": "float", "range": [1, 10]},
                        "border_count": {"type": "int", "range": [32, 255]},
                        "verbose": {"type": "int", "value": 0}
                    },
                    "notes": "Excellent for categorical features, often wins competitions. Handles categorical variables natively."
                },
                "tabpfn": {
                    "class": "TabPFNClassifier",
                    "module": "tabpfn",
                    "hyperparams": {},
                    "notes": "No hyperparameter tuning needed - uses meta-learning. Best for datasets with <10K samples and <100 features."
                }
            },
            "regression": {
                "linear_regression": {
                    "class": "LinearRegression",
                    "module": "sklearn.linear_model",
                    "hyperparams": {}
                },
                "ridge": {
                    "class": "Ridge",
                    "module": "sklearn.linear_model",
                    "hyperparams": {
                        "alpha": {"type": "float", "range": [0.1, 100], "log": True}
                    }
                },
                "random_forest": {
                    "class": "RandomForestRegressor",
                    "module": "sklearn.ensemble",
                    "hyperparams": {
                        "n_estimators": {"type": "int", "range": [50, 300]},
                        "max_depth": {"type": "int", "range": [3, 20]},
                        "min_samples_split": {"type": "int", "range": [2, 20]}
                    }
                },
                "xgboost": {
                    "class": "XGBRegressor",
                    "module": "xgboost",
                    "hyperparams": {
                        "n_estimators": {"type": "int", "range": [50, 300]},
                        "max_depth": {"type": "int", "range": [3, 10]},
                        "learning_rate": {"type": "float", "range": [0.01, 0.3]}
                    }
                }
            },
            "survival": {
                "cox_ph": {
                    "class": "CoxPHSurvivalAnalysis",
                    "module": "sksurv.linear_model",
                    "hyperparams": {
                        "alpha": {"type": "float", "range": [0.001, 10], "log": True}
                    }
                },
                "coxnet": {
                    "class": "CoxnetSurvivalAnalysis",
                    "module": "sksurv.linear_model",
                    "hyperparams": {
                        "l1_ratio": {"type": "float", "range": [0.0, 1.0]},
                        "alpha_min_ratio": {"type": "float", "range": [0.0001, 0.1], "log": True},
                        "n_alphas": {"type": "int", "value": 100}
                    },
                    "notes": "Elastic Net regularized Cox model - excellent for high-dimensional data"
                },
                "random_survival_forest": {
                    "class": "RandomSurvivalForest",
                    "module": "sksurv.ensemble",
                    "hyperparams": {
                        "n_estimators": {"type": "int", "range": [50, 200]},
                        "max_depth": {"type": "int", "range": [3, 15]},
                        "min_samples_split": {"type": "int", "range": [6, 20]}
                    }
                },
                "gradient_boosting_survival": {
                    "class": "GradientBoostingSurvivalAnalysis",
                    "module": "sksurv.ensemble",
                    "hyperparams": {
                        "n_estimators": {"type": "int", "range": [50, 200]},
                        "learning_rate": {"type": "float", "range": [0.01, 0.2]},
                        "max_depth": {"type": "int", "range": [3, 8]}
                    }
                },
                "deepsurv": {
                    "class": "DeepSurvWrapper",
                    "module": "src.ml.deep_models",
                    "hyperparams": {
                        "num_nodes": {"type": "categorical", "values": [[32, 32], [64, 64], [128, 64, 32]]},
                        "dropout": {"type": "float", "range": [0.0, 0.5]},
                        "learning_rate": {"type": "float", "range": [0.001, 0.1], "log": True},
                        "batch_size": {"type": "categorical", "values": [64, 128, 256]},
                        "epochs": {"type": "int", "value": 100}
                    },
                    "notes": "Deep learning-based Cox model - handles complex non-linear relationships"
                }
            }
        }
    
    def get_task_type(self) -> TaskType:
        return TaskType.MODEL_SELECTION
    
    async def execute(self, inputs: Dict[str, Any]) -> AgentResult:
        """Execute model selection"""
        self.validate_inputs(inputs, ["data_analysis", "feature_info"])
        
        data_analysis = inputs["data_analysis"]
        feature_info = inputs["feature_info"]
        
        # Get LLM recommendations
        llm_recommendations = await self._get_llm_model_recommendations(
            data_analysis, feature_info
        )
        
        # Select models based on task type and LLM recommendations
        selected_models = self._select_models(data_analysis, llm_recommendations)
        
        result = {
            "selected_models": selected_models,
            "task_type": data_analysis.get("task_type"),
            "llm_recommendations": llm_recommendations,
            "model_configs": {model: self.model_configs[data_analysis.get("task_type", "classification")][model] 
                            for model in selected_models if model in self.model_configs.get(data_analysis.get("task_type", "classification"), {})},
            "evaluation_strategy": self._get_evaluation_strategy(data_analysis.get("task_type"))
        }
        
        return self.create_result(
            inputs=inputs,
            outputs=result,
            confidence_score=0.85
        )
    
    async def select_models(self, data_analysis: Dict[str, Any], feature_info: Dict[str, Any]) -> Dict[str, Any]:
        """Main entry point for model selection"""
        result = await self.execute({
            "data_analysis": data_analysis,
            "feature_info": feature_info
        })
        
        if result.status.value == "completed":
            return result.outputs
        else:
            raise Exception(f"Model selection failed: {result.error_message}")
    
    async def _get_llm_model_recommendations(
        self, 
        data_analysis: Dict[str, Any], 
        feature_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Get LLM recommendations for model selection"""
        self.log("Getting LLM recommendations for model selection...")
        
        try:
            # Prepare data characteristics
            data_characteristics = {
                "task_type": data_analysis.get("task_type"),
                "n_samples": feature_info.get("n_samples_train", 0),
                "n_features": feature_info.get("n_features", 0),
                "target_distribution": feature_info.get("data_quality_report", {}).get("target_distribution", {}),
                "categorical_features": len(feature_info.get("categorical_features", [])),
                "numerical_features": len(feature_info.get("numerical_features", [])),
                "data_quality_issues": data_analysis.get("data_quality", []),
                "domain": "clinical/medical"
            }
            
            # Use LLM for model suggestions
            recommendations = await self.llm_client.suggest_models(
                data_analysis.get("task_type", "classification"), 
                data_characteristics
            )
            
            self.log("LLM model recommendations obtained")
            return recommendations
            
        except Exception as e:
            self.log(f"LLM model recommendations failed: {str(e)}", "ERROR")
            return self._get_default_model_recommendations(data_analysis)
    
    def _get_default_model_recommendations(self, data_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Default model recommendations if LLM fails - uses smart heuristics based on data characteristics"""
        task_type = data_analysis.get("task_type", "classification")
        
        if task_type == "classification":
            return {
                "recommended_models": [
                    {"name": "autogluon", "priority": 1, "reasoning": "AutoML ensemble that combines multiple models with automatic preprocessing and tuning"},
                    {"name": "catboost", "priority": 2, "reasoning": "Excellent for categorical features, often wins competitions"},
                    {"name": "xgboost", "priority": 3, "reasoning": "Strong performance on tabular data"},
                    {"name": "logistic_regression", "priority": 4, "reasoning": "Good baseline for binary classification"},
                    {"name": "random_forest", "priority": 5, "reasoning": "Robust and handles mixed data types"}
                ],
                "evaluation_metrics": ["accuracy", "f1", "roc_auc", "precision", "recall"]
            }
        elif task_type == "regression":
            return {
                "recommended_models": [
                    {"name": "linear_regression", "priority": 1, "reasoning": "Good baseline"},
                    {"name": "random_forest", "priority": 2, "reasoning": "Handles non-linearity"},
                    {"name": "xgboost", "priority": 3, "reasoning": "Strong performance on tabular data"}
                ],
                "evaluation_metrics": ["mae", "mse", "rmse", "r2"]
            }
        else:  # survival
            return {
                "recommended_models": [
                    {"name": "cox_ph", "priority": 1, "reasoning": "Standard survival analysis baseline"},
                    {"name": "coxnet", "priority": 2, "reasoning": "Regularized Cox model for high-dimensional data"},
                    {"name": "random_survival_forest", "priority": 3, "reasoning": "Non-parametric survival model"},
                    {"name": "deepsurv", "priority": 4, "reasoning": "Deep learning for complex non-linear relationships"}
                ],
                "evaluation_metrics": ["concordance_index", "integrated_brier_score"]
            }
    
    def _select_models(self, data_analysis: Dict[str, Any], llm_recommendations: Dict[str, Any]) -> List[str]:
        """Select final list of models to train - default 3 models, max 5"""
        task_type = data_analysis.get("task_type", "classification")
        
        # Get recommended models from LLM
        recommended = llm_recommendations.get("recommended_models", [])
        
        # Determine optimal number of models based on config or default to 3
        min_models = getattr(self.config.ml, 'min_models_to_train', 3)
        max_models = getattr(self.config.ml, 'max_models_to_train', 5)
        
        # Extract model names and prioritize
        if isinstance(recommended, list) and len(recommended) > 0:
            if isinstance(recommended[0], dict):
                # LLM returned structured recommendations
                model_names = [model["name"] for model in recommended[:max_models]]
            else:
                # LLM returned simple list
                model_names = recommended[:max_models]
        else:
            # Fallback to config defaults
            model_names = list(self.config.ml.models_to_try)
        
        # Filter to models available for this task type
        available_models = set(self.model_configs.get(task_type, {}).keys())
        selected = [model for model in model_names if model in available_models]
        
        # Ensure we have at least min_models
        if len(selected) < min_models:
            # Add more models from available set prioritizing diversity
            fallback_models = list(available_models - set(selected))
            selected.extend(fallback_models[:min_models - len(selected)])
        
        # Limit to max_models
        selected = selected[:max_models]
        
        self.log(f"Selected {len(selected)} models for {task_type}: {selected}")
        return selected
    
    def _get_evaluation_strategy(self, task_type: str) -> Dict[str, Any]:
        """Get evaluation strategy for the task type"""
        
        if task_type == "classification":
            return {
                "metrics": ["accuracy", "precision", "recall", "f1", "roc_auc"],
                "primary_metric": "roc_auc",
                "cv_strategy": "stratified_k_fold",
                "cv_folds": self.config.ml.cv_folds
            }
        elif task_type == "regression":
            return {
                "metrics": ["mae", "mse", "rmse", "r2"],
                "primary_metric": "r2",
                "cv_strategy": "k_fold",
                "cv_folds": self.config.ml.cv_folds
            }
        else:  # survival
            return {
                "metrics": ["concordance_index", "integrated_brier_score"],
                "primary_metric": "concordance_index",
                "cv_strategy": "k_fold",
                "cv_folds": self.config.ml.cv_folds
            }
    
    def get_model_class(self, model_name: str, task_type: str):
        """Get the actual model class for instantiation"""
        try:
            config = self.model_configs[task_type][model_name]
            module_name = config["module"]
            class_name = config["class"]
            
            # Import the module and get the class
            import importlib
            module = importlib.import_module(module_name)
            model_class = getattr(module, class_name)
            
            return model_class
            
        except Exception as e:
            self.log(f"Failed to import {model_name}: {str(e)}", "ERROR")
            return None
    
    def get_hyperparameter_space(self, model_name: str, task_type: str) -> Dict[str, Any]:
        """Get hyperparameter search space for a model"""
        try:
            return self.model_configs[task_type][model_name]["hyperparams"]
        except KeyError:
            self.log(f"No hyperparameters defined for {model_name}", "WARNING")
            return {}
