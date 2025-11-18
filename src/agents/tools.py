"""Tool definitions and executors for the ReAct ML agent"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
import json

from ..core.config import Config
from ..data.analyzer import DataAnalyzer
from ..ml.feature_engineer import FeatureEngineer
from ..ml.model_selector import ModelSelector
from ..ml.trainer import ModelTrainer
from .error_analyzer import ErrorAnalyzer


class MLToolkit:
    """
    Toolkit of ML functions that the ReAct agent can use.
    Each tool is a function the LLM can call with specific parameters.
    """
    
    def __init__(self, config: Config):
        self.config = config
        
        # Initialize agents
        self.data_analyzer = DataAnalyzer(config)
        self.feature_engineer = FeatureEngineer(config)
        self.model_selector = ModelSelector(config)
        self.model_trainer = ModelTrainer(config)
        self.error_analyzer = ErrorAnalyzer(config)
        
        # State storage
        self.state = {
            "dataset_path": None,
            "testset_path": None,
            "objective": None,
            "data_analysis": None,
            "feature_result": None,
            "trained_models": {},
            "evaluation_results": {},
            "error_analyses": {},
            "best_score": 0.0,
            "best_model": None
        }
    
    def get_tool_definitions(self) -> List[Dict[str, Any]]:
        """
        Get OpenAI function calling tool definitions
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": "analyze_data",
                    "description": "Analyze the dataset to understand its structure, identify the target variable, determine task type (classification/regression), and get feature information. This should be your first step.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "dataset_path": {
                                "type": "string",
                                "description": "Path to the dataset file"
                            },
                            "objective": {
                                "type": "string",
                                "description": "The ML objective or what you're trying to predict"
                            }
                        },
                        "required": ["dataset_path", "objective"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "engineer_features",
                    "description": "Apply feature engineering: preprocessing, scaling, encoding, and transformations. You can specify strategies based on data analysis results.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "scaling_strategy": {
                                "type": "string",
                                "enum": ["standard", "minmax", "robust"],
                                "description": "How to scale numerical features"
                            },
                            "encoding_strategy": {
                                "type": "string",
                                "enum": ["onehot", "label"],
                                "description": "How to encode categorical features"
                            },
                            "handle_imbalance": {
                                "type": "boolean",
                                "description": "Whether to apply techniques for imbalanced classes"
                            }
                        },
                        "required": []
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "select_models",
                    "description": "Select which ML models to train based on the task type and data characteristics. Automatically chooses appropriate models based on dataset size. Returns recommended models.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "prefer_simple": {
                                "type": "boolean",
                                "description": "Optional: prefer simpler models. Generally not needed as model selection is automatic based on dataset size."
                            },
                            "prefer_interpretable": {
                                "type": "boolean",
                                "description": "If true, prioritize interpretable models (e.g., linear models, tree-based) over black-box models"
                            }
                        },
                        "required": []
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "train_model",
                    "description": "Train a specific model with optional hyperparameter suggestions. The model will be optimized using cross-validation.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "model_name": {
                                "type": "string",
                                "description": "Name of model to train (e.g., 'logistic_regression', 'random_forest', 'xgboost')"
                            },
                            "quick_mode": {
                                "type": "boolean",
                                "description": "If true, use fewer optimization trials for faster training"
                            }
                        },
                        "required": ["model_name"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "evaluate_model",
                    "description": "Evaluate a trained model on the test set and get performance metrics.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "model_name": {
                                "type": "string",
                                "description": "Name of the trained model to evaluate"
                            }
                        },
                        "required": ["model_name"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "analyze_errors",
                    "description": "Analyze which predictions the model got wrong to understand failure patterns. Very useful for determining how to improve the model.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "model_name": {
                                "type": "string",
                                "description": "Name of the model to analyze"
                            },
                            "n_examples": {
                                "type": "integer",
                                "description": "Number of error examples to show (default 10)"
                            }
                        },
                        "required": ["model_name"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_feature_importance",
                    "description": "Get feature importance scores to understand which features the model relies on most.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "model_name": {
                                "type": "string",
                                "description": "Name of the trained model"
                            },
                            "top_n": {
                                "type": "integer",
                                "description": "Number of top features to return (default 10)"
                            }
                        },
                        "required": ["model_name"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_current_state",
                    "description": "Get summary of current progress: what's been done, best results so far, etc.",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "refine_features",
                    "description": "Refine feature engineering based on model performance. Use this when you want to improve features after seeing model results. The LLM will analyze current performance and suggest new feature engineering strategies.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "performance_feedback": {
                                "type": "string",
                                "description": "Brief description of what you observed about model performance (e.g., 'Low accuracy on test set', 'Features seem redundant', 'Need better risk stratification')"
                            },
                            "focus_areas": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Optional: Specific areas to focus on (e.g., ['feature_interactions', 'feature_selection', 'transformations'])"
                            }
                        },
                        "required": ["performance_feedback"]
                    }
                }
            }
        ]
    
    async def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a tool and return results
        """
        try:
            if tool_name == "analyze_data":
                return await self._analyze_data(**arguments)
            elif tool_name == "engineer_features":
                return await self._engineer_features(**arguments)
            elif tool_name == "select_models":
                return await self._select_models(**arguments)
            elif tool_name == "train_model":
                return await self._train_model(**arguments)
            elif tool_name == "evaluate_model":
                return await self._evaluate_model(**arguments)
            elif tool_name == "analyze_errors":
                return await self._analyze_errors(**arguments)
            elif tool_name == "get_feature_importance":
                return await self._get_feature_importance(**arguments)
            elif tool_name == "get_current_state":
                return self._get_current_state(**arguments)
            elif tool_name == "refine_features":
                return await self._refine_features(**arguments)
            else:
                return {"error": f"Unknown tool: {tool_name}"}
        except Exception as e:
            return {
                "error": str(e),
                "tool": tool_name,
                "arguments": arguments
            }
    
    async def _analyze_data(self, dataset_path: str, objective: str) -> Dict[str, Any]:
        """Analyze dataset"""
        print(f"  🔧 Executing: analyze_data({dataset_path})")
        
        self.state["dataset_path"] = dataset_path
        self.state["objective"] = objective
        
        data_analysis = await self.data_analyzer.analyze_dataset(dataset_path, objective)
        self.state["data_analysis"] = data_analysis
        
        # Return summary for LLM
        task_type = data_analysis.get("task_type")
        message = (f"Analyzed dataset with {data_analysis.get('n_rows')} rows. "
                  f"Suggested target: '{data_analysis.get('suggested_target')}'. "
                  f"Task type: {task_type}.")
        
        # Add survival info if applicable
        if task_type == "survival" and data_analysis.get("time_variable"):
            message += f" Time variable: '{data_analysis.get('time_variable')}'."
        
        return {
            "success": True,
            "summary": {
                "n_rows": data_analysis.get("n_rows"),
                "n_cols": data_analysis.get("n_cols"),
                "suggested_target": data_analysis.get("suggested_target"),
                "task_type": task_type,
                "n_features": len(data_analysis.get("suggested_features", [])),
                "n_categorical": len(data_analysis.get("categorical_features", [])),
                "n_numerical": len(data_analysis.get("numerical_features", [])),
                "data_quality_issues": data_analysis.get("data_quality", [])[:3],  # Top 3 issues
                "time_variable": data_analysis.get("time_variable") if task_type == "survival" else None
            },
            "message": message
        }
    
    async def _engineer_features(
        self,
        scaling_strategy: str = "standard",
        encoding_strategy: str = "onehot",
        handle_imbalance: bool = False
    ) -> Dict[str, Any]:
        """Engineer features"""
        print(f"  🔧 Executing: engineer_features(scaling={scaling_strategy}, encoding={encoding_strategy})")
        
        if self.state["data_analysis"] is None:
            return {"error": "Must run analyze_data first"}
        
        data_analysis = self.state["data_analysis"]
        target = data_analysis.get("suggested_target")
        
        # Override LLM recommendations
        llm_recommendations = {
            "numerical_transformations": {
                "scaling_strategy": scaling_strategy,
                "imputation_strategy": "median"
            },
            "categorical_encoding": {
                "low_cardinality": encoding_strategy,
                "imputation_strategy": "most_frequent"
            }
        }
        
        # Get time variable for survival tasks
        time_variable = data_analysis.get("time_variable") if data_analysis.get("task_type") == "survival" else None
        
        feature_result = await self.feature_engineer.engineer_features(
            self.state["dataset_path"],
            data_analysis,
            target,
            time_variable=time_variable,
            testset_path=self.state.get("testset_path")
        )
        
        # Update the recommendations in the result
        feature_result["llm_recommendations"] = llm_recommendations
        
        self.state["feature_result"] = feature_result
        
        return {
            "success": True,
            "summary": {
                "n_features": feature_result.get("n_features"),
                "n_train_samples": feature_result.get("n_samples_train"),
                "n_test_samples": feature_result.get("n_samples_test"),
                "scaling_used": scaling_strategy,
                "encoding_used": encoding_strategy
            },
            "message": f"Feature engineering complete. Created {feature_result.get('n_features')} features "
                      f"from {feature_result.get('n_samples_train')} training samples."
        }
    
    async def _select_models(
        self,
        prefer_simple: bool = False,
        prefer_interpretable: bool = False
    ) -> Dict[str, Any]:
        """Select models"""
        print(f"  🔧 Executing: select_models(simple={prefer_simple}, interpretable={prefer_interpretable})")
        
        if self.state["feature_result"] is None:
            return {"error": "Must run engineer_features first"}
        
        data_analysis = self.state["data_analysis"]
        feature_result = self.state["feature_result"]
        
        n_samples = feature_result.get("n_samples_train", 0)
        task_type = data_analysis.get("task_type", "classification")
        
        # Customize model selection based on preferences and data size
        # Only force simple models for truly small datasets (< 200 samples)
        # Ignore prefer_simple flag for larger datasets - let the model selector decide
        if n_samples < 200:
            # Force simple models for small datasets - task type aware
            if task_type == "survival":
                simple_models = ["cox_ph", "random_survival_forest"]
            elif task_type == "regression":
                simple_models = ["linear_regression", "ridge"]
            else:  # classification
                simple_models = ["logistic_regression", "random_forest"]
            
            reason = "user preference" if prefer_simple else "small dataset"
            return {
                "success": True,
                "selected_models": simple_models,
                "message": f"Selected simple models due to {reason} ({n_samples} samples): {simple_models}"
            }
        
        selection_result = await self.model_selector.select_models(
            data_analysis, feature_result
        )
        
        selected = selection_result.get("selected_models", [])
        
        if prefer_interpretable:
            # Prioritize interpretable models - task type aware
            if task_type == "survival":
                interpretable = ["cox_ph", "random_survival_forest"]
            elif task_type == "regression":
                interpretable = ["linear_regression", "ridge"]
            else:  # classification
                interpretable = ["logistic_regression", "random_forest"]
            
            selected = [m for m in interpretable if m in selected] + \
                      [m for m in selected if m not in interpretable]
            selected = selected[:3]  # Limit to top 3
        
        return {
            "success": True,
            "selected_models": selected,
            "task_type": selection_result.get("task_type"),
            "message": f"Selected {len(selected)} models to train: {selected}"
        }
    
    async def _train_model(
        self,
        model_name: str,
        quick_mode: bool = False
    ) -> Dict[str, Any]:
        """Train a single model"""
        print(f"  🔧 Executing: train_model({model_name}, quick={quick_mode})")
        
        if self.state["feature_result"] is None:
            return {"error": "Must run engineer_features first"}
        
        # Temporarily reduce trials if quick mode
        original_trials = self.config.ml.optuna_trials
        if quick_mode:
            self.config.ml.optuna_trials = min(20, original_trials)
        
        try:
            training_result = await self.model_trainer.train_models(
                self.state["feature_result"],
                [model_name]
            )
            
            trained_models = training_result.get("trained_models", {})
            if model_name in trained_models:
                model_info = trained_models[model_name]
                self.state["trained_models"][model_name] = model_info
                
                cv_score = model_info.get("cv_score", 0)
                training_time = model_info.get("training_time", 0)
                
                return {
                    "success": True,
                    "model_name": model_name,
                    "cv_score": float(cv_score),
                    "training_time": float(training_time),
                    "message": f"Trained {model_name}. CV score: {cv_score:.3f}, Time: {training_time:.1f}s"
                }
            else:
                return {"error": f"Failed to train {model_name}"}
        finally:
            # Restore original trials
            self.config.ml.optuna_trials = original_trials
    
    async def _evaluate_model(self, model_name: str) -> Dict[str, Any]:
        """Evaluate model on test set"""
        print(f"  🔧 Executing: evaluate_model({model_name})")
        
        if model_name not in self.state["trained_models"]:
            return {"error": f"Model {model_name} not trained yet"}
        
        # Create minimal training result for evaluation
        training_result = {
            "trained_models": {model_name: self.state["trained_models"][model_name]},
            "task_type": self.state["feature_result"]["task_type"],
            "feature_data": self.state["feature_result"]
        }
        
        evaluation_result = await self.model_trainer.evaluate_models(training_result)
        
        eval_info = evaluation_result["evaluation_results"].get(model_name, {})
        metrics = eval_info.get("metrics", {})
        
        # Store evaluation
        self.state["evaluation_results"][model_name] = eval_info
        
        # Update best score
        task_type = self.state["feature_result"]["task_type"]
        if task_type == "classification":
            primary_metric = "accuracy"
            score = metrics.get(primary_metric, 0)
        elif task_type == "survival":
            primary_metric = "concordance_index"
            score = metrics.get(primary_metric, 0)
        else:  # regression
            primary_metric = "r2"
            score = metrics.get(primary_metric, 0)
        
        if score > self.state["best_score"]:
            self.state["best_score"] = score
            self.state["best_model"] = model_name
        
        return {
            "success": True,
            "model_name": model_name,
            "metrics": metrics,
            "primary_metric": primary_metric,
            "primary_score": float(score),
            "is_best": score == self.state["best_score"],
            "message": f"Evaluated {model_name}. {primary_metric}: {score:.3f}"
        }
    
    async def _analyze_errors(
        self,
        model_name: str,
        n_examples: int = 10
    ) -> Dict[str, Any]:
        """Analyze model errors"""
        print(f"  🔧 Executing: analyze_errors({model_name})")
        
        if model_name not in self.state["trained_models"]:
            return {"error": f"Model {model_name} not trained yet"}
        
        model_info = self.state["trained_models"][model_name]
        model = model_info["model"]
        
        feature_data = self.state["feature_result"]
        data_splits = feature_data["data_splits"]
        
        X_test = data_splits["X_test"]
        y_test = data_splits["y_test"]
        
        # Check if this is survival data (structured array)
        is_survival = hasattr(y_test, 'dtype') and y_test.dtype.names is not None
        
        if is_survival:
            # Get the score, ensuring it's not None
            score = model_info.get("val_score") or model_info.get("cv_score") or 0.0
            return {
                "error": "Error analysis is not yet implemented for survival models. "
                        "For survival analysis, focus on improving concordance index (C-index). "
                        f"Current C-index: {score:.3f}. Suggestions: Try feature selection, "
                        "different time discretization, or ensemble methods."
            }
        
        # Convert X_test to DataFrame for AutoGluon
        X_test_input = X_test
        if model_name == "autogluon" or "TabularPredictor" in str(type(model)):
            import pandas as pd
            # AutoGluon requires DataFrame
            X_test_input = pd.DataFrame(X_test)
            feature_names = feature_data.get("feature_names")
            if feature_names and len(feature_names) == X_test.shape[1]:
                X_test_input.columns = feature_names
        
        # Get predictions
        y_pred = model.predict(X_test_input)
        y_pred_proba = None
        if hasattr(model, "predict_proba"):
            y_pred_proba = model.predict_proba(X_test_input)
        
        # Analyze errors
        error_analysis = self.error_analyzer.analyze_misclassifications(
            X_test, y_test, y_pred,
            feature_data.get("feature_names", []),
            y_pred_proba,
            n_examples
        )
        
        # Generate suggestions
        suggestions = self.error_analyzer.generate_improvement_suggestions(error_analysis)
        error_analysis["suggestions"] = suggestions
        
        # Store analysis
        self.state["error_analyses"][model_name] = error_analysis
        
        # Create summary for LLM
        summary = {
            "error_rate": error_analysis.get("error_rate", 0),
            "total_errors": error_analysis.get("total_errors", 0),
            "class_errors": error_analysis.get("class_errors", {}),
            "top_confusion": list(error_analysis.get("confusion_patterns", {}).values())[:2],
            "suggestions": suggestions[:3]  # Top 3 suggestions
        }
        
        return {
            "success": True,
            "model_name": model_name,
            "summary": summary,
            "message": f"Error analysis complete. Error rate: {error_analysis.get('error_rate', 0):.1%}. "
                      f"Found {len(suggestions)} improvement suggestions."
        }
    
    async def _get_feature_importance(
        self,
        model_name: str,
        top_n: int = 10
    ) -> Dict[str, Any]:
        """Get feature importance"""
        print(f"  🔧 Executing: get_feature_importance({model_name}, top_n={top_n})")
        
        if model_name not in self.state["trained_models"]:
            return {"error": f"Model {model_name} not trained yet"}
        
        model_info = self.state["trained_models"][model_name]
        model = model_info["model"]
        feature_names = self.state["feature_result"].get("feature_names", [])
        
        importances = None
        
        # Try different methods to get feature importance
        try:
            # Special handling for AutoGluon
            if model_name == "autogluon" or "TabularPredictor" in str(type(model)):
                # AutoGluon has a feature_importance() method
                importance_df = model.feature_importance(data=None, subsample_size=1000)
                importances = importance_df['importance'].values
                # Use AutoGluon's feature names (may differ from original after preprocessing)
                feature_names = importance_df['feature'].values.tolist()
            elif hasattr(model, "feature_importances_"):
                importances = model.feature_importances_
            elif hasattr(model, "coef_"):
                # For Cox models and other linear models
                coef = model.coef_
                # Handle different coefficient shapes
                if hasattr(coef, 'ndim'):
                    if coef.ndim > 1:
                        importances = np.abs(coef[0])
                    else:
                        importances = np.abs(coef)
                else:
                    importances = np.abs(np.array(coef))
            elif hasattr(model, "params_"):
                # Some survival models store coefficients in params_
                importances = np.abs(model.params_)
            else:
                return {
                    "success": False,
                    "message": f"Model {model_name} does not support feature importance extraction"
                }
            
            # Ensure importances is a numpy array
            if not isinstance(importances, np.ndarray):
                importances = np.array(importances)
            
            # Flatten if necessary
            if importances.ndim > 1:
                importances = importances.flatten()
            
            # Sort by importance
            if len(importances) == 0:
                return {
                    "success": False,
                    "message": f"No feature importances available for {model_name}"
                }
            
            indices = np.argsort(importances)[::-1][:top_n]
            
            top_features = []
            for idx in indices:
                if idx < len(feature_names) and idx < len(importances):
                    top_features.append({
                        "feature": feature_names[idx],
                        "importance": float(importances[idx])
                    })
            
            return {
                "success": True,
                "model_name": model_name,
                "top_features": top_features,
                "message": f"Top {len(top_features)} important features for {model_name}"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to extract feature importance: {str(e)}",
                "message": f"Could not get feature importance for {model_name}"
            }
    
    def _get_current_state(self) -> Dict[str, Any]:
        """Get current progress summary"""
        print(f"  🔧 Executing: get_current_state()")
        
        return {
            "success": True,
            "state": {
                "dataset_analyzed": self.state["data_analysis"] is not None,
                "features_engineered": self.state["feature_result"] is not None,
                "n_models_trained": len(self.state["trained_models"]),
                "trained_models": list(self.state["trained_models"].keys()),
                "best_model": self.state["best_model"],
                "best_score": float(self.state["best_score"]),
                "models_evaluated": list(self.state["evaluation_results"].keys()),
                "error_analyses_done": list(self.state["error_analyses"].keys())
            },
            "message": f"Progress: {len(self.state['trained_models'])} models trained. "
                      f"Best: {self.state['best_model']} ({self.state['best_score']:.3f})"
        }
    
    async def _refine_features(
        self,
        performance_feedback: str,
        focus_areas: List[str] = None
    ) -> Dict[str, Any]:
        """
        Refine feature engineering based on model performance feedback.
        
        This implements iterative feature engineering refinement by:
        1. Analyzing current performance
        2. Getting LLM suggestions for improvements
        3. Re-running feature engineering with new configuration
        4. Re-training models (agent's responsibility)
        """
        print(f"  🔧 Executing: refine_features(feedback='{performance_feedback[:50]}...')")
        
        if self.state["feature_result"] is None:
            return {"error": "Must run engineer_features first before refining"}
        
        if not self.state["trained_models"]:
            return {"error": "Must train at least one model before refining features"}
        
        # Gather context for LLM
        current_config = self.state["feature_result"].get("llm_recommendations", {})
        current_performance = {
            "best_model": self.state["best_model"],
            "best_score": self.state["best_score"],
            "all_models": {
                name: {
                    "cv_score": info.get("cv_score", 0),
                    "training_time": info.get("training_time", 0)
                }
                for name, info in self.state["trained_models"].items()
            }
        }
        
        # Get feature engineering history if available
        feature_history = self.feature_engineer.feature_engineering_history
        
        # Build critique prompt
        critique_context = {
            "current_feature_config": current_config,
            "current_performance": current_performance,
            "performance_feedback": performance_feedback,
            "focus_areas": focus_areas or ["feature_interactions", "feature_selection", "transformations"],
            "task_type": self.state["feature_result"].get("task_type"),
            "n_current_features": self.state["feature_result"].get("n_features"),
            "feature_creation_report": self.state["feature_result"].get("feature_creation_report"),
            "feature_selection_report": self.state["feature_result"].get("feature_selection_report"),
            "previous_iterations": len(feature_history)
        }
        
        # Get LLM suggestions for improvement
        self.log("Getting LLM suggestions for feature engineering improvements...")
        
        try:
            # Create prompt for feature improvement
            improvement_prompt = f"""
You are a feature engineering expert. Analyze the current model performance and suggest improvements.

CURRENT SITUATION:
- Task: {critique_context['task_type']}
- Current features: {critique_context['n_current_features']}
- Best model: {current_performance['best_model']}
- Best score: {current_performance['best_score']:.3f}
- Performance feedback: {performance_feedback}
- Iterations so far: {critique_context['previous_iterations']}

CURRENT FEATURE ENGINEERING CONFIG:
{json.dumps(current_config, indent=2)}

FOCUS AREAS: {', '.join(critique_context['focus_areas'])}

Based on this, suggest SPECIFIC improvements to feature engineering. Consider:
1. Are there important feature interactions we're missing?
2. Should we add/remove certain transformations?
3. Is feature selection too aggressive or too lenient?
4. Are there domain-specific features (oncology) we should create?

Provide a COMPLETE new feature engineering configuration in the same JSON format as the current config.
Be specific about which features to interact, transform, or select.
"""
            
            # Use LLM to get improvement suggestions
            from ..llm.client import LLMClient
            llm = LLMClient(self.config.llm.model, self.config.llm.api_key)
            
            improved_config = await llm.complete_json(
                improvement_prompt,
                system_message="You are a feature engineering expert. Provide detailed, actionable feature engineering improvements."
            )
            
            self.log("LLM improvement suggestions obtained")
            
        except Exception as e:
            self.log(f"LLM improvement suggestions failed: {str(e)}", level="WARNING")
            # Fallback: make conservative improvements
            improved_config = current_config.copy()
            
            # Adjust feature selection to be more/less aggressive
            if "too many features" in performance_feedback.lower() or "overfitting" in performance_feedback.lower():
                if "feature_selection" in improved_config:
                    improved_config["feature_selection"]["max_features"] = int(
                        critique_context['n_current_features'] * 0.7
                    )
            elif "underfitting" in performance_feedback.lower() or "more features" in performance_feedback.lower():
                if "feature_selection" in improved_config:
                    improved_config["feature_selection"]["max_features"] = None
        
        # Re-run feature engineering with improved configuration
        self.log("Re-running feature engineering with improved configuration...")
        
        data_analysis = self.state["data_analysis"]
        target = data_analysis.get("suggested_target")
        time_variable = data_analysis.get("time_variable") if data_analysis.get("task_type") == "survival" else None
        
        try:
            feature_result = await self.feature_engineer.engineer_features(
                self.state["dataset_path"],
                data_analysis,
                target,
                time_variable=time_variable,
                testset_path=self.state.get("testset_path"),
                feature_engineering_config=improved_config
            )
            
            # Update state
            old_n_features = self.state["feature_result"].get("n_features")
            self.state["feature_result"] = feature_result
            new_n_features = feature_result.get("n_features")
            
            # Clear trained models since features changed
            self.log("⚠️  Features changed - previous models are no longer compatible. You'll need to retrain.")
            old_models = list(self.state["trained_models"].keys())
            self.state["trained_models"] = {}
            self.state["evaluation_results"] = {}
            self.state["best_score"] = 0.0
            self.state["best_model"] = None
            
            return {
                "success": True,
                "summary": {
                    "old_n_features": old_n_features,
                    "new_n_features": new_n_features,
                    "feature_change": new_n_features - old_n_features,
                    "iteration": len(feature_history),
                    "cleared_models": old_models
                },
                "message": f"Feature engineering refined! Features: {old_n_features} → {new_n_features}. "
                          f"Previous models cleared. Please retrain models with new features."
            }
            
        except Exception as e:
            self.log(f"Feature refinement failed: {str(e)}", level="ERROR")
            return {
                "error": f"Failed to refine features: {str(e)}",
                "suggestion": "Try running engineer_features again with default settings"
            }
    
    def log(self, message: str, level: str = "INFO"):
        """Helper logging method"""
        print(f"  [{level}] {message}")

