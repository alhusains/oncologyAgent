"""Model training module with hyperparameter optimization"""

# CRITICAL: Set thread limits BEFORE any numerical library imports
# This prevents "Resource temporarily unavailable" errors on shared clusters
import os
# Use ML_N_JOBS env var if set, otherwise default to 2 (safe for shared clusters)
_thread_limit = os.environ.get("ML_N_JOBS", "2")
os.environ.setdefault("OMP_NUM_THREADS", _thread_limit)
os.environ.setdefault("OPENBLAS_NUM_THREADS", _thread_limit)
os.environ.setdefault("MKL_NUM_THREADS", _thread_limit)
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", _thread_limit)
os.environ.setdefault("NUMEXPR_NUM_THREADS", _thread_limit)

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold, PredefinedSplit, cross_validate
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_absolute_error, mean_squared_error, r2_score
)
import optuna
import pickle
import time
from pathlib import Path

from ..core.base_agent import LLMAgent
from ..core.state import TaskType, AgentResult
from ..core.config import Config
from .model_selector import ModelSelector


class ModelTrainer(LLMAgent):
    """Agent for training and evaluating machine learning models"""
    
    def __init__(self, config: Config):
        super().__init__("model_trainer", config)
        self.selector = ModelSelector(config)
        self.trained_models = {}
        self.evaluation_results = {}
        
    def get_task_type(self) -> TaskType:
        return TaskType.MODEL_TRAINING
    
    async def execute(self, inputs: Dict[str, Any]) -> AgentResult:
        """Execute model training"""
        self.validate_inputs(inputs, ["feature_data", "selected_models"])
        
        feature_data = inputs["feature_data"]
        selected_models = inputs["selected_models"]
        
        # Train models
        training_results = await self._train_models(feature_data, selected_models)
        
        return self.create_result(
            inputs=inputs,
            outputs=training_results,
            confidence_score=0.9
        )
    
    async def train_models(self, feature_data: Dict[str, Any], selected_models: List[str]) -> Dict[str, Any]:
        """Main entry point for model training"""
        result = await self.execute({
            "feature_data": feature_data,
            "selected_models": selected_models
        })
        
        if result.status.value == "completed":
            return result.outputs
        else:
            raise Exception(f"Model training failed: {result.error_message}")
    
    async def evaluate_models(self, training_result: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate trained models on test set"""
        self.log("Evaluating models on test set...")
        
        try:
            trained_models = training_result.get("trained_models", {})
            feature_data = training_result.get("feature_data", {})
            task_type = training_result.get("task_type", "classification")
            
            # Get test data
            data_splits = feature_data.get("data_splits", {})
            X_test = data_splits.get("X_test")
            y_test = data_splits.get("y_test")
            
            if X_test is None or y_test is None:
                raise ValueError("Test data not available")
            
            # Evaluate each model
            evaluation_results = {}
            best_model = None
            best_score = -np.inf
            
            for model_name, model_info in trained_models.items():
                self.log(f"Evaluating {model_name}...")
                
                model = model_info["model"]
                
                # Convert X_test to DataFrame for AutoGluon
                X_test_input = X_test
                if model_name == "autogluon" or "TabularPredictor" in str(type(model)):
                    # AutoGluon requires DataFrame
                    X_test_input = pd.DataFrame(X_test)
                    feature_names = feature_data.get("feature_names")
                    if feature_names and len(feature_names) == X_test.shape[1]:
                        X_test_input.columns = feature_names
                
                # Make predictions
                try:
                    if task_type == "classification":
                        y_pred = model.predict(X_test_input)
                        y_pred_proba = None
                        if hasattr(model, "predict_proba"):
                            y_pred_proba = model.predict_proba(X_test_input)
                        
                        # Calculate metrics
                        metrics = self._calculate_classification_metrics(y_test, y_pred, y_pred_proba)
                        
                    elif task_type == "regression":
                        y_pred = model.predict(X_test_input)
                        metrics = self._calculate_regression_metrics(y_test, y_pred)
                        
                    else:  # survival
                        # For survival analysis, need training data for IBS
                        y_train = data_splits.get("y_train")
                        # Survival models don't use AutoGluon, so use original X_test
                        metrics = self._calculate_survival_metrics(model, X_test, y_test, y_train)
                    
                    evaluation_results[model_name] = {
                        "metrics": metrics,
                        "cv_score": model_info.get("cv_score", 0),
                        "training_time": model_info.get("training_time", 0),
                        "hyperparameters": model_info.get("best_params", {})
                    }
                    
                    # Track best model
                    primary_metric = self._get_primary_metric(task_type, y_test)
                    score = metrics.get(primary_metric, 0)
                    if score > best_score:
                        best_score = score
                        best_model = model_name
                    
                except Exception as e:
                    self.log(f"Evaluation failed for {model_name}: {str(e)}", "ERROR")
                    evaluation_results[model_name] = {
                        "metrics": {},
                        "error": str(e)
                    }
            
            result = {
                "evaluation_results": evaluation_results,
                "best_model": best_model,
                "best_score": best_score,
                "task_type": task_type,
                "test_set_size": len(y_test),
                "feature_data": feature_data
            }
            
            self.log(f"Model evaluation completed. Best model: {best_model} (score: {best_score:.3f})")
            return result
            
        except Exception as e:
            self.log(f"Model evaluation failed: {str(e)}", "ERROR")
            raise
    
    async def _train_models(self, feature_data: Dict[str, Any], selected_models: List[str]) -> Dict[str, Any]:
        """Train selected models with hyperparameter optimization"""
        self.log(f"Training {len(selected_models)} models...")
        
        # Get data
        data_splits = feature_data.get("data_splits", {})
        X_train = data_splits.get("X_train")
        y_train = data_splits.get("y_train")
        X_val = data_splits.get("X_val")
        y_val = data_splits.get("y_val")
        cv_groups = data_splits.get("cv_groups")  # Preset CV groups if available
        task_type = feature_data.get("task_type", "classification")
        
        if X_train is None or y_train is None:
            raise ValueError("Training data not available")
        
        # Log if using preset CV
        if cv_groups is not None:
            n_folds = len(np.unique(cv_groups))
            self.log(f"Using preset CV groups with {n_folds} folds from dataset")
        
        # Special handling for AutoGluon (classification only)
        if task_type == "classification" and "autogluon" in selected_models:
            self.log("Using AutoGluon for classification - training ensemble of models")
            return await self._train_autogluon(X_train, y_train, X_val, y_val, feature_data)
        
        trained_models = {}
        
        for model_name in selected_models:
            self.log(f"Training {model_name}...")
            
            try:
                start_time = time.time()
                
                # Get model class and hyperparameter space
                model_class = self.selector.get_model_class(model_name, task_type)
                if model_class is None:
                    self.log(f"Could not load {model_name}, skipping", "WARNING")
                    continue
                
                param_space = self.selector.get_hyperparameter_space(model_name, task_type)
                
                # Optimize hyperparameters
                best_params, best_score, cv_metrics = await self._optimize_hyperparameters(
                    model_class, param_space, X_train, y_train, task_type, cv_groups
                )
                
                # Inject thread limits into params BEFORE final model creation
                final_params = best_params.copy()
                model_class_name = model_class.__name__.lower()
                if 'catboost' in model_class_name:
                    final_params['thread_count'] = self.config.ml.n_jobs
                elif 'xgb' in model_class_name:
                    final_params['nthread'] = self.config.ml.n_jobs
                    final_params['n_jobs'] = self.config.ml.n_jobs
                elif 'lgbm' in model_class_name or 'lightgbm' in model_class_name:
                    final_params['n_jobs'] = self.config.ml.n_jobs
                elif 'randomforest' in model_class_name or 'forest' in model_class_name:
                    final_params['n_jobs'] = self.config.ml.n_jobs
                
                # Train final model with best parameters
                final_model = model_class(**final_params)
                
                # Set random state if available
                if hasattr(final_model, 'random_state'):
                    final_model.set_params(random_state=self.config.data.random_state)
                
                # Handle special cases for certain models
                if 'verbose' in final_model.get_params():
                    final_model.set_params(verbose=0)
                
                final_model.fit(X_train, y_train)
                
                training_time = time.time() - start_time
                
                # Validate on validation set if available
                val_score = None
                if X_val is not None and y_val is not None:
                    if task_type == "classification":
                        val_pred = final_model.predict(X_val)
                        val_score = accuracy_score(y_val, val_pred)
                    elif task_type == "regression":
                        val_pred = final_model.predict(X_val)
                        val_score = r2_score(y_val, val_pred)
                
                trained_models[model_name] = {
                    "model": final_model,
                    "best_params": best_params,
                    "cv_score": best_score,
                    "cv_metrics": cv_metrics,  # Full CV metrics (not just single score)
                    "val_score": val_score,
                    "training_time": training_time
                }
                
                self.log(f"{model_name} training completed in {training_time:.2f}s, CV score: {best_score:.3f}")
                
            except Exception as e:
                self.log(f"Training failed for {model_name}: {str(e)}", "ERROR")
                continue
        
        result = {
            "trained_models": trained_models,
            "task_type": task_type,
            "feature_data": feature_data,
            "n_models_trained": len(trained_models)
        }
        
        return result
    
    async def _train_autogluon(
        self, 
        X_train: np.ndarray, 
        y_train: np.ndarray,
        X_val: Optional[np.ndarray],
        y_val: Optional[np.ndarray],
        feature_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Train AutoGluon TabularPredictor for classification.
        
        AutoGluon handles:
        - Multiple model training (LightGBM, CatBoost, XGBoost, RandomForest, etc.)
        - Hyperparameter optimization
        - Model ensembling
        - All automatically!
        
        Args:
            X_train: Training features (already preprocessed)
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            feature_data: Feature engineering result containing metadata
            
        Returns:
            Training result in same format as _train_models
        """
        try:
            from autogluon.tabular import TabularPredictor
            import tempfile
            import shutil
        except ImportError:
            self.log("AutoGluon not available, falling back to individual models", "ERROR")
            raise ImportError("AutoGluon not installed. Install with: pip install autogluon.tabular")
        
        self.log("=" * 70)
        self.log("Training AutoGluon TabularPredictor")
        self.log("=" * 70)
        
        # Set XGBoost-specific environment variables to prevent segfaults
        # These must be set before XGBoost is imported/used
        os.environ['OMP_NUM_THREADS'] = '1'
        os.environ['OPENBLAS_NUM_THREADS'] = '1'
        os.environ['MKL_NUM_THREADS'] = '1'
        
        start_time = time.time()
        
        # Prepare data for AutoGluon (needs DataFrame with target column)
        train_data = pd.DataFrame(X_train)
        
        # Get feature names if available
        feature_names = feature_data.get("feature_names")
        if feature_names and len(feature_names) == X_train.shape[1]:
            train_data.columns = feature_names
        
        # Add target column
        train_data['__target__'] = y_train
        
        # Prepare validation data if available
        tuning_data = None
        if X_val is not None and y_val is not None:
            tuning_data = pd.DataFrame(X_val)
            if feature_names and len(feature_names) == X_val.shape[1]:
                tuning_data.columns = feature_names
            tuning_data['__target__'] = y_val
            self.log(f"Using validation set for tuning: {len(tuning_data)} samples")
        
        # Determine evaluation metric based on task
        n_classes = len(np.unique(y_train))
        if n_classes == 2:
            eval_metric = 'roc_auc'
            self.log("Binary classification - using ROC-AUC metric")
        else:
            eval_metric = 'accuracy'
            self.log(f"Multiclass classification ({n_classes} classes) - using accuracy metric")
        
        # Create temporary directory for AutoGluon models
        temp_dir = tempfile.mkdtemp(prefix='autogluon_')
        self.log(f"AutoGluon models will be saved to: {temp_dir}")
        
        try:
            # Configure AutoGluon for Compute Canada environment
            time_limit = self.config.ml.max_training_time_minutes * 60
            
            self.log(f"Training with time limit: {time_limit}s ({self.config.ml.max_training_time_minutes} minutes)")
            self.log(f"Using {self.config.ml.n_jobs} CPU cores (thread-safe for shared clusters)")
            self.log("XGBoost configured with n_jobs=1 to prevent segmentation faults on shared clusters")
            
            # Initialize predictor
            predictor = TabularPredictor(
                label='__target__',
                eval_metric=eval_metric,
                path=temp_dir,
                verbosity=2  # Show progress
            )
            
            # Configure XGBoost hyperparameters for Compute Canada
            # This prevents segmentation faults on shared clusters
            xgboost_params = {
                'XGB': {
                    'n_jobs': 1,           # CRITICAL: Use single thread for XGBoost
                    'nthread': 1,          # Alternative parameter name
                },
            }
            
            # Train with appropriate settings for Compute Canada
            predictor.fit(
                train_data=train_data,
                tuning_data=tuning_data,
                time_limit=time_limit,
                presets='medium_quality',  # Balance between speed and quality
                hyperparameters=xgboost_params,  # Force XGBoost to use 1 thread
                num_cpus=self.config.ml.n_jobs,  # Control thread usage for other models
                num_gpus=0,  # No GPU on CPU nodes
                excluded_model_types=['NN_TORCH', 'FASTAI', 'NN_MXNET'],  # Skip deep learning to avoid heavy dependencies
                verbosity=2
            )
            
            training_time = time.time() - start_time
            
            # Get model information
            leaderboard = predictor.leaderboard(silent=True)
            best_model_name = predictor.model_best
            
            self.log("=" * 70)
            self.log(f"AutoGluon training completed in {training_time:.2f}s")
            self.log(f"Best model: {best_model_name}")
            self.log("=" * 70)
            
            # Get CV score from leaderboard
            best_model_info = leaderboard[leaderboard['model'] == best_model_name].iloc[0]
            cv_score = best_model_info['score_val']
            
            # Log top models
            self.log("\nTop 5 models trained by AutoGluon:")
            for idx, row in leaderboard.head(5).iterrows():
                self.log(f"  {row['model']:30s} - Score: {row['score_val']:.4f}")
            
            # Calculate validation score if validation data provided
            val_score = None
            if tuning_data is not None:
                val_predictions = predictor.predict(tuning_data.drop('__target__', axis=1))
                val_score = accuracy_score(y_val, val_predictions)
                self.log(f"\nValidation accuracy: {val_score:.4f}")
            
            # Return in same format as regular training
            result = {
                "trained_models": {
                    "autogluon": {
                        "model": predictor,
                        "best_params": {"ensemble": best_model_name},  # Store which ensemble was best
                        "cv_score": cv_score,
                        "cv_metrics": {
                            eval_metric: cv_score,
                            "n_models_trained": len(leaderboard)
                        },
                        "val_score": val_score,
                        "training_time": training_time,
                        "autogluon_leaderboard": leaderboard.to_dict('records'),
                        "model_path": temp_dir  # Store path for later cleanup
                    }
                },
                "task_type": "classification",
                "feature_data": feature_data,
                "n_models_trained": 1  # One AutoGluon ensemble (contains multiple models internally)
            }
            
            self.log(f"✅ AutoGluon training successful - {len(leaderboard)} models trained and ensembled")
            return result
            
        except Exception as e:
            # Clean up temp directory on failure
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            self.log(f"AutoGluon training failed: {str(e)}", "ERROR")
            raise
    
    async def _optimize_hyperparameters(
        self, 
        model_class, 
        param_space: Dict[str, Any], 
        X_train: np.ndarray, 
        y_train: np.ndarray,
        task_type: str,
        cv_groups: Optional[np.ndarray] = None
    ) -> Tuple[Dict[str, Any], float, Dict[str, Any]]:
        """Optimize hyperparameters using Optuna"""
        
        if not param_space:
            # No hyperparameters to optimize - inject thread params in constructor
            model_name = model_class.__name__.lower()
            init_params = {}
            if 'catboost' in model_name:
                init_params['thread_count'] = self.config.ml.n_jobs
                init_params['verbose'] = 0
            elif 'xgb' in model_name:
                init_params['nthread'] = self.config.ml.n_jobs
                init_params['n_jobs'] = self.config.ml.n_jobs
            elif 'lgbm' in model_name or 'lightgbm' in model_name:
                init_params['n_jobs'] = self.config.ml.n_jobs
            elif 'randomforest' in model_name or 'forest' in model_name:
                init_params['n_jobs'] = self.config.ml.n_jobs
            
            model = model_class(**init_params)
            score, cv_metrics = self._evaluate_model_cv(model, X_train, y_train, task_type, cv_groups)
            return {}, score, cv_metrics
        
        def objective(trial):
            # Suggest hyperparameters
            params = {}
            for param_name, param_config in param_space.items():
                if param_config["type"] == "int":
                    if "range" in param_config:
                        params[param_name] = trial.suggest_int(
                            param_name, param_config["range"][0], param_config["range"][1]
                        )
                    else:
                        params[param_name] = param_config["value"]
                elif param_config["type"] == "float":
                    if "range" in param_config:
                        log_scale = param_config.get("log", False)
                        params[param_name] = trial.suggest_float(
                            param_name, param_config["range"][0], param_config["range"][1], log=log_scale
                        )
                    else:
                        params[param_name] = param_config["value"]
                elif param_config["type"] == "categorical":
                    params[param_name] = trial.suggest_categorical(
                        param_name, param_config["values"]
                    )
            
            # Inject thread limits BEFORE model creation (critical for CatBoost)
            # CatBoost creates thread pool in __init__, so we must pass thread_count in constructor
            model_name = model_class.__name__.lower()
            if 'catboost' in model_name:
                params['thread_count'] = self.config.ml.n_jobs
            elif 'xgb' in model_name:
                params['nthread'] = self.config.ml.n_jobs
                params['n_jobs'] = self.config.ml.n_jobs
            elif 'lgbm' in model_name or 'lightgbm' in model_name:
                params['n_jobs'] = self.config.ml.n_jobs
            elif 'randomforest' in model_name or 'forest' in model_name:
                params['n_jobs'] = self.config.ml.n_jobs
            
            # Create and evaluate model
            try:
                model = model_class(**params)
                
                # Set random state if available
                if hasattr(model, 'random_state'):
                    model.set_params(random_state=self.config.data.random_state)
                
                # Handle verbose parameter
                if 'verbose' in model.get_params():
                    model.set_params(verbose=0)
                
                score, _ = self._evaluate_model_cv(model, X_train, y_train, task_type, cv_groups)
                return score
                
            except Exception as e:
                # Return poor score for failed trials
                return -np.inf if task_type == "regression" else 0
        
        # Run optimization
        study = optuna.create_study(
            direction="maximize" if task_type != "regression" else "maximize",
            sampler=optuna.samplers.TPESampler(seed=self.config.data.random_state)
        )
        
        # Limit optimization time
        max_trials = min(self.config.ml.optuna_trials, 50)  # Reasonable limit
        
        study.optimize(objective, n_trials=max_trials, timeout=300)  # 5 minutes max
        
        # Inject thread limits into best params BEFORE model creation
        best_params = study.best_params.copy()
        model_name = model_class.__name__.lower()
        if 'catboost' in model_name:
            best_params['thread_count'] = self.config.ml.n_jobs
        elif 'xgb' in model_name:
            best_params['nthread'] = self.config.ml.n_jobs
            best_params['n_jobs'] = self.config.ml.n_jobs
        elif 'lgbm' in model_name or 'lightgbm' in model_name:
            best_params['n_jobs'] = self.config.ml.n_jobs
        elif 'randomforest' in model_name or 'forest' in model_name:
            best_params['n_jobs'] = self.config.ml.n_jobs
        
        # Get full CV metrics for best model
        best_model = model_class(**best_params)
        if hasattr(best_model, 'random_state'):
            best_model.set_params(random_state=self.config.data.random_state)
        if 'verbose' in best_model.get_params():
            best_model.set_params(verbose=0)
        
        _, cv_metrics = self._evaluate_model_cv(best_model, X_train, y_train, task_type, cv_groups)
        
        return study.best_params, study.best_value, cv_metrics
    
    def _evaluate_model_cv(self, model, X: np.ndarray, y: np.ndarray, task_type: str, 
                          cv_groups: Optional[np.ndarray] = None) -> Tuple[float, Dict[str, Any]]:
        """Evaluate model using cross-validation
        
        Returns:
            Tuple of (primary_score, cv_metrics_dict)
        """
        
        try:
            # Handle survival tasks separately (they have structured arrays)
            if task_type == "survival":
                return self._evaluate_survival_cv(model, X, y, cv_groups)
            
            # Determine CV strategy for classification/regression
            if cv_groups is not None:
                # Use preset CV groups from dataset
                cv = PredefinedSplit(cv_groups)
                self.log(f"Using preset CV split with {len(np.unique(cv_groups))} folds")
            elif task_type == "classification":
                # Use stratified CV for classification
                cv = StratifiedKFold(n_splits=self.config.ml.cv_folds, shuffle=True, 
                                   random_state=self.config.data.random_state)
            else:
                # Use regular CV for regression
                cv = KFold(n_splits=self.config.ml.cv_folds, shuffle=True, 
                          random_state=self.config.data.random_state)
            
            # Define scoring metrics based on task type
            if task_type == "classification":
                scoring = {
                    'accuracy': 'accuracy',
                    'precision': 'precision_weighted',
                    'recall': 'recall_weighted',
                    'f1': 'f1_weighted',
                    'roc_auc': 'roc_auc' if len(np.unique(y)) == 2 else 'roc_auc_ovr_weighted'
                }
                primary_metric = "roc_auc" if len(np.unique(y)) == 2 else "accuracy"
            else:  # regression
                scoring = {
                    'mae': 'neg_mean_absolute_error',
                    'mse': 'neg_mean_squared_error',
                    'r2': 'r2'
                }
                primary_metric = "r2"
            
            # Perform cross-validation with multiple metrics
            # Use n_jobs=1 to avoid thread creation issues on shared clusters
            cv_results = cross_validate(model, X, y, cv=cv, scoring=scoring, 
                                       n_jobs=1, return_train_score=False)
            
            # Compute mean metrics
            cv_metrics = {}
            for metric_name in scoring.keys():
                scores = cv_results[f'test_{metric_name}']
                # Handle negative metrics (mae, mse)
                if metric_name in ['mae', 'mse']:
                    scores = -scores  # Convert back to positive
                cv_metrics[metric_name] = float(np.mean(scores))
                cv_metrics[f'{metric_name}_std'] = float(np.std(scores))
            
            primary_score = cv_metrics[primary_metric]
            
            return primary_score, cv_metrics
            
        except Exception as e:
            self.log(f"CV evaluation failed: {str(e)}", "WARNING")
            return 0.0, {}
    
    def _evaluate_survival_cv(self, model, X: np.ndarray, y: np.ndarray, 
                              cv_groups: Optional[np.ndarray] = None) -> Tuple[float, Dict[str, Any]]:
        """
        Evaluate survival model using cross-validation with C-index and time-dependent AUC
        
        Args:
            model: Survival model
            X: Features
            y: Survival data (structured array)
            cv_groups: Optional preset CV groups
            
        Returns:
            Tuple of (mean_c_index, cv_metrics_dict)
        """
        try:
            from sksurv.metrics import concordance_index_censored, cumulative_dynamic_auc
            from sklearn.model_selection import KFold
            from sklearn.base import clone
        except ImportError:
            self.log("scikit-survival not available for CV", "WARNING")
            return 0.5, {}
        
        try:
            # Determine CV strategy
            if cv_groups is not None:
                cv = PredefinedSplit(cv_groups)
                self.log(f"Using preset CV split with {len(np.unique(cv_groups))} folds for survival")
            else:
                # Use KFold (can't stratify on structured array directly)
                cv = KFold(n_splits=self.config.ml.cv_folds, shuffle=True, 
                          random_state=self.config.data.random_state)
            
            c_indices = []
            
            for train_idx, test_idx in cv.split(X):
                X_train_fold, X_test_fold = X[train_idx], X[test_idx]
                y_train_fold, y_test_fold = y[train_idx], y[test_idx]
                
                # Clone and train model
                model_fold = clone(model)
                model_fold.fit(X_train_fold, y_train_fold)
                
                # Predict and calculate C-index only (primary metric for CV)
                risk_scores = model_fold.predict(X_test_fold)
                c_index = concordance_index_censored(
                    y_test_fold['event'],
                    y_test_fold['time'],
                    risk_scores
                )[0]
                c_indices.append(c_index)
            
            # Calculate mean and std for C-index
            mean_c_index = np.mean(c_indices)
            std_c_index = np.std(c_indices)
            
            cv_metrics = {
                'concordance_index': float(mean_c_index),
                'concordance_index_std': float(std_c_index)
            }
            
            # Log summary
            log_msg = f"Survival CV: C-index = {mean_c_index:.3f} (±{std_c_index:.3f})"
            
            self.log(log_msg)
            
            return mean_c_index, cv_metrics
            
        except Exception as e:
            self.log(f"Survival CV evaluation failed: {str(e)}", "WARNING")
            return 0.5, {}
    
    def _calculate_classification_metrics(self, y_true, y_pred, y_pred_proba=None) -> Dict[str, float]:
        """Calculate classification metrics"""
        n_classes = len(np.unique(y_true))
        
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, average='weighted', zero_division=0),
            "recall": recall_score(y_true, y_pred, average='weighted', zero_division=0),
            "f1": f1_score(y_true, y_pred, average='weighted', zero_division=0)
        }
        
        # Add ROC AUC
        if y_pred_proba is not None:
            try:
                if n_classes == 2:
                    # Binary classification - use probability of positive class
                    metrics["roc_auc"] = roc_auc_score(y_true, y_pred_proba[:, 1])
                else:
                    # Multiclass classification - use one-vs-rest with weighted average
                    metrics["roc_auc"] = roc_auc_score(y_true, y_pred_proba, 
                                                      multi_class='ovr', average='weighted')
            except (ValueError, IndexError) as e:
                # Fallback if ROC AUC calculation fails
                self.log(f"ROC AUC calculation failed: {str(e)}", "WARNING")
                metrics["roc_auc"] = 0.0
        else:
            metrics["roc_auc"] = 0.0
        
        return metrics
    
    def _calculate_regression_metrics(self, y_true, y_pred) -> Dict[str, float]:
        """Calculate regression metrics"""
        return {
            "mae": mean_absolute_error(y_true, y_pred),
            "mse": mean_squared_error(y_true, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
            "r2": r2_score(y_true, y_pred)
        }
    
    def _calculate_survival_metrics(self, model, X_test, y_test, y_train=None) -> Dict[str, float]:
        """
        Calculate survival analysis metrics using scikit-survival
        
        Args:
            model: Trained survival model
            X_test: Test features
            y_test: Test survival data (structured array)
            y_train: Training survival data (for IBS calculation)
            
        Returns:
            Dictionary with concordance_index and integrated_brier_score
        """
        try:
            from sksurv.metrics import (
                concordance_index_censored,
                integrated_brier_score,
                cumulative_dynamic_auc
            )
        except ImportError:
            self.log("scikit-survival not available, returning dummy metrics", "WARNING")
            return {
                "concordance_index": 0.5,
                "integrated_brier_score": None
            }
        
        metrics = {}
        
        try:
            # Concordance Index (C-index)
            # Primary metric for survival analysis
            # Measures how well the model ranks survival times
            # 1.0 = perfect, 0.5 = random, 0.0 = perfectly wrong
            
            if hasattr(model, 'predict'):
                risk_scores = model.predict(X_test)
                
                # C-index requires event indicator, time, and risk scores
                c_index_result = concordance_index_censored(
                    y_test['event'],
                    y_test['time'],
                    risk_scores
                )
                
                metrics["concordance_index"] = float(c_index_result[0])
                self.log(f"C-index: {metrics['concordance_index']:.3f}")
            else:
                metrics["concordance_index"] = None
                self.log("Model does not support predict(), C-index not calculated", "WARNING")
            
            # Integrated Brier Score (IBS)
            # Measures calibration of survival probability predictions
            # Lower is better (0 = perfect)
            # Only works if model supports predict_survival_function
            
            if hasattr(model, 'predict_survival_function') and y_train is not None:
                try:
                    # Get survival functions
                    surv_funcs = model.predict_survival_function(X_test)
                    
                    # Select time points for evaluation (10 points between 5th and 95th percentile)
                    all_times = np.concatenate([y_train['time'], y_test['time']])
                    times = np.percentile(all_times[all_times > 0], np.linspace(5, 95, 10))
                    
                    # Convert StepFunction objects to 2D array
                    # Each row is a sample, each column is a time point
                    surv_matrix = np.row_stack([
                        fn(times) for fn in surv_funcs
                    ])
                    
                    # Calculate IBS - returns a scalar
                    ibs = integrated_brier_score(y_train, y_test, surv_matrix, times)
                    
                    # IBS is already a scalar, just convert to float
                    if isinstance(ibs, (int, float, np.number)):
                        metrics["integrated_brier_score"] = float(ibs)
                        self.log(f"Integrated Brier Score: {metrics['integrated_brier_score']:.3f}")
                    else:
                        # If it's not a scalar, take mean or handle accordingly
                        self.log(f"IBS returned unexpected type: {type(ibs)}, value: {ibs}", "WARNING")
                        metrics["integrated_brier_score"] = None
                    
                except Exception as e:
                    import traceback
                    self.log(f"IBS calculation failed: {str(e)}", "WARNING")
                    self.log(f"Full traceback: {traceback.format_exc()}", "DEBUG")
                    metrics["integrated_brier_score"] = None
            else:
                if not hasattr(model, 'predict_survival_function'):
                    self.log(f"Model {type(model).__name__} does not have predict_survival_function method", "DEBUG")
                if y_train is None:
                    self.log("y_train is None, cannot calculate IBS", "WARNING")
                metrics["integrated_brier_score"] = None
            
            # Additional metrics if available
            if metrics["concordance_index"] is not None:
                # Calculate time-dependent AUC at fixed time points (6, 12, 24 months)
                try:
                    # Get all times to determine time units (days vs months)
                    all_times = np.concatenate([y_train['time'], y_test['time']])
                    median_all_time = np.median(all_times[all_times > 0])
                    
                    # Heuristic: if median time > 365, likely in days; otherwise in months
                    # Also check if max time > 365 as additional evidence
                    max_time = np.max(all_times)
                    if median_all_time > 365 or max_time > 1000:
                        # Time is in days, convert to months
                        time_unit = "days"
                        conversion_factor = 30.44  # Average days per month
                        self.log(f"Detected time unit: days (median={median_all_time:.1f}). Converting to months.")
                    else:
                        # Time is already in months
                        time_unit = "months"
                        conversion_factor = 1.0
                        self.log(f"Detected time unit: months (median={median_all_time:.1f})")
                    
                    # Define evaluation time points in months
                    eval_months = [6, 12, 24]
                    
                    # Convert to the dataset's time unit
                    eval_times = [m * conversion_factor for m in eval_months]
                    
                    # Filter to only use times within a reasonable range
                    # Use 95th percentile instead of max to be more robust to outliers
                    # and allow evaluation at timepoints where we have sufficient data
                    time_95th = np.percentile(all_times, 95)
                    max_observed_time = np.max(all_times)
                    
                    # Use 95th percentile, but extended by 10% to allow some extrapolation
                    valid_eval_times = [t for t in eval_times if t <= time_95th * 1.1]
                    valid_eval_months = [eval_months[i] for i, t in enumerate(eval_times) if t <= time_95th * 1.1]
                    
                    self.log(f"Max observed time: {max_observed_time:.1f} {time_unit}, 95th percentile: {time_95th:.1f}")
                    self.log(f"Valid evaluation times: {[(eval_months[i], eval_times[i]) for i in range(len(eval_times)) if eval_times[i] <= time_95th * 1.1]}")
                    
                    if len(valid_eval_times) > 0 and hasattr(model, 'predict'):
                        risk_estimates = model.predict(X_test)
                        
                        # Calculate AUC at each time point individually
                        # This is more robust than calculating all at once
                        metrics["time_dependent_auc"] = {}
                        
                        for i, (month, time) in enumerate(zip(valid_eval_months, valid_eval_times)):
                            try:
                                # Calculate AUC for this single timepoint
                                _, auc_val = cumulative_dynamic_auc(
                                    y_train, y_test, risk_estimates, [time]
                                )
                                
                                # Handle both scalar and array returns
                                if isinstance(auc_val, (list, np.ndarray)):
                                    auc_val = float(auc_val[0])
                                else:
                                    auc_val = float(auc_val)
                                
                                metrics["time_dependent_auc"][f"{month}mo"] = auc_val
                                self.log(f"Time-dependent AUC at {month} months: {auc_val:.3f}")
                                
                            except Exception as e:
                                self.log(f"Could not calculate AUC at {month} months: {str(e)}", "WARNING")
                                # Continue to try other timepoints
                        
                        # If no valid times, set to None
                        if not metrics["time_dependent_auc"]:
                            metrics["time_dependent_auc"] = None
                    else:
                        if len(valid_eval_times) == 0:
                            self.log(f"No evaluation times within observed range (max={max_observed_time:.1f} {time_unit})", "WARNING")
                        metrics["time_dependent_auc"] = None
                        
                except Exception as e:
                    import traceback
                    self.log(f"Time-dependent AUC calculation failed: {str(e)}", "WARNING")
                    self.log(f"Full traceback: {traceback.format_exc()}", "DEBUG")
                    metrics["time_dependent_auc"] = None
            
            return metrics
            
        except Exception as e:
            self.log(f"Survival metrics calculation failed: {str(e)}", "ERROR")
            return {
                "concordance_index": None,
                "integrated_brier_score": None
            }
    
    def _get_primary_metric(self, task_type: str, y_data: np.ndarray = None) -> str:
        """Get the primary metric for model comparison"""
        if task_type == "classification":
            # Use ROC AUC if available (works for both binary and multiclass)
            # Fall back to accuracy for multiclass if ROC AUC isn't reliable
            if y_data is not None:
                # Check if it's a structured array (survival data)
                if hasattr(y_data, 'dtype') and y_data.dtype.names:
                    return "concordance_index"  # It's survival data
                
                n_classes = len(np.unique(y_data))
                if n_classes == 2:
                    return "roc_auc"
                else:
                    # For multiclass, use accuracy as primary metric
                    # (ROC AUC multiclass can be less interpretable)
                    return "accuracy"
            return "roc_auc"  # Default for binary
        elif task_type == "regression":
            return "r2"
        else:  # survival
            return "concordance_index"
    
    def save_models(self, output_dir: str, trained_models: Dict[str, Any]) -> Dict[str, str]:
        """Save trained models to disk"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        saved_paths = {}
        
        for model_name, model_info in trained_models.items():
            try:
                model_file = output_path / f"{model_name}_model.pkl"
                
                with open(model_file, 'wb') as f:
                    pickle.dump({
                        'model': model_info['model'],
                        'best_params': model_info.get('best_params', {}),
                        'cv_score': model_info.get('cv_score', 0),
                        'training_time': model_info.get('training_time', 0)
                    }, f)
                
                saved_paths[model_name] = str(model_file)
                self.log(f"Model {model_name} saved to {model_file}")
                
            except Exception as e:
                self.log(f"Failed to save {model_name}: {str(e)}", "ERROR")
        
        return saved_paths
    
    def load_model(self, model_path: str) -> Dict[str, Any]:
        """Load a saved model"""
        with open(model_path, 'rb') as f:
            return pickle.load(f)
