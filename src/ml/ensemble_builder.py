"""
Ensemble Building for ML Models

Creates ensemble models for both classification and survival analysis tasks.
Supports multiple ensemble strategies inspired by Kaggle grandmasters.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Union
import warnings
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, roc_auc_score
from scipy import stats

try:
    from sksurv.metrics import concordance_index_censored
    from sksurv.linear_model import CoxPHSurvivalAnalysis
    SKSURV_AVAILABLE = True
except ImportError:
    SKSURV_AVAILABLE = False

from ..core.base_agent import LLMAgent
from ..core.state import TaskType, AgentResult
from ..core.config import Config


class VotingEnsemble(BaseEstimator, ClassifierMixin):
    """
    Voting ensemble for classification.
    Averages predicted probabilities (soft voting).
    """
    
    def __init__(self, models: List[Tuple[str, Any]], weights: Optional[List[float]] = None):
        self.models = models  # List of (name, model) tuples
        self.weights = weights if weights is not None else [1.0] * len(models)
        self.classes_ = None
        
    def fit(self, X, y):
        """Already fitted models - just store classes"""
        self.classes_ = np.unique(y)
        return self
    
    def predict_proba(self, X):
        """Average predicted probabilities"""
        probas = []
        total_weight = 0
        
        for (name, model), weight in zip(self.models, self.weights):
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X)
                probas.append(proba * weight)
                total_weight += weight
            else:
                warnings.warn(f"Model {name} doesn't support predict_proba, skipping")
        
        if not probas:
            raise ValueError("No models support predict_proba")
        
        avg_proba = np.sum(probas, axis=0) / total_weight
        return avg_proba
    
    def predict(self, X):
        """Predict class labels"""
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]


class StackingEnsemble(BaseEstimator, ClassifierMixin):
    """
    Stacking ensemble for classification.
    Trains a meta-model on base model predictions.
    """
    
    def __init__(self, base_models: List[Tuple[str, Any]], meta_model: Any, cv_folds: int = 5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.cv_folds = cv_folds
        self.classes_ = None
        
    def fit(self, X, y):
        """Generate meta-features and train meta-model"""
        self.classes_ = np.unique(y)
        
        # Generate out-of-fold predictions as meta-features
        meta_features = self._generate_meta_features(X, y)
        
        # Train meta-model
        self.meta_model.fit(meta_features, y)
        return self
    
    def _generate_meta_features(self, X, y):
        """Generate meta-features using cross-validation"""
        n_samples = len(X)
        n_models = len(self.base_models)
        n_classes = len(self.classes_)
        
        # Initialize meta-features array
        meta_features = np.zeros((n_samples, n_models * n_classes))
        
        # Use stratified K-fold
        kf = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
        
        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X, y)):
            X_train_fold = X.iloc[train_idx] if hasattr(X, 'iloc') else X[train_idx]
            X_val_fold = X.iloc[val_idx] if hasattr(X, 'iloc') else X[val_idx]
            y_train_fold = y.iloc[train_idx] if hasattr(y, 'iloc') else y[train_idx]
            
            # Generate predictions from each base model
            for model_idx, (name, model) in enumerate(self.base_models):
                try:
                    # Clone and refit model on fold
                    from sklearn.base import clone
                    fold_model = clone(model)
                    fold_model.fit(X_train_fold, y_train_fold)
                    
                    # Get predictions for validation fold
                    if hasattr(fold_model, 'predict_proba'):
                        proba = fold_model.predict_proba(X_val_fold)
                    else:
                        # Use one-hot encoded predictions if predict_proba not available
                        pred = fold_model.predict(X_val_fold)
                        proba = np.eye(n_classes)[pred]
                    
                    # Store in meta-features
                    start_col = model_idx * n_classes
                    end_col = start_col + n_classes
                    meta_features[val_idx, start_col:end_col] = proba
                    
                except Exception as e:
                    warnings.warn(f"Error generating meta-features for {name}: {str(e)}")
        
        return meta_features
    
    def predict_proba(self, X):
        """Generate meta-features and predict with meta-model"""
        meta_features = self._get_meta_features_for_prediction(X)
        
        if hasattr(self.meta_model, 'predict_proba'):
            return self.meta_model.predict_proba(meta_features)
        else:
            # Fallback to one-hot encoding
            pred = self.meta_model.predict(meta_features)
            n_classes = len(self.classes_)
            return np.eye(n_classes)[pred]
    
    def _get_meta_features_for_prediction(self, X):
        """Generate meta-features for prediction (using all base models)"""
        n_samples = len(X)
        n_models = len(self.base_models)
        n_classes = len(self.classes_)
        meta_features = np.zeros((n_samples, n_models * n_classes))
        
        for model_idx, (name, model) in enumerate(self.base_models):
            try:
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(X)
                else:
                    pred = model.predict(X)
                    proba = np.eye(n_classes)[pred]
                
                start_col = model_idx * n_classes
                end_col = start_col + n_classes
                meta_features[:, start_col:end_col] = proba
            except Exception as e:
                warnings.warn(f"Error getting predictions from {name}: {str(e)}")
        
        return meta_features
    
    def predict(self, X):
        """Predict class labels"""
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]


class SurvivalEnsemble(BaseEstimator):
    """
    Ensemble for survival analysis.
    Combines risk scores from multiple survival models.
    """
    
    def __init__(self, models: List[Tuple[str, Any]], 
                 weights: Optional[List[float]] = None,
                 method: str = 'mean'):
        """
        Args:
            models: List of (name, model) tuples
            weights: Optional weights for each model
            method: 'mean', 'median', or 'weighted'
        """
        self.models = models
        self.weights = weights if weights is not None else [1.0] * len(models)
        self.method = method
        
    def fit(self, X, y):
        """Already fitted models - nothing to do"""
        return self
    
    def predict(self, X):
        """
        Predict risk scores by combining base model predictions.
        
        Returns:
            Combined risk scores (higher = higher risk)
        """
        risk_scores = []
        valid_weights = []
        
        for (name, model), weight in zip(self.models, self.weights):
            try:
                if hasattr(model, 'predict'):
                    scores = model.predict(X)
                    risk_scores.append(scores)
                    valid_weights.append(weight)
                else:
                    warnings.warn(f"Model {name} doesn't support predict, skipping")
            except Exception as e:
                warnings.warn(f"Error getting predictions from {name}: {str(e)}")
        
        if not risk_scores:
            raise ValueError("No models could generate predictions")
        
        risk_scores = np.array(risk_scores)
        
        # Combine based on method
        if self.method == 'mean' or self.method == 'weighted':
            # Normalize weights
            valid_weights = np.array(valid_weights)
            valid_weights = valid_weights / valid_weights.sum()
            
            # Weighted average
            combined = np.average(risk_scores, axis=0, weights=valid_weights)
        elif self.method == 'median':
            combined = np.median(risk_scores, axis=0)
        elif self.method == 'rank':
            # Average of ranks (robust to different scales)
            ranked = np.array([stats.rankdata(scores) for scores in risk_scores])
            combined = np.mean(ranked, axis=0)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        return combined
    
    def score(self, X, y):
        """Calculate concordance index"""
        if not SKSURV_AVAILABLE:
            raise ImportError("scikit-survival required for survival ensembles")
        
        risk_scores = self.predict(X)
        c_index = concordance_index_censored(
            y['event'].astype(bool),
            y['time'],
            risk_scores
        )[0]
        return c_index


class SurvivalStackingEnsemble(BaseEstimator):
    """
    Stacking ensemble for survival analysis.
    Trains a Cox model on base model risk scores as features.
    """
    
    def __init__(self, base_models: List[Tuple[str, Any]], cv_folds: int = 5):
        self.base_models = base_models
        self.cv_folds = cv_folds
        self.meta_model = None
        
    def fit(self, X, y):
        """Generate meta-features and train Cox meta-model"""
        if not SKSURV_AVAILABLE:
            raise ImportError("scikit-survival required for survival stacking")
        
        # Generate out-of-fold risk scores as meta-features
        meta_features = self._generate_meta_features(X, y)
        
        # Train Cox model on meta-features
        self.meta_model = CoxPHSurvivalAnalysis(alpha=0.1)
        self.meta_model.fit(meta_features, y)
        return self
    
    def _generate_meta_features(self, X, y):
        """Generate meta-features using cross-validation"""
        n_samples = len(X)
        n_models = len(self.base_models)
        meta_features = np.zeros((n_samples, n_models))
        
        # Use K-fold (no stratification for survival)
        kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
        
        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
            X_train_fold = X.iloc[train_idx] if hasattr(X, 'iloc') else X[train_idx]
            X_val_fold = X.iloc[val_idx] if hasattr(X, 'iloc') else X[val_idx]
            y_train_fold = y[train_idx]
            
            # Generate risk scores from each base model
            for model_idx, (name, model) in enumerate(self.base_models):
                try:
                    # Clone and refit model on fold
                    from sklearn.base import clone
                    fold_model = clone(model)
                    fold_model.fit(X_train_fold, y_train_fold)
                    
                    # Get risk scores for validation fold
                    risk_scores = fold_model.predict(X_val_fold)
                    meta_features[val_idx, model_idx] = risk_scores
                    
                except Exception as e:
                    warnings.warn(f"Error generating meta-features for {name}: {str(e)}")
        
        return meta_features
    
    def predict(self, X):
        """Generate meta-features and predict with Cox meta-model"""
        meta_features = self._get_meta_features_for_prediction(X)
        return self.meta_model.predict(meta_features)
    
    def _get_meta_features_for_prediction(self, X):
        """Generate meta-features for prediction"""
        n_samples = len(X)
        n_models = len(self.base_models)
        meta_features = np.zeros((n_samples, n_models))
        
        for model_idx, (name, model) in enumerate(self.base_models):
            try:
                risk_scores = model.predict(X)
                meta_features[:, model_idx] = risk_scores
            except Exception as e:
                warnings.warn(f"Error getting predictions from {name}: {str(e)}")
        
        return meta_features
    
    def score(self, X, y):
        """Calculate concordance index"""
        risk_scores = self.predict(X)
        c_index = concordance_index_censored(
            y['event'].astype(bool),
            y['time'],
            risk_scores
        )[0]
        return c_index


class EnsembleBuilder(LLMAgent):
    """
    Agent for building ensemble models from trained base models.
    
    Supports both classification and survival analysis tasks.
    Implements various ensemble strategies.
    """
    
    def __init__(self, config: Config):
        super().__init__("ensemble_builder", config)
    
    def get_task_type(self) -> TaskType:
        """Return task type for this agent"""
        return TaskType.MODEL_TRAINING
    
    async def execute(self, inputs: Dict[str, Any]) -> AgentResult:
        """
        Execute ensemble creation (implements abstract method from LLMAgent).
        
        This is an alternative entry point to create_ensemble() that follows
        the standard agent interface.
        """
        self.validate_inputs(inputs, ["trained_models", "feature_data", "task_type", "ensemble_type"])
        
        result = self.create_ensemble(
            trained_models=inputs["trained_models"],
            feature_data=inputs["feature_data"],
            task_type=inputs["task_type"],
            ensemble_type=inputs["ensemble_type"],
            models_to_ensemble=inputs.get("models_to_ensemble"),
            meta_model_name=inputs.get("meta_model_name")
        )
        
        if result.get("success"):
            return self.create_result(
                inputs=inputs,
                outputs=result,
                confidence_score=0.9
            )
        else:
            return self.create_result(
                inputs=inputs,
                outputs={},
                confidence_score=0.0,
                error=result.get("error", "Ensemble creation failed")
            )
        
    def create_ensemble(
        self,
        trained_models: Dict[str, Dict[str, Any]],
        feature_data: Dict[str, Any],
        task_type: str,
        ensemble_type: str = "weighted",
        models_to_ensemble: Optional[List[str]] = None,
        meta_model_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create an ensemble from trained models.
        
        Args:
            trained_models: Dictionary of trained models
            feature_data: Feature engineering results with data splits
            task_type: "classification" or "survival"
            ensemble_type: Type of ensemble ("voting", "weighted", "stacking", "blending")
            models_to_ensemble: Optional list of model names (defaults to all)
            meta_model_name: Meta-model for stacking (optional)
            
        Returns:
            Dictionary with ensemble model and performance metrics
        """
        self.log(f"Creating {ensemble_type} ensemble for {task_type} task")
        
        # Select models to ensemble
        if models_to_ensemble is None:
            models_to_ensemble = list(trained_models.keys())
        
        self.log(f"Models available in state: {list(trained_models.keys())}")
        self.log(f"Models requested for ensemble: {models_to_ensemble}")
        
        # Validate we have enough models
        if len(models_to_ensemble) < 2:
            return {
                "error": f"Need at least 2 models for ensemble, only {len(models_to_ensemble)} available",
                "models_available": list(trained_models.keys())
            }
        
        # Get base models
        base_models = []
        model_scores = []
        
        for model_name in models_to_ensemble:
            if model_name not in trained_models:
                self.log(f"Model {model_name} not found, skipping", "WARNING")
                continue
                
            model_info = trained_models[model_name]
            model = model_info.get("model")
            cv_score = model_info.get("cv_score", 0.5)
            
            if model is None:
                self.log(f"Model {model_name} has no model object, skipping", "WARNING")
                continue
            
            base_models.append((model_name, model))
            model_scores.append(cv_score)
        
        if len(base_models) < 2:
            return {
                "error": f"Only {len(base_models)} valid models found for ensemble",
                "models_attempted": models_to_ensemble
            }
        
        self.log(f"Building ensemble from {len(base_models)} models: {[name for name, _ in base_models]}")
        
        # Create ensemble based on task type and ensemble type
        try:
            if task_type == "classification":
                ensemble = self._create_classification_ensemble(
                    base_models, model_scores, ensemble_type, meta_model_name
                )
            elif task_type == "survival":
                ensemble = self._create_survival_ensemble(
                    base_models, model_scores, ensemble_type
                )
            else:
                return {"error": f"Task type {task_type} not supported for ensembles"}
            
            # Evaluate ensemble
            data_splits = feature_data.get("data_splits", {})
            X_train = data_splits.get("X_train")
            y_train = data_splits.get("y_train")
            
            if X_train is None or y_train is None:
                return {"error": "Training data not available for ensemble evaluation"}
            
            # Fit ensemble (just stores info, models already trained)
            ensemble.fit(X_train, y_train)
            
            # Calculate weights for CV evaluation
            weights = None
            if ensemble_type == "weighted":
                scores_array = np.array(model_scores)
                weights = scores_array / scores_array.sum()
            
            # Cross-validation score
            cv_score = self._evaluate_ensemble_cv(
                ensemble, X_train, y_train, task_type,
                base_model_scores=model_scores,
                weights=weights
            )
            
            # Test set evaluation (if available)
            X_test = data_splits.get("X_test")
            y_test = data_splits.get("y_test")
            test_score = None
            
            if X_test is not None and y_test is not None:
                test_score = self._evaluate_ensemble_test(ensemble, X_test, y_test, task_type)
            
            self.log(f"Ensemble CV score: {cv_score:.4f}")
            if test_score is not None:
                self.log(f"Ensemble test score: {test_score:.4f}")
            
            return {
                "success": True,
                "ensemble": ensemble,
                "ensemble_type": ensemble_type,
                "task_type": task_type,
                "cv_score": float(cv_score),
                "test_score": float(test_score) if test_score is not None else None,
                "models_used": [name for name, _ in base_models],
                "n_models": len(base_models),
                "model_weights": model_scores if ensemble_type == "weighted" else None
            }
            
        except Exception as e:
            self.log(f"Error creating ensemble: {str(e)}", "ERROR")
            import traceback
            traceback.print_exc()
            return {"error": str(e)}
    
    def _create_classification_ensemble(
        self,
        base_models: List[Tuple[str, Any]],
        model_scores: List[float],
        ensemble_type: str,
        meta_model_name: Optional[str]
    ) -> BaseEstimator:
        """Create classification ensemble"""
        
        if ensemble_type == "voting":
            # Simple voting (equal weights)
            return VotingEnsemble(base_models)
        
        elif ensemble_type == "weighted":
            # Weighted by CV scores
            # Normalize scores to [0, 1] and use as weights
            scores_array = np.array(model_scores)
            weights = scores_array / scores_array.sum()
            return VotingEnsemble(base_models, weights=weights.tolist())
        
        elif ensemble_type == "stacking" or ensemble_type == "blending":
            # Stacking with meta-model
            if meta_model_name is None or meta_model_name == "logistic_regression":
                meta_model = LogisticRegression(max_iter=1000, random_state=42)
            elif meta_model_name == "ridge":
                meta_model = Ridge(random_state=42)
            else:
                # Default to logistic regression
                self.log(f"Unknown meta-model {meta_model_name}, using logistic regression", "WARNING")
                meta_model = LogisticRegression(max_iter=1000, random_state=42)
            
            cv_folds = 5 if ensemble_type == "stacking" else 3
            return StackingEnsemble(base_models, meta_model, cv_folds=cv_folds)
        
        else:
            raise ValueError(f"Unknown ensemble type for classification: {ensemble_type}")
    
    def _create_survival_ensemble(
        self,
        base_models: List[Tuple[str, Any]],
        model_scores: List[float],
        ensemble_type: str
    ) -> BaseEstimator:
        """Create survival ensemble"""
        
        if not SKSURV_AVAILABLE:
            raise ImportError("scikit-survival required for survival ensembles")
        
        if ensemble_type == "voting" or ensemble_type == "averaging":
            # Simple averaging of risk scores
            return SurvivalEnsemble(base_models, method='mean')
        
        elif ensemble_type == "weighted":
            # Weighted by C-index scores
            scores_array = np.array(model_scores)
            weights = scores_array / scores_array.sum()
            return SurvivalEnsemble(base_models, weights=weights.tolist(), method='weighted')
        
        elif ensemble_type == "median":
            # Median of risk scores (robust to outliers)
            return SurvivalEnsemble(base_models, method='median')
        
        elif ensemble_type == "rank":
            # Rank-based ensemble
            return SurvivalEnsemble(base_models, method='rank')
        
        elif ensemble_type == "stacking" or ensemble_type == "blending":
            # Stacking with Cox meta-model
            cv_folds = 5 if ensemble_type == "stacking" else 3
            return SurvivalStackingEnsemble(base_models, cv_folds=cv_folds)
        
        else:
            raise ValueError(f"Unknown ensemble type for survival: {ensemble_type}")
    
    def _evaluate_ensemble_cv(
        self,
        ensemble: BaseEstimator,
        X: pd.DataFrame,
        y: Union[pd.Series, np.ndarray],
        task_type: str,
        base_model_scores: List[float] = None,
        weights: List[float] = None
    ) -> float:
        """
        Evaluate ensemble using cross-validation.
        
        For ensembles with pre-trained models (voting, weighted), we compute
        the weighted average of base model CV scores.
        
        For stacking, we do proper CV since the meta-model is trained fresh.
        """
        
        # For stacking ensembles, do proper CV
        if isinstance(ensemble, (StackingEnsemble, SurvivalStackingEnsemble)):
            try:
                if task_type == "classification":
                    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                    scoring = 'accuracy'
                else:  # survival
                    cv = KFold(n_splits=5, shuffle=True, random_state=42)
                    scoring = None
                
                scores = cross_val_score(ensemble, X, y, cv=cv, scoring=scoring, n_jobs=1)
                return np.mean(scores)
            except Exception as e:
                self.log(f"CV evaluation failed, using training score: {str(e)}", "WARNING")
                if task_type == "classification":
                    y_pred = ensemble.predict(X)
                    return accuracy_score(y, y_pred)
                else:
                    return ensemble.score(X, y)
        
        # For voting/weighted ensembles with pre-trained models,
        # use weighted average of base model CV scores
        if base_model_scores is not None:
            if weights is not None:
                # Weighted average
                return np.average(base_model_scores, weights=weights)
            else:
                # Simple average
                return np.mean(base_model_scores)
        
        # Fallback: evaluate on full training set
        self.log("No base model scores provided, evaluating on training set", "WARNING")
        try:
            if task_type == "classification":
                y_pred = ensemble.predict(X)
                return accuracy_score(y, y_pred)
            else:
                return ensemble.score(X, y)
        except Exception as e:
            self.log(f"Error evaluating ensemble: {str(e)}", "ERROR")
            return 0.0
    
    def _evaluate_ensemble_test(
        self,
        ensemble: BaseEstimator,
        X_test: pd.DataFrame,
        y_test: Union[pd.Series, np.ndarray],
        task_type: str
    ) -> float:
        """Evaluate ensemble on test set"""
        
        try:
            if task_type == "classification":
                y_pred = ensemble.predict(X_test)
                return accuracy_score(y_test, y_pred)
            else:  # survival
                return ensemble.score(X_test, y_test)
                
        except Exception as e:
            self.log(f"Error in test evaluation: {str(e)}", "ERROR")
            return 0.0
