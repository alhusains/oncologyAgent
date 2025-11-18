"""Error analysis module to help agent understand failures"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from collections import Counter

from ..core.config import Config


class ErrorAnalyzer:
    """Analyzes model errors to provide insights for improvement"""
    
    def __init__(self, config: Config):
        self.config = config
    
    def analyze_misclassifications(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        y_pred: np.ndarray,
        feature_names: List[str],
        y_pred_proba: Optional[np.ndarray] = None,
        n_examples: int = 10
    ) -> Dict[str, Any]:
        """
        Analyze misclassified examples to understand patterns
        
        Args:
            X_test: Test features
            y_test: True labels
            y_pred: Predicted labels
            feature_names: Names of features
            y_pred_proba: Prediction probabilities (optional)
            n_examples: Number of examples to analyze per class
            
        Returns:
            Dictionary with error analysis results
        """
        
        # Find misclassified indices
        misclassified_mask = y_test != y_pred
        misclassified_indices = np.where(misclassified_mask)[0]
        
        if len(misclassified_indices) == 0:
            return {
                "error_rate": 0.0,
                "total_errors": 0,
                "message": "Perfect predictions! No errors to analyze."
            }
        
        # Convert to DataFrame for easier analysis
        df_test = pd.DataFrame(X_test, columns=feature_names)
        df_test['true_label'] = y_test
        df_test['predicted_label'] = y_pred
        df_test['is_error'] = misclassified_mask
        
        if y_pred_proba is not None:
            if y_pred_proba.ndim == 1:
                df_test['confidence'] = y_pred_proba
            else:
                # For each prediction, get the probability of the predicted class
                df_test['confidence'] = np.max(y_pred_proba, axis=1)
        
        # Overall error statistics
        error_rate = len(misclassified_indices) / len(y_test)
        
        # Per-class error analysis
        class_errors = self._analyze_per_class_errors(df_test)
        
        # Confusion patterns
        confusion_patterns = self._analyze_confusion_patterns(
            df_test[df_test['is_error']]
        )
        
        # Feature characteristics of errors
        error_features = self._analyze_error_features(
            df_test[df_test['is_error']],
            df_test[~df_test['is_error']],
            feature_names
        )
        
        # Sample misclassified examples
        error_examples = self._get_error_examples(
            df_test[df_test['is_error']],
            n_examples
        )
        
        # Confidence analysis
        confidence_analysis = self._analyze_confidence(df_test)
        
        result = {
            "error_rate": float(error_rate),
            "total_errors": int(len(misclassified_indices)),
            "total_samples": int(len(y_test)),
            "class_errors": class_errors,
            "confusion_patterns": confusion_patterns,
            "error_features": error_features,
            "error_examples": error_examples,
            "confidence_analysis": confidence_analysis
        }
        
        return result
    
    def _analyze_per_class_errors(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze errors per class"""
        class_analysis = {}
        
        for true_class in df['true_label'].unique():
            class_mask = df['true_label'] == true_class
            class_df = df[class_mask]
            
            n_samples = len(class_df)
            n_errors = class_df['is_error'].sum()
            error_rate = n_errors / n_samples if n_samples > 0 else 0
            
            # What classes are confused with this class?
            errors_df = class_df[class_df['is_error']]
            confused_with = {}
            if len(errors_df) > 0:
                for pred_class, count in errors_df['predicted_label'].value_counts().items():
                    confused_with[int(pred_class)] = int(count)
            
            class_analysis[int(true_class)] = {
                "n_samples": int(n_samples),
                "n_errors": int(n_errors),
                "error_rate": float(error_rate),
                "confused_with": confused_with
            }
        
        return class_analysis
    
    def _analyze_confusion_patterns(self, errors_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze which classes are most commonly confused"""
        if len(errors_df) == 0:
            return {}
        
        confusion_pairs = []
        for _, row in errors_df.iterrows():
            pair = (int(row['true_label']), int(row['predicted_label']))
            confusion_pairs.append(pair)
        
        # Count most common confusions
        confusion_counts = Counter(confusion_pairs)
        
        patterns = {}
        for (true_class, pred_class), count in confusion_counts.most_common(5):
            key = f"true_{true_class}_pred_{pred_class}"
            patterns[key] = {
                "true_class": true_class,
                "predicted_class": pred_class,
                "count": int(count),
                "description": f"Class {true_class} misclassified as {pred_class}"
            }
        
        return patterns
    
    def _analyze_error_features(
        self,
        errors_df: pd.DataFrame,
        correct_df: pd.DataFrame,
        feature_names: List[str]
    ) -> Dict[str, Any]:
        """Compare feature distributions between errors and correct predictions"""
        
        feature_analysis = {}
        
        # Analyze top features that differ between errors and correct predictions
        for feature in feature_names[:10]:  # Analyze top 10 features
            if feature not in errors_df.columns:
                continue
            
            try:
                error_mean = errors_df[feature].mean()
                correct_mean = correct_df[feature].mean()
                error_std = errors_df[feature].std()
                correct_std = correct_df[feature].std()
                
                # Calculate difference
                mean_diff = abs(error_mean - correct_mean)
                relative_diff = mean_diff / (abs(correct_mean) + 1e-10)
                
                if relative_diff > 0.1:  # Only report significant differences
                    feature_analysis[feature] = {
                        "error_mean": float(error_mean),
                        "correct_mean": float(correct_mean),
                        "error_std": float(error_std),
                        "correct_std": float(correct_std),
                        "relative_difference": float(relative_diff)
                    }
            except (TypeError, ValueError):
                # Skip non-numeric features
                continue
        
        return feature_analysis
    
    def _get_error_examples(
        self,
        errors_df: pd.DataFrame,
        n_examples: int
    ) -> List[Dict[str, Any]]:
        """Get sample misclassified examples"""
        
        examples = []
        
        # Get diverse examples (different confusion types)
        confusion_types = errors_df.groupby(['true_label', 'predicted_label'])
        
        for (true_class, pred_class), group in confusion_types:
            # Get up to 2 examples per confusion type
            sample = group.head(2)
            
            for idx, row in sample.iterrows():
                example = {
                    "true_class": int(row['true_label']),
                    "predicted_class": int(row['predicted_label']),
                }
                
                if 'confidence' in row:
                    example["confidence"] = float(row['confidence'])
                
                # Add top 5 feature values
                feature_cols = [col for col in row.index 
                               if col not in ['true_label', 'predicted_label', 'is_error', 'confidence']]
                example["features"] = {}
                for col in feature_cols[:5]:
                    try:
                        example["features"][col] = float(row[col])
                    except (TypeError, ValueError):
                        example["features"][col] = str(row[col])
                
                examples.append(example)
                
                if len(examples) >= n_examples:
                    break
            
            if len(examples) >= n_examples:
                break
        
        return examples
    
    def _analyze_confidence(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze prediction confidence for errors vs correct predictions"""
        
        if 'confidence' not in df.columns:
            return {"available": False}
        
        errors_df = df[df['is_error']]
        correct_df = df[~df['is_error']]
        
        analysis = {
            "available": True,
            "error_confidence_mean": float(errors_df['confidence'].mean()) if len(errors_df) > 0 else 0.0,
            "correct_confidence_mean": float(correct_df['confidence'].mean()) if len(correct_df) > 0 else 0.0,
            "low_confidence_errors": int((errors_df['confidence'] < 0.6).sum()) if len(errors_df) > 0 else 0,
            "high_confidence_errors": int((errors_df['confidence'] > 0.8).sum()) if len(errors_df) > 0 else 0
        }
        
        analysis["insight"] = self._generate_confidence_insight(analysis)
        
        return analysis
    
    def _generate_confidence_insight(self, analysis: Dict[str, Any]) -> str:
        """Generate human-readable insight about confidence"""
        
        if not analysis["available"]:
            return "Confidence information not available"
        
        error_conf = analysis["error_confidence_mean"]
        correct_conf = analysis["correct_confidence_mean"]
        high_conf_errors = analysis["high_confidence_errors"]
        
        if error_conf > 0.7:
            return "Model is overconfident in its errors - may need calibration"
        elif high_conf_errors > 0:
            return f"{high_conf_errors} high-confidence errors suggest systematic blind spots"
        elif error_conf < 0.5:
            return "Low confidence on errors - model recognizes uncertainty"
        else:
            return "Model has reasonable confidence calibration"
    
    def generate_improvement_suggestions(
        self,
        error_analysis: Dict[str, Any]
    ) -> List[str]:
        """Generate actionable suggestions based on error analysis"""
        
        suggestions = []
        
        # Check error rate
        error_rate = error_analysis.get("error_rate", 0)
        if error_rate > 0.5:
            suggestions.append(
                "High error rate (>50%) - consider collecting more data or trying simpler models"
            )
        elif error_rate > 0.3:
            suggestions.append(
                "Moderate error rate - try feature engineering or ensemble methods"
            )
        
        # Check class imbalance in errors
        class_errors = error_analysis.get("class_errors", {})
        max_error_rate = 0
        problematic_class = None
        for class_id, info in class_errors.items():
            if info["error_rate"] > max_error_rate:
                max_error_rate = info["error_rate"]
                problematic_class = class_id
        
        if max_error_rate > 0.5 and problematic_class is not None:
            suggestions.append(
                f"Class {problematic_class} has {max_error_rate:.1%} error rate - "
                "consider class-specific features or oversampling"
            )
        
        # Check confusion patterns
        confusion = error_analysis.get("confusion_patterns", {})
        if len(confusion) > 0:
            top_confusion = list(confusion.values())[0]
            suggestions.append(
                f"Most common confusion: Class {top_confusion['true_class']} → "
                f"Class {top_confusion['predicted_class']}. "
                "Consider adding features to distinguish these classes"
            )
        
        # Check feature differences
        error_features = error_analysis.get("error_features", {})
        if len(error_features) > 0:
            top_feature = list(error_features.keys())[0]
            suggestions.append(
                f"Feature '{top_feature}' shows different distribution in errors - "
                "consider feature transformation or interaction terms"
            )
        
        # Check confidence
        conf_analysis = error_analysis.get("confidence_analysis", {})
        if conf_analysis.get("high_confidence_errors", 0) > 0:
            suggestions.append(
                "Model is confident but wrong on some examples - "
                "may benefit from additional features or model ensemble"
            )
        
        return suggestions

