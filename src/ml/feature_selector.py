"""
Feature selection module for identifying and removing redundant or uninformative features.

Handles:
- Missing value analysis
- Low variance features
- Highly correlated features
- Feature importance-based selection
- Mutual information-based selection
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression, VarianceThreshold
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
import warnings


class FeatureSelector:
    """
    Intelligent feature selection with multiple strategies.
    """
    
    def __init__(
        self,
        missing_threshold: float = 0.5,
        variance_threshold: float = 0.01,
        correlation_threshold: float = 0.95,
        max_features: Optional[int] = None
    ):
        """
        Initialize feature selector.
        
        Args:
            missing_threshold: Remove features with more than this fraction of missing values
            variance_threshold: Remove features with variance below this threshold
            correlation_threshold: Remove features with correlation above this threshold
            max_features: Maximum number of features to keep (None = no limit)
        """
        self.missing_threshold = missing_threshold
        self.variance_threshold = variance_threshold
        self.correlation_threshold = correlation_threshold
        self.max_features = max_features
        
        self.removed_features = {
            "high_missing": [],
            "low_variance": [],
            "high_correlation": [],
            "low_importance": []
        }
        self.selected_features = []
        self.feature_scores = {}
        
    def fit_select(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        task_type: str,
        feature_names: List[str]
    ) -> Tuple[List[str], Dict[str, Any]]:
        """
        Perform feature selection.
        
        Args:
            X: Feature dataframe
            y: Target variable
            task_type: 'classification', 'regression', or 'survival'
            feature_names: List of feature names
            
        Returns:
            Tuple of (selected_features, selection_report)
        """
        if len(feature_names) == 0:
            return [], {"error": "No features provided"}
        
        # Start with all features
        remaining_features = feature_names.copy()
        
        # 1. Remove features with too many missing values
        remaining_features = self._filter_missing_values(X, remaining_features)
        
        # 2. Remove low variance features (only for numerical features)
        remaining_features = self._filter_low_variance(X, remaining_features)
        
        # 3. Remove highly correlated features
        remaining_features = self._filter_correlations(X, remaining_features)
        
        # 4. If still too many features, use importance-based selection
        if self.max_features and len(remaining_features) > self.max_features:
            remaining_features = self._select_by_importance(
                X, y, remaining_features, task_type
            )
        
        self.selected_features = remaining_features
        
        # Generate report
        report = self._generate_selection_report(feature_names)
        
        return self.selected_features, report
    
    def _filter_missing_values(self, X: pd.DataFrame, features: List[str]) -> List[str]:
        """Remove features with too many missing values."""
        high_missing = []
        
        for feature in features:
            if feature in X.columns:
                missing_frac = X[feature].isnull().sum() / len(X)
                if missing_frac > self.missing_threshold:
                    high_missing.append(feature)
        
        self.removed_features["high_missing"] = high_missing
        remaining = [f for f in features if f not in high_missing]
        
        if high_missing:
            print(f"  🗑️  Removed {len(high_missing)} features with >{self.missing_threshold:.0%} missing values")
            print(f"      Features removed: {', '.join(high_missing)}")
        
        return remaining
    
    def _filter_low_variance(self, X: pd.DataFrame, features: List[str]) -> List[str]:
        """Remove features with very low variance."""
        low_variance = []
        
        # Only check numerical features
        numerical_features = [f for f in features if f in X.columns and pd.api.types.is_numeric_dtype(X[f])]
        
        for feature in numerical_features:
            try:
                # Calculate variance (after filling NaN with mean)
                variance = X[feature].fillna(X[feature].mean()).var()
                if variance < self.variance_threshold:
                    low_variance.append(feature)
            except:
                # If variance calculation fails, keep the feature
                pass
        
        self.removed_features["low_variance"] = low_variance
        remaining = [f for f in features if f not in low_variance]
        
        if low_variance:
            print(f"  🗑️  Removed {len(low_variance)} low-variance features")
            print(f"      Features removed: {', '.join(low_variance)}")
        
        return remaining
    
    def _filter_correlations(self, X: pd.DataFrame, features: List[str]) -> List[str]:
        """Remove highly correlated features."""
        # Only check numerical features
        numerical_features = [f for f in features if f in X.columns and pd.api.types.is_numeric_dtype(X[f])]
        
        if len(numerical_features) < 2:
            return features
        
        try:
            # Calculate correlation matrix
            X_numerical = X[numerical_features].fillna(X[numerical_features].mean())
            corr_matrix = X_numerical.corr().abs()
            
            # Find pairs of highly correlated features
            high_corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    if corr_matrix.iloc[i, j] > self.correlation_threshold:
                        high_corr_pairs.append((
                            corr_matrix.columns[i],
                            corr_matrix.columns[j],
                            corr_matrix.iloc[i, j]
                        ))
            
            # Remove one feature from each highly correlated pair
            # Keep the first one, remove the second
            to_remove = set()
            for feat1, feat2, corr in high_corr_pairs:
                if feat2 not in to_remove:
                    to_remove.add(feat2)
            
            high_corr = list(to_remove)
            self.removed_features["high_correlation"] = high_corr
            remaining = [f for f in features if f not in high_corr]
            
            if high_corr:
                print(f"  🗑️  Removed {len(high_corr)} highly correlated features (r>{self.correlation_threshold})")
                print(f"      Features removed: {', '.join(high_corr)}")
            
            return remaining
            
        except Exception as e:
            warnings.warn(f"Correlation filtering failed: {str(e)}")
            return features
    
    def _select_by_importance(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        features: List[str],
        task_type: str
    ) -> List[str]:
        """Select top features by importance/mutual information."""
        
        try:
            # Prepare data
            X_subset = X[features].fillna(X[features].mean() if len(X[features].select_dtypes(include=[np.number]).columns) > 0 else 0)
            
            # Handle survival task
            if task_type == "survival":
                if hasattr(y, 'dtype') and y.dtype.names:
                    # Structured array - use event indicator for selection
                    y_for_selection = y['event'].astype(int)
                    selection_method = mutual_info_classif
                else:
                    y_for_selection = y
                    selection_method = mutual_info_regression
            elif task_type == "classification":
                y_for_selection = y
                selection_method = mutual_info_classif
            else:  # regression
                y_for_selection = y
                selection_method = mutual_info_regression
            
            # Calculate feature scores
            scores = selection_method(X_subset, y_for_selection, random_state=42)
            
            # Store scores
            for feat, score in zip(features, scores):
                self.feature_scores[feat] = float(score)
            
            # Select top features
            top_indices = np.argsort(scores)[::-1][:self.max_features]
            selected = [features[i] for i in top_indices]
            
            # Track removed features
            removed = [f for f in features if f not in selected]
            self.removed_features["low_importance"] = removed
            
            if removed:
                print(f"  🗑️  Selected top {self.max_features} features by importance")
            
            return selected
            
        except Exception as e:
            warnings.warn(f"Importance-based selection failed: {str(e)}")
            # Fallback: just return first max_features
            return features[:self.max_features]
    
    def _generate_selection_report(self, original_features: List[str]) -> Dict[str, Any]:
        """Generate feature selection report."""
        total_removed = sum(len(v) for v in self.removed_features.values())
        
        report = {
            "original_n_features": len(original_features),
            "selected_n_features": len(self.selected_features),
            "n_features_removed": total_removed,
            "removal_breakdown": {
                "high_missing": len(self.removed_features["high_missing"]),
                "low_variance": len(self.removed_features["low_variance"]),
                "high_correlation": len(self.removed_features["high_correlation"]),
                "low_importance": len(self.removed_features["low_importance"])
            },
            "removed_features": self.removed_features,
            "selected_features": self.selected_features,
            "feature_scores": self.feature_scores,
            "selection_criteria": {
                "missing_threshold": self.missing_threshold,
                "variance_threshold": self.variance_threshold,
                "correlation_threshold": self.correlation_threshold,
                "max_features": self.max_features
            }
        }
        
        return report
    
    def get_removed_features(self) -> Dict[str, List[str]]:
        """Get dictionary of removed features by reason."""
        return self.removed_features
    
    def get_selected_features(self) -> List[str]:
        """Get list of selected features."""
        return self.selected_features
    
    def get_feature_scores(self) -> Dict[str, float]:
        """Get feature importance scores."""
        return self.feature_scores


def analyze_feature_quality(
    X: pd.DataFrame,
    feature_names: List[str]
) -> Dict[str, Any]:
    """
    Analyze feature quality without removing anything.
    
    Useful for understanding feature characteristics before selection.
    
    Args:
        X: Feature dataframe
        feature_names: List of feature names
        
    Returns:
        Dictionary with feature quality metrics
    """
    quality_report = {
        "missing_values": {},
        "variance": {},
        "unique_values": {},
        "data_types": {}
    }
    
    for feature in feature_names:
        if feature not in X.columns:
            continue
        
        # Missing values
        missing_frac = X[feature].isnull().sum() / len(X)
        quality_report["missing_values"][feature] = float(missing_frac)
        
        # Variance (for numerical features)
        if pd.api.types.is_numeric_dtype(X[feature]):
            variance = X[feature].fillna(X[feature].mean()).var()
            quality_report["variance"][feature] = float(variance)
        
        # Unique values
        n_unique = X[feature].nunique()
        quality_report["unique_values"][feature] = int(n_unique)
        
        # Data type
        quality_report["data_types"][feature] = str(X[feature].dtype)
    
    return quality_report

