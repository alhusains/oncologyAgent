"""
Structured feature engineering operations for dynamic feature creation.

This module provides a declarative approach to feature engineering where operations
are specified in a structured format (JSON/dict) rather than as code. This makes
feature engineering:
- Auditable and version-controllable
- Safe (no arbitrary code execution)
- Easy to validate and debug
- LLM-friendly for automated feature engineering
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from sklearn.preprocessing import PolynomialFeatures
import warnings


class FeatureCreator:
    """
    Apply structured feature engineering operations.
    
    Supports:
    - Feature interactions (numerical and categorical)
    - Log/sqrt/square transformations
    - Polynomial features
    - Binning/discretization
    - Ratio features
    - Domain-specific features (oncology)
    """
    
    def __init__(self, operations: Optional[Dict[str, Any]] = None, domain: str = "general"):
        """
        Initialize feature creator.
        
        Args:
            operations: Dictionary specifying feature engineering operations
            domain: Domain context (e.g., "oncology", "general")
        """
        self.operations = operations or {}
        self.domain = domain
        self.created_features = []
        self.feature_mapping = {}  # Maps new feature names to their creation logic
        
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all feature engineering operations.
        
        Args:
            df: Input dataframe
            
        Returns:
            Dataframe with new features added
        """
        df_new = df.copy()
        
        # Apply feature interactions
        for interaction in self.operations.get("feature_interactions", []):
            df_new = self._create_interaction(df_new, interaction)
        
        # Apply transformations (log, sqrt, square)
        for transform in self.operations.get("transformations", []):
            df_new = self._apply_transformation(df_new, transform)
        
        # Apply polynomial features
        if "polynomial_features" in self.operations:
            df_new = self._create_polynomial(df_new, self.operations["polynomial_features"])
        
        # Apply binning
        for feature, config in self.operations.get("binning", {}).items():
            df_new = self._create_bins(df_new, feature, config)
        
        # Apply ratio features
        for ratio in self.operations.get("ratio_features", []):
            df_new = self._create_ratio(df_new, ratio)
        
        # Apply domain-specific features
        if self.domain == "oncology" and "oncology_features" in self.operations:
            df_new = self._create_oncology_features(df_new, self.operations["oncology_features"])
        
        return df_new
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform new data using fitted operations.
        
        Note: For most operations (interactions, ratios, etc.), fit_transform and
        transform are the same. Binning may need stored bin edges in future versions.
        
        Args:
            df: Input dataframe
            
        Returns:
            Transformed dataframe
        """
        return self.fit_transform(df)
    
    def _create_interaction(self, df: pd.DataFrame, interaction: Dict[str, Any]) -> pd.DataFrame:
        """
        Create feature interaction.
        
        Args:
            df: Input dataframe
            interaction: Dict with keys:
                - name: Name of new feature
                - features: List of feature names to interact
                - operation: 'multiply', 'add', 'concat' (for categorical)
                - feature_types: Optional list of types ('numerical' or 'categorical')
        
        Returns:
            Dataframe with new interaction feature
        """
        name = interaction["name"]
        features = interaction["features"]
        operation = interaction.get("operation", "multiply")
        feature_types = interaction.get("feature_types", ["numerical"] * len(features))
        
        # Check if features exist
        missing = [f for f in features if f not in df.columns]
        if missing:
            warnings.warn(f"Cannot create interaction '{name}': missing features {missing}")
            return df
        
        try:
            if all(ft == "numerical" for ft in feature_types):
                # Numerical interaction
                if operation == "multiply":
                    df[name] = df[features[0]]
                    for feat in features[1:]:
                        df[name] = df[name] * df[feat]
                elif operation == "add":
                    df[name] = df[features[0]]
                    for feat in features[1:]:
                        df[name] = df[name] + df[feat]
                elif operation == "subtract":
                    df[name] = df[features[0]] - df[features[1]]
                else:
                    warnings.warn(f"Unknown numerical operation: {operation}")
                    return df
            else:
                # Categorical interaction (concatenation)
                df[name] = df[features[0]].astype(str)
                for feat in features[1:]:
                    df[name] = df[name] + "_" + df[feat].astype(str)
            
            self.created_features.append(name)
            self.feature_mapping[name] = interaction
            
        except Exception as e:
            warnings.warn(f"Failed to create interaction '{name}': {str(e)}")
        
        return df
    
    def _apply_transformation(self, df: pd.DataFrame, transform: Dict[str, Any]) -> pd.DataFrame:
        """
        Apply mathematical transformation to features.
        
        Args:
            df: Input dataframe
            transform: Dict with keys:
                - features: List of feature names
                - transform_type: 'log', 'log1p', 'sqrt', 'square', 'cube'
                - prefix: Optional prefix for new feature names
        
        Returns:
            Dataframe with transformed features
        """
        features = transform.get("features", [])
        transform_type = transform.get("transform_type", "log1p")
        prefix = transform.get("prefix", transform_type)
        
        for feature in features:
            if feature not in df.columns:
                warnings.warn(f"Cannot transform '{feature}': not found in dataframe")
                continue
            
            new_name = f"{prefix}_{feature}"
            
            try:
                if transform_type == "log":
                    # Regular log (requires positive values)
                    df[new_name] = np.log(df[feature].clip(lower=1e-10))
                elif transform_type == "log1p":
                    # log(1 + x) - handles zeros
                    df[new_name] = np.log1p(df[feature].clip(lower=0))
                elif transform_type == "sqrt":
                    df[new_name] = np.sqrt(df[feature].clip(lower=0))
                elif transform_type == "square":
                    df[new_name] = df[feature] ** 2
                elif transform_type == "cube":
                    df[new_name] = df[feature] ** 3
                else:
                    warnings.warn(f"Unknown transformation type: {transform_type}")
                    continue
                
                self.created_features.append(new_name)
                self.feature_mapping[new_name] = transform
                
            except Exception as e:
                warnings.warn(f"Failed to transform '{feature}': {str(e)}")
        
        return df
    
    def _create_polynomial(self, df: pd.DataFrame, poly_config: Dict[str, Any]) -> pd.DataFrame:
        """
        Create polynomial features.
        
        Args:
            df: Input dataframe
            poly_config: Dict with keys:
                - features: List of feature names
                - degree: Polynomial degree (default 2)
                - interaction_only: Only interaction terms, no powers (default True)
                - include_bias: Include bias term (default False)
        
        Returns:
            Dataframe with polynomial features
        """
        features = poly_config.get("features", [])
        degree = poly_config.get("degree", 2)
        interaction_only = poly_config.get("interaction_only", True)
        include_bias = poly_config.get("include_bias", False)
        
        # Check if features exist
        missing = [f for f in features if f not in df.columns]
        if missing:
            warnings.warn(f"Cannot create polynomial features: missing {missing}")
            return df
        
        try:
            poly = PolynomialFeatures(
                degree=degree,
                interaction_only=interaction_only,
                include_bias=include_bias
            )
            
            # Extract feature subset
            X_subset = df[features].values
            X_poly = poly.fit_transform(X_subset)
            
            # Get feature names
            poly_names = poly.get_feature_names_out(features)
            
            # Add new features (skip existing ones)
            for i, name in enumerate(poly_names):
                if name not in features:  # Skip original features
                    df[f"poly_{name}"] = X_poly[:, i]
                    self.created_features.append(f"poly_{name}")
            
            self.feature_mapping["polynomial_features"] = poly_config
            
        except Exception as e:
            warnings.warn(f"Failed to create polynomial features: {str(e)}")
        
        return df
    
    def _create_bins(self, df: pd.DataFrame, feature: str, config: Dict[str, Any]) -> pd.DataFrame:
        """
        Create binned/discretized version of continuous feature.
        
        Args:
            df: Input dataframe
            feature: Feature name to bin
            config: Dict with keys:
                - method: 'quantile', 'uniform', or 'custom'
                - n_bins: Number of bins (for quantile/uniform)
                - labels: Optional bin labels
                - bins: Custom bin edges (for custom method)
        
        Returns:
            Dataframe with binned feature
        """
        if feature not in df.columns:
            warnings.warn(f"Cannot bin '{feature}': not found in dataframe")
            return df
        
        method = config.get("method", "quantile")
        n_bins = config.get("n_bins", 5)
        labels = config.get("labels")
        custom_bins = config.get("bins")
        
        new_name = f"{feature}_binned"
        
        try:
            if method == "quantile":
                df[new_name] = pd.qcut(
                    df[feature], 
                    q=n_bins, 
                    labels=labels,
                    duplicates='drop'
                )
            elif method == "uniform":
                df[new_name] = pd.cut(
                    df[feature],
                    bins=n_bins,
                    labels=labels
                )
            elif method == "custom" and custom_bins:
                df[new_name] = pd.cut(
                    df[feature],
                    bins=custom_bins,
                    labels=labels
                )
            else:
                warnings.warn(f"Unknown binning method: {method}")
                return df
            
            # Convert to string for consistency
            df[new_name] = df[new_name].astype(str)
            
            self.created_features.append(new_name)
            self.feature_mapping[new_name] = config
            
        except Exception as e:
            warnings.warn(f"Failed to bin '{feature}': {str(e)}")
        
        return df
    
    def _create_ratio(self, df: pd.DataFrame, ratio: Dict[str, Any]) -> pd.DataFrame:
        """
        Create ratio feature.
        
        Args:
            df: Input dataframe
            ratio: Dict with keys:
                - name: Name of new feature
                - numerator: Numerator feature name
                - denominator: Denominator feature name
                - denominator_power: Power to raise denominator (default 1)
        
        Returns:
            Dataframe with ratio feature
        """
        name = ratio["name"]
        numerator = ratio["numerator"]
        denominator = ratio["denominator"]
        denom_power = ratio.get("denominator_power", 1)
        
        if numerator not in df.columns or denominator not in df.columns:
            warnings.warn(f"Cannot create ratio '{name}': missing features")
            return df
        
        try:
            # Add small epsilon to avoid division by zero
            denom_values = df[denominator] ** denom_power
            df[name] = df[numerator] / (denom_values + 1e-8)
            
            self.created_features.append(name)
            self.feature_mapping[name] = ratio
            
        except Exception as e:
            warnings.warn(f"Failed to create ratio '{name}': {str(e)}")
        
        return df
    
    def _create_oncology_features(self, df: pd.DataFrame, oncology_config: Dict[str, Any]) -> pd.DataFrame:
        """
        Create domain-specific oncology features.
        
        Common oncology features:
        - Risk scores (age × stage, smoking × sex, etc.)
        - BMI (weight / height²)
        - Performance status interactions
        - Biomarker transformations
        
        Args:
            df: Input dataframe
            oncology_config: Dict specifying which oncology features to create
        
        Returns:
            Dataframe with oncology-specific features
        """
        
        # BMI calculation
        if oncology_config.get("calculate_bmi", False):
            if "weight_kg" in df.columns and "height_m" in df.columns:
                df["bmi"] = df["weight_kg"] / (df["height_m"] ** 2 + 1e-8)
                self.created_features.append("bmi")
            elif "weight" in df.columns and "height" in df.columns:
                # Try without units
                df["bmi"] = df["weight"] / (df["height"] ** 2 + 1e-8)
                self.created_features.append("bmi")
        
        # Age-stage risk
        if oncology_config.get("age_stage_risk", False):
            if "age" in df.columns and any("stage" in col.lower() for col in df.columns):
                stage_col = next((col for col in df.columns if "stage" in col.lower()), None)
                if stage_col:
                    # Map stages to risk levels (I=1, II=2, III=3, IV=4)
                    stage_map = {"I": 1, "II": 2, "III": 3, "IV": 4, "1": 1, "2": 2, "3": 3, "4": 4}
                    df["age_stage_risk"] = df["age"] * df[stage_col].astype(str).map(stage_map).fillna(2)
                    self.created_features.append("age_stage_risk")
        
        # Smoking-sex interaction (for lung cancer, etc.)
        if oncology_config.get("smoking_sex_interaction", False):
            if "sex" in df.columns and any("smok" in col.lower() for col in df.columns):
                smoking_col = next((col for col in df.columns if "smok" in col.lower()), None)
                if smoking_col:
                    sex_map = {"M": 1, "Male": 1, "F": 0.7, "Female": 0.7}
                    smoking_map = {"current": 2.0, "Current": 2.0, "former": 1.2, "Former": 1.2, "never": 0.1, "Never": 0.1}
                    
                    sex_risk = df["sex"].astype(str).map(sex_map).fillna(1.0)
                    smoking_risk = df[smoking_col].astype(str).map(smoking_map).fillna(1.0)
                    df["smoking_sex_risk"] = sex_risk * smoking_risk
                    self.created_features.append("smoking_sex_risk")
        
        # Biomarker log transforms (PSA, CEA, etc.)
        biomarkers = oncology_config.get("log_biomarkers", [])
        for biomarker in biomarkers:
            if biomarker in df.columns:
                df[f"log_{biomarker}"] = np.log1p(df[biomarker].clip(lower=0))
                self.created_features.append(f"log_{biomarker}")
        
        return df
    
    def get_created_features(self) -> List[str]:
        """Get list of newly created feature names."""
        return self.created_features
    
    def get_feature_report(self) -> Dict[str, Any]:
        """Get report of feature engineering operations."""
        return {
            "n_features_created": len(self.created_features),
            "created_features": self.created_features,
            "operations_applied": {
                "interactions": len(self.operations.get("feature_interactions", [])),
                "transformations": len(self.operations.get("transformations", [])),
                "polynomial": 1 if "polynomial_features" in self.operations else 0,
                "binning": len(self.operations.get("binning", {})),
                "ratios": len(self.operations.get("ratio_features", [])),
                "domain_specific": 1 if self.domain == "oncology" and "oncology_features" in self.operations else 0
            },
            "feature_mapping": self.feature_mapping
        }


def get_default_oncology_operations() -> Dict[str, Any]:
    """
    Get default feature engineering operations for oncology datasets.
    
    Returns:
        Dictionary of recommended operations for oncology data
    """
    return {
        "oncology_features": {
            "calculate_bmi": True,
            "age_stage_risk": True,
            "smoking_sex_interaction": True,
            "log_biomarkers": ["psa_level", "cea_level", "ca125", "afp", "tumor_size", "tumor_volume"]
        },
        "transformations": [
            {
                "features": ["tumor_size", "tumor_volume"],
                "transform_type": "log1p",
                "prefix": "log"
            }
        ]
    }

