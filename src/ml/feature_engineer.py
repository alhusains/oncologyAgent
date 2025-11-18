"""Feature engineering module with LLM-guided preprocessing"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pickle
import warnings

from ..core.base_agent import LLMAgent
from ..core.state import TaskType, AgentResult
from ..core.config import Config
from ..llm.prompts import PromptTemplates
from .feature_creator import FeatureCreator, get_default_oncology_operations
from .feature_selector import FeatureSelector, analyze_feature_quality


class FeatureEngineer(LLMAgent):
    """Agent for feature engineering with LLM guidance"""
    
    def __init__(self, config: Config):
        super().__init__("feature_engineer", config)
        self.preprocessor = None
        self.feature_names = None
        self.feature_creator = None
        self.feature_selector = None
        self.feature_engineering_history = []  # Track iterations for refinement
        
    def get_task_type(self) -> TaskType:
        return TaskType.FEATURE_ENGINEERING
    
    async def execute(self, inputs: Dict[str, Any]) -> AgentResult:
        """Execute feature engineering"""
        self.validate_inputs(inputs, ["dataset_path", "data_analysis", "target_variable"])
        
        dataset_path = inputs["dataset_path"]
        data_analysis = inputs["data_analysis"]
        target_variable = inputs["target_variable"]
        time_variable = inputs.get("time_variable", None)
        testset_path = inputs.get("testset_path", None)
        feature_engineering_config = inputs.get("feature_engineering_config", None)  # For refinement
        
        # Load the data
        df = await self._load_data(dataset_path)
        
        # Load test set if provided
        df_test = None
        if testset_path:
            self.log(f"Loading pre-split test set from: {testset_path}")
            df_test = await self._load_data(testset_path)
        
        # Get LLM recommendations for feature engineering
        # Use provided config if available (for refinement), otherwise get new recommendations
        if feature_engineering_config:
            self.log("Using provided feature engineering configuration (refinement mode)")
            llm_recommendations = feature_engineering_config
        else:
            llm_recommendations = await self._get_llm_recommendations(data_analysis, df)
        
        # Perform feature engineering
        try:
            engineering_result = await self._engineer_features(
                df, data_analysis, llm_recommendations, target_variable, time_variable, df_test
            )
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            self.log(f"Feature engineering failed with error: {str(e)}", "ERROR")
            self.log(f"Full traceback:\n{error_details}", "ERROR")
            raise
        
        return self.create_result(
            inputs=inputs,
            outputs=engineering_result,
            confidence_score=0.85
        )
    
    async def engineer_features(
        self, 
        dataset_path: str, 
        data_analysis: Dict[str, Any], 
        target_variable: str,
        time_variable: Optional[str] = None,
        testset_path: Optional[str] = None,
        feature_engineering_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Main entry point for feature engineering"""
        result = await self.execute({
            "dataset_path": dataset_path,
            "data_analysis": data_analysis,
            "target_variable": target_variable,
            "time_variable": time_variable,
            "testset_path": testset_path,
            "feature_engineering_config": feature_engineering_config
        })
        
        if result.status.value == "completed":
            return result.outputs
        else:
            raise Exception(f"Feature engineering failed: {result.error_message}")
    
    async def _load_data(self, dataset_path: str) -> pd.DataFrame:
        """Load dataset"""
        self.log(f"Loading dataset: {dataset_path}")
        
        file_path = Path(dataset_path)
        if file_path.suffix.lower() == '.xlsx':
            df = pd.read_excel(dataset_path)
        elif file_path.suffix.lower() == '.csv':
            df = pd.read_csv(dataset_path)
        elif file_path.suffix.lower() == '.parquet':
            df = pd.read_parquet(dataset_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
        
        self.log(f"Dataset loaded: {df.shape}")
        
        # Check for duplicate column names
        duplicate_cols = df.columns[df.columns.duplicated()].unique()
        if len(duplicate_cols) > 0:
            self.log(f"WARNING: Found {len(duplicate_cols)} duplicate column names: {list(duplicate_cols)}", "WARNING")
            self.log("Renaming duplicate columns...", "WARNING")
            
            # Rename duplicates by appending _1, _2, etc.
            cols = pd.Series(df.columns)
            for dup in duplicate_cols:
                dup_indices = cols[cols == dup].index
                for i, idx in enumerate(dup_indices[1:], start=1):  # Keep first, rename rest
                    cols[idx] = f"{dup}_{i}"
            df.columns = cols
            
            self.log(f"Columns after renaming: {list(df.columns)}")
        
        return df
    
    async def _get_llm_recommendations(self, data_analysis: Dict[str, Any], df: pd.DataFrame) -> Dict[str, Any]:
        """Get LLM recommendations for feature engineering"""
        self.log("Getting LLM recommendations for feature engineering...")
        
        try:
            # Analyze feature quality
            feature_names = data_analysis.get("suggested_features", [])
            quality_report = analyze_feature_quality(df, feature_names[:20])  # Sample for efficiency
            
            # Prepare data for LLM with ACTUAL feature names
            data_schema = {
                "shape": f"{data_analysis.get('n_rows', 0)} rows × {data_analysis.get('n_cols', 0)} columns",
                "target_variable": data_analysis.get("suggested_target"),
                "task_type": data_analysis.get("task_type"),
                "categorical_features": data_analysis.get("categorical_features", []),  # Include ALL
                "numerical_features": data_analysis.get("numerical_features", []),  # Include ALL
                "data_quality_issues": data_analysis.get("data_quality", []),
                "missing_values": data_analysis.get("missing_values", {}),
                "feature_quality_sample": quality_report,
                "note": "Above are the ACTUAL feature names in this dataset. Base your suggestions on these specific features."
            }
            
            target_info = {
                "target": data_analysis.get("suggested_target"),
                "type": data_analysis.get("task_type"),
                "domain": "oncology"  # Domain-specific
            }
            
            # Use LLM to suggest feature engineering
            recommendations = await self.llm_client.suggest_feature_engineering(
                data_schema, target_info
            )
            
            # DEBUG: Log FULL LLM response for debugging
            import json
            self.log("="*80)
            self.log("FULL LLM RESPONSE (for debugging):")
            self.log(json.dumps(recommendations, indent=2))
            self.log("="*80)
            
            # DEBUG: Log what LLM returned
            self.log(f"LLM raw recommendations keys: {list(recommendations.keys())}")
            
            # Handle various response formats from LLM
            # Sometimes LLM returns 'feature_engineering_techniques' or other keys
            if "feature_creator_operations" not in recommendations:
                self.log("No feature_creator_operations in LLM response, checking alternatives...")
                
                # Check if LLM used a different key
                possible_keys = ['feature_engineering_techniques', 'feature_engineering', 
                                'feature_creation', 'advanced_features', 'new_features',
                                'feature_engineering_suggestions']  # LLM loves this one!
                found_key = None
                for key in possible_keys:
                    if key in recommendations:
                        self.log(f"Found alternative key: '{key}', will try to parse it")
                        found_key = key
                        break
                
                # Try to extract operations from whatever the LLM returned
                recommendations["feature_creator_operations"] = self._extract_structured_operations(
                    recommendations, data_analysis, source_key=found_key
                )
            elif not recommendations["feature_creator_operations"]:
                self.log("Empty feature_creator_operations, extracting from recommendations...")
                recommendations["feature_creator_operations"] = self._extract_structured_operations(
                    recommendations, data_analysis
                )
            
            # Log what we ended up with
            if recommendations.get("feature_creator_operations"):
                ops = recommendations["feature_creator_operations"]
                self.log(f"Final feature creator operations: {list(ops.keys()) if isinstance(ops, dict) else 'Invalid format'}")
            else:
                self.log("WARNING: No feature operations extracted - will skip feature creation", "WARNING")
            
            # Add feature selection config if not present
            if "feature_selection" not in recommendations:
                recommendations["feature_selection"] = {
                    "enabled": True,
                    "missing_threshold": 0.3,      # More strict: remove if >30% missing
                    "variance_threshold": 0.01,    # Keep default for now
                    "correlation_threshold": 0.85,  # More aggressive: remove highly correlated (>0.85)
                    "max_features": None  # No limit by default
                }
            
            self.log("LLM recommendations obtained and processed")
            return recommendations
            
        except Exception as e:
            self.log(f"LLM recommendations failed: {str(e)}", "ERROR")
            return self._get_default_recommendations(data_analysis)
    
    def _get_default_recommendations(self, data_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Default feature engineering recommendations if LLM fails"""
        return {
            "numerical_transformations": {
                "scaling_strategy": "standard",
                "imputation_strategy": "median"
            },
            "categorical_encoding": {
                "low_cardinality": "onehot",
                "high_cardinality": "drop",  # Changed from "label" to "drop"
                "imputation_strategy": "most_frequent"
            },
            "feature_selection": {
                "enabled": True,
                "missing_threshold": 0.3,      # More strict
                "variance_threshold": 0.01,
                "correlation_threshold": 0.85,  # More aggressive
                "max_features": None
            },
            "feature_creator_operations": get_default_oncology_operations()
        }
    
    def _extract_structured_operations(
        self, 
        recommendations: Dict[str, Any], 
        data_analysis: Dict[str, Any],
        source_key: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Extract structured operations from LLM recommendations for backwards compatibility.
        
        This converts old-style or alternative format recommendations into the new structured format.
        
        Args:
            recommendations: Full LLM response
            data_analysis: Data analysis results
            source_key: Alternative key where LLM put the recommendations (if not in root)
        """
        operations = {}
        
        # If LLM used an alternative key, look there first
        source = recommendations
        if source_key and source_key in recommendations:
            self.log(f"Extracting from alternative key: {source_key}")
            source = recommendations[source_key]
            # DEBUG: Log what's actually inside
            if isinstance(source, dict):
                self.log(f"Contents of '{source_key}': {list(source.keys())}")
            else:
                self.log(f"WARNING: '{source_key}' is not a dict, it's {type(source).__name__}", "WARNING")
        
        # Extract feature interactions if mentioned
        if "feature_interactions" in source:
            interactions = source["feature_interactions"]
            if isinstance(interactions, list) and len(interactions) > 0:
                operations["feature_interactions"] = interactions
                self.log(f"Extracted {len(interactions)} feature interactions")
            elif isinstance(interactions, dict):
                # Maybe LLM put them in a nested structure
                self.log(f"feature_interactions is a dict with keys: {list(interactions.keys())}")
                
                # Handle "suggestions" key with verbose format
                if "suggestions" in interactions:
                    suggestions = interactions["suggestions"]
                    if isinstance(suggestions, list):
                        parsed_interactions = self._parse_suggestion_interactions(suggestions, data_analysis)
                        if parsed_interactions:
                            operations["feature_interactions"] = parsed_interactions
                            self.log(f"Parsed {len(parsed_interactions)} interactions from suggestions")
                
                # Handle "interaction_terms" key with string format
                elif "interaction_terms" in interactions:
                    terms = interactions["interaction_terms"]
                    if isinstance(terms, dict) and "features" in terms:
                        # Parse strings like "feature1 * feature2"
                        string_interactions = terms["features"]
                        parsed_interactions = self._parse_string_interactions(string_interactions)
                        if parsed_interactions:
                            operations["feature_interactions"] = parsed_interactions
                            self.log(f"Parsed {len(parsed_interactions)} interactions from strings")
                
                # Try to extract from nested dict
                elif "interactions" in interactions:
                    operations["feature_interactions"] = interactions["interactions"]
                    self.log(f"Extracted interactions from nested dict")
        
        # Extract transformations from multiple possible locations
        if "transformations" in source:
            operations["transformations"] = source["transformations"]
            self.log(f"Extracted transformations")
        elif "numerical_transformations" in source:
            num_transforms = source["numerical_transformations"]
            if "log_transforms" in num_transforms and num_transforms["log_transforms"]:
                operations["transformations"] = [{
                    "features": num_transforms["log_transforms"],
                    "transform_type": "log1p",
                    "prefix": "log"
                }]
                self.log(f"Extracted log transforms from numerical_transformations")
        elif "numerical_feature_transformations" in source:
            # LLM might use this key
            num_transforms = source["numerical_feature_transformations"]
            self.log(f"Found numerical_feature_transformations with keys: {list(num_transforms.keys()) if isinstance(num_transforms, dict) else type(num_transforms)}")
            if isinstance(num_transforms, dict):
                # Handle "suggestions" format
                if "suggestions" in num_transforms:
                    suggestions = num_transforms["suggestions"]
                    if isinstance(suggestions, list):
                        parsed_transforms = self._parse_suggestion_transformations(suggestions)
                        if parsed_transforms:
                            operations["transformations"] = parsed_transforms
                            self.log(f"Parsed {len(parsed_transforms)} transformations from suggestions")
                # Try various extraction patterns
                elif "log_transforms" in num_transforms and num_transforms["log_transforms"]:
                    operations["transformations"] = [{
                        "features": num_transforms["log_transforms"],
                        "transform_type": "log1p",
                        "prefix": "log"
                    }]
                    self.log(f"Extracted log transforms")
                elif "log_transform" in num_transforms:
                    # Handle "log_transform" (singular) with nested structure
                    log_config = num_transforms["log_transform"]
                    if isinstance(log_config, dict) and "features" in log_config:
                        features = log_config["features"]
                        if features:
                            operations["transformations"] = [{
                                "features": features,
                                "transform_type": "log1p",
                                "prefix": "log"
                            }]
                            self.log(f"Extracted log transforms from log_transform: {len(features)} features")
                elif "transformations" in num_transforms:
                    operations["transformations"] = num_transforms["transformations"]
                    self.log(f"Extracted nested transformations")
        
        # Extract polynomial features
        if "polynomial_features" in source:
            operations["polynomial_features"] = source["polynomial_features"]
            self.log(f"Extracted polynomial features")
        
        # Extract binning
        if "binning" in source:
            operations["binning"] = source["binning"]
            self.log(f"Extracted binning operations")
        
        # Extract ratio features
        if "ratio_features" in source:
            operations["ratio_features"] = source["ratio_features"]
            self.log(f"Extracted ratio features")
        
        # Extract oncology features if present
        if "oncology_features" in source:
            operations["oncology_features"] = source["oncology_features"]
            self.log(f"Extracted oncology-specific features")
        
        # Check for "new_feature_creation" key (LLM might use this)
        if "new_feature_creation" in source:
            new_features = source["new_feature_creation"]
            self.log(f"Found new_feature_creation with keys: {list(new_features.keys()) if isinstance(new_features, dict) else type(new_features)}")
            if isinstance(new_features, dict):
                # Try to extract operations from here
                for key in ["feature_interactions", "transformations", "ratio_features", "polynomial_features"]:
                    if key in new_features and new_features[key]:
                        if key not in operations:  # Don't override if already extracted
                            operations[key] = new_features[key]
                            self.log(f"Extracted {key} from new_feature_creation")
        
        # If we got nothing useful, log it clearly
        if not operations:
            self.log("No operations extracted, using minimal oncology defaults for survival analysis")
            # Don't add defaults - let it run without feature creation
            # This way we can see if the LLM is being too conservative
        else:
            self.log(f"Successfully extracted operations: {list(operations.keys())}")
        
        return operations
    
    def _parse_string_interactions(self, string_interactions: List[str]) -> List[Dict[str, Any]]:
        """
        Parse LLM's string-format interactions like "feature1 * feature2" into structured format.
        
        Args:
            string_interactions: List of strings like ["age * lymph_nodes", "sex * smoking"]
            
        Returns:
            List of structured interaction dicts
        """
        parsed = []
        
        for interaction_str in string_interactions:
            # Parse strings like "feature1 * feature2"
            if '*' in interaction_str:
                parts = [p.strip() for p in interaction_str.split('*')]
                if len(parts) == 2:
                    # Create structured interaction
                    parsed.append({
                        "name": f"{parts[0]}_x_{parts[1]}".replace(' ', '_'),
                        "features": parts,
                        "operation": "multiply",
                        "feature_types": ["numerical", "numerical"]  # Assume numerical for *
                    })
                    self.log(f"Parsed interaction: {parts[0]} × {parts[1]}")
        
        return parsed
    
    def _parse_suggestion_interactions(self, suggestions: List[Dict], data_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Parse LLM's suggestion format for interactions.
        
        Format 1:
        [
          {"features": ["feature1", "feature2"], "interaction": "Multiplicative Interaction", ...},
          ...
        ]
        
        Format 2:
        [
          {"interaction": "feature1 * feature2", "description": "..."},
          ...
        ]
        
        Args:
            suggestions: List of interaction suggestions from LLM
            data_analysis: Data analysis dict containing feature type information
        """
        parsed = []
        
        # Extract feature type lists from data_analysis
        categorical_features = set(data_analysis.get("categorical_features", []))
        numerical_features = set(data_analysis.get("numerical_features", []))
        
        for suggestion in suggestions:
            # Format 1: Has "features" array
            if "features" in suggestion and isinstance(suggestion["features"], list):
                features = suggestion["features"]
                if len(features) >= 2:
                    feat1, feat2 = features[0], features[1]
                    
                    # Determine actual feature types from data analysis
                    feat1_is_cat = feat1 in categorical_features
                    feat2_is_cat = feat2 in categorical_features
                    feat1_is_num = feat1 in numerical_features
                    feat2_is_num = feat2 in numerical_features
                    
                    # Determine operation based on ACTUAL feature types
                    if feat1_is_cat and feat2_is_cat:
                        # Both categorical → concatenate
                        operation = "concat"
                        feature_types = ["categorical", "categorical"]
                    elif feat1_is_num and feat2_is_num:
                        # Both numerical → multiply
                        operation = "multiply"
                        feature_types = ["numerical", "numerical"]
                    elif (feat1_is_cat and feat2_is_num) or (feat1_is_num and feat2_is_cat):
                        # Mixed → check LLM's intention from keywords
                        interaction_type = suggestion.get("interaction", "").lower()
                        description = suggestion.get("description", "").lower()
                        technique = suggestion.get("technique", "").lower()
                        combined_text = f"{interaction_type} {description} {technique}"
                        
                        if any(kw in combined_text for kw in ["concat", "combine", "combined"]):
                            operation = "concat"
                            feature_types = ["categorical", "categorical"]
                        else:
                            operation = "multiply"
                            feature_types = ["numerical", "numerical"]
                    else:
                        # Unknown types, check LLM keywords
                        interaction_type = suggestion.get("interaction", "").lower()
                        description = suggestion.get("description", "").lower()
                        technique = suggestion.get("technique", "").lower()
                        combined_text = f"{interaction_type} {description} {technique}"
                        
                        # Check for operation keywords
                        if any(kw in combined_text for kw in ["concat", "combine", "combined", "categorical"]):
                            operation = "concat"
                            feature_types = ["categorical", "categorical"]
                        else:
                            operation = "multiply"
                            feature_types = ["numerical", "numerical"]
                    
                    parsed.append({
                        "name": f"{'_x_'.join([feat1, feat2])}".replace(' ', '_'),
                        "features": [feat1, feat2],
                        "operation": operation,
                        "feature_types": feature_types
                    })
                    self.log(f"Parsed suggestion interaction ({operation}): {feat1} × {feat2} "
                            f"[{feat1}: {'cat' if feat1_is_cat else 'num' if feat1_is_num else '?'}, "
                            f"{feat2}: {'cat' if feat2_is_cat else 'num' if feat2_is_num else '?'}]")
            
            # Format 2: Has "interaction" as string with "*"
            elif "interaction" in suggestion and isinstance(suggestion["interaction"], str):
                interaction_str = suggestion["interaction"]
                if '*' in interaction_str:
                    parts = [p.strip() for p in interaction_str.split('*')]
                    if len(parts) == 2:
                        feat1, feat2 = parts[0], parts[1]
                        
                        # Check feature types
                        feat1_is_cat = feat1 in categorical_features
                        feat2_is_cat = feat2 in categorical_features
                        feat1_is_num = feat1 in numerical_features
                        feat2_is_num = feat2 in numerical_features
                        
                        # Determine operation
                        if feat1_is_cat and feat2_is_cat:
                            operation = "concat"
                            feature_types = ["categorical", "categorical"]
                        elif feat1_is_num and feat2_is_num:
                            operation = "multiply"
                            feature_types = ["numerical", "numerical"]
                        else:
                            # Default to multiply for "*" operator
                            operation = "multiply"
                            feature_types = ["numerical", "numerical"]
                        
                        parsed.append({
                            "name": f"{feat1}_x_{feat2}".replace(' ', '_'),
                            "features": [feat1, feat2],
                            "operation": operation,
                            "feature_types": feature_types
                        })
                        self.log(f"Parsed suggestion interaction (string format, {operation}): {feat1} × {feat2}")
        
        return parsed
    
    def _parse_suggestion_transformations(self, suggestions: List[Dict]) -> List[Dict[str, Any]]:
        """
        Parse LLM's suggestion format for transformations.
        
        Format:
        [
          {"feature": "feature1", "transformation": "Log Transformation", ...},
          OR
          {"feature": "feature1", "technique": "Log Transformation", ...},
          ...
        ]
        """
        parsed_transforms = []
        log_features = []
        sqrt_features = []
        
        for suggestion in suggestions:
            if "feature" in suggestion:
                # Check for both "transformation" and "technique" keys (LLM might use either)
                transform_key = None
                if "transformation" in suggestion:
                    transform_key = "transformation"
                elif "technique" in suggestion:
                    transform_key = "technique"
                
                if transform_key:
                    feature = suggestion["feature"]
                    transform = suggestion[transform_key].lower()
                    
                    if "log" in transform:
                        log_features.append(feature)
                        self.log(f"  → Found log transform for: {feature}")
                    elif "sqrt" in transform or "square root" in transform:
                        sqrt_features.append(feature)
                        self.log(f"  → Found sqrt transform for: {feature}")
                    elif "binning" in transform:
                        # Note: binning needs to be handled separately with bin specifications
                        self.log(f"  → Found binning suggestion for {feature} (needs separate handling)")
        
        # Group all log transforms together
        if log_features:
            parsed_transforms.append({
                "features": log_features,
                "transform_type": "log1p",
                "prefix": "log"
            })
            self.log(f"Parsed {len(log_features)} log transforms from suggestions")
        
        # Group all sqrt transforms together
        if sqrt_features:
            parsed_transforms.append({
                "features": sqrt_features,
                "transform_type": "sqrt",
                "prefix": "sqrt"
            })
            self.log(f"Parsed {len(sqrt_features)} sqrt transforms from suggestions")
        
        return parsed_transforms
    
    async def _engineer_features(
        self, 
        df: pd.DataFrame, 
        data_analysis: Dict[str, Any], 
        llm_recommendations: Dict[str, Any],
        target_variable: str,
        time_variable: Optional[str] = None,
        df_test: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """Perform the actual feature engineering"""
        
        self.log("Starting feature engineering...")
        
        # Check if pre-split test set is provided
        use_presplit_test = df_test is not None
        if use_presplit_test:
            self.log(f"Using pre-split test set with {len(df_test)} samples")
        
        # Get feature lists
        feature_columns = data_analysis.get("suggested_features", [])
        categorical_features = data_analysis.get("categorical_features", [])
        numerical_features = data_analysis.get("numerical_features", [])
        task_type = data_analysis.get("task_type")
        
        # === DIAGNOSTIC: Show feature cardinality ===
        self.log("=" * 80)
        self.log("FEATURE CARDINALITY ANALYSIS")
        self.log("=" * 80)
        self.log(f"Total features from analyzer: {len(feature_columns)}")
        self.log(f"  - Categorical: {len(categorical_features)}")
        self.log(f"  - Numerical: {len(numerical_features)}")
        
        # Check if identifier columns are present
        identifier_patterns = ['id', 'patient', 'sample', 'subject', 'record']
        potential_identifiers = [col for col in feature_columns 
                                if any(pattern in col.lower() for pattern in identifier_patterns)]
        if potential_identifiers:
            self.log(f"  ⚠️  POTENTIAL IDENTIFIERS FOUND: {potential_identifiers}", "WARNING")
            self.log("  These should have been excluded by data_analyzer!", "WARNING")
        
        # Check for high-cardinality features that might cause explosion
        high_cardinality_features = []
        for col in categorical_features:
            if col in df.columns:
                # Handle duplicate column names
                col_data = df[col]
                if isinstance(col_data, pd.DataFrame):
                    col_data = col_data.iloc[:, 0]
                
                n_unique = col_data.nunique()
                if hasattr(n_unique, 'item'):
                    n_unique = n_unique.item()
                
                if n_unique > 50:
                    high_cardinality_features.append((col, n_unique))
                    self.log(f"  ⚠️  HIGH CARDINALITY: '{col}' has {n_unique} unique values", "WARNING")
        
        if high_cardinality_features:
            self.log(f"Found {len(high_cardinality_features)} high-cardinality features (>50 unique values)", "WARNING")
            self.log("These will cause feature explosion if one-hot encoded!", "WARNING")
        self.log("=" * 80)
        
        # Check if we should preserve CV column for preset cross-validation
        cv_groups = None
        if self.config.data.use_preset_CV and 'CV' in df.columns:
            self.log("Preserving CV column for preset cross-validation groups")
            cv_groups = df['CV'].copy()
            # Explicitly exclude CV from feature columns
            feature_columns = [col for col in feature_columns if col != 'CV']
            categorical_features = [col for col in categorical_features if col != 'CV']
            numerical_features = [col for col in numerical_features if col != 'CV']
        elif self.config.data.use_preset_CV:
            self.log("Warning: use_preset_CV is True but 'CV' column not found in dataset", "WARNING")
        
        # For survival analysis, also need time variable
        columns_to_keep = feature_columns + [target_variable]
        if task_type == "survival" and time_variable:
            columns_to_keep.append(time_variable)
        
        available_columns = [col for col in columns_to_keep if col in df.columns]
        
        self.log(f"Working with {len(available_columns)} columns (including target)")
        
        # Create working dataframe
        df_work = df[available_columns].copy()
        
        # === NEW: Feature Creation Step ===
        # Apply structured feature engineering operations BEFORE separating target
        feature_creator_ops = llm_recommendations.get("feature_creator_operations", {})
        if feature_creator_ops and any(feature_creator_ops.values()):
            self.log("Applying feature creation operations...")
            self.feature_creator = FeatureCreator(feature_creator_ops, domain="oncology")
            
            # Extract only features (not target/time) for transformation
            features_only = [col for col in df_work.columns if col != target_variable]
            if task_type == "survival" and time_variable:
                features_only = [col for col in features_only if col != time_variable]
            
            # Create dataframe with just features for transformation
            df_features = df_work[features_only].copy()
            df_features_transformed = self.feature_creator.fit_transform(df_features)
            
            # Add back target and time variables
            # Use pandas concat to avoid issues with duplicate column names
            cols_to_add = [col for col in df_work.columns if col not in df_features_transformed.columns]
            if cols_to_add:
                df_features_transformed = pd.concat([
                    df_features_transformed,
                    df_work[cols_to_add]
                ], axis=1)
            
            df_work = df_features_transformed
            
            # Update feature lists with newly created features
            created_features = self.feature_creator.get_created_features()
            self.log(f"Created {len(created_features)} new features")
            
            # Update categorical/numerical feature lists
            for new_feat in created_features:
                if new_feat in df_work.columns:
                    if df_work[new_feat].dtype in ['object', 'category']:
                        categorical_features.append(new_feat)
                    else:
                        numerical_features.append(new_feat)
            
            # Apply same transformations to test set if provided
            if use_presplit_test:
                df_test_features = df_test[[col for col in features_only if col in df_test.columns]].copy()
                df_test_features_transformed = self.feature_creator.transform(df_test_features)
                
                # Add back target and time
                cols_to_add_test = [col for col in df_test.columns if col not in df_test_features_transformed.columns]
                if cols_to_add_test:
                    df_test_features_transformed = pd.concat([
                        df_test_features_transformed,
                        df_test[cols_to_add_test]
                    ], axis=1)
                
                df_test = df_test_features_transformed
        
        # Update feature_columns list to include newly created features
        feature_columns = [col for col in df_work.columns if col not in [target_variable, time_variable]]
        
        # Update available_columns to include newly created features for test set
        columns_to_keep = feature_columns + [target_variable]
        if task_type == "survival" and time_variable:
            columns_to_keep.append(time_variable)
        available_columns = columns_to_keep
        
        # Separate features and target
        if target_variable in df_work.columns:
            if task_type == "survival" and time_variable and time_variable in df_work.columns:
                self.log(f"Extracting survival variables: target='{target_variable}', time='{time_variable}'")
                self.log(f"time_variable type: {type(time_variable)}, value: {time_variable}")
                
                X = df_work.drop(columns=[target_variable, time_variable])
                y_event = df_work[target_variable]
                
                # Ensure we get a Series for time_variable
                if isinstance(time_variable, str):
                    y_time = df_work[time_variable]
                    self.log(f"Extracted y_time: type={type(y_time)}, shape={y_time.shape if hasattr(y_time, 'shape') else 'N/A'}")
                    
                    # Handle duplicate column names - pandas returns DataFrame if column name is duplicated
                    if isinstance(y_time, pd.DataFrame):
                        self.log(f"WARNING: Multiple columns with name '{time_variable}', using first column", "WARNING")
                        y_time = y_time.iloc[:, 0]
                else:
                    self.log(f"WARNING: time_variable is not a string: {time_variable}", "WARNING")
                    y_time = df_work[time_variable]
                    if isinstance(y_time, pd.DataFrame):
                        y_time = y_time.iloc[:, 0]
            else:
                X = df_work.drop(columns=[target_variable])
                y_event = df_work[target_variable]
                y_time = None
        else:
            raise ValueError(f"Target variable '{target_variable}' not found in dataset")
        
        # Identify actual categorical and numerical columns in our working set
        actual_categorical = [col for col in categorical_features if col in X.columns]
        actual_numerical = [col for col in numerical_features if col in X.columns]
        
        # Handle missing columns
        remaining_cols = [col for col in X.columns if col not in actual_categorical + actual_numerical]
        if remaining_cols:
            # Infer types for remaining columns
            for col in remaining_cols:
                if X[col].dtype in ['object', 'category']:
                    actual_categorical.append(col)
                else:
                    actual_numerical.append(col)
        
        self.log(f"Categorical features: {len(actual_categorical)}")
        self.log(f"Numerical features: {len(actual_numerical)}")
        
        # Convert categorical columns to strings to handle mixed types
        for col in actual_categorical:
            X[col] = X[col].astype(str)
        
        # === NEW: Feature Selection Step ===
        # Apply feature selection to remove redundant/uninformative features
        feature_selection_config = llm_recommendations.get("feature_selection", {})
        if feature_selection_config.get("enabled", True):
            self.log("Performing feature selection...")
            
            all_features = actual_categorical + actual_numerical
            
            # Initialize feature selector
            self.feature_selector = FeatureSelector(
                missing_threshold=feature_selection_config.get("missing_threshold", 0.5),
                variance_threshold=feature_selection_config.get("variance_threshold", 0.01),
                correlation_threshold=feature_selection_config.get("correlation_threshold", 0.95),
                max_features=feature_selection_config.get("max_features", None)
            )
            
            # Perform selection on training data (X includes features before split)
            # We need y_event for selection, so we'll do selection here before splitting
            selected_features, selection_report = self.feature_selector.fit_select(
                X, y_event, task_type, all_features
            )
            
            # Update feature lists
            actual_categorical = [f for f in actual_categorical if f in selected_features]
            actual_numerical = [f for f in actual_numerical if f in selected_features]
            
            # Filter X to only selected features
            X = X[selected_features]
            
            self.log(f"Feature selection complete: {len(selected_features)} features selected "
                    f"(removed {selection_report['n_features_removed']})")
        
        # Create preprocessing pipeline (pass X for cardinality analysis)
        # This returns the preprocessor AND the list of features that were actually kept
        preprocessor, kept_categorical = self._create_preprocessing_pipeline(
            actual_categorical, actual_numerical, llm_recommendations, df=X
        )
        
        # Update actual_categorical to only include features that weren't dropped
        actual_categorical = kept_categorical
        
        # Handle target variable and splitting based on task type
        if use_presplit_test:
            # Use pre-split test set
            self.log("Using pre-split test set - training data will be split into train/val only")
            
            # Process test set features
            X_test = df_test[available_columns].copy()
            if target_variable in X_test.columns:
                if task_type == "survival" and time_variable and time_variable in X_test.columns:
                    X_test = X_test.drop(columns=[target_variable, time_variable])
                    y_test_event = df_test[target_variable]
                    y_test_time = df_test[time_variable]
                    if isinstance(y_test_time, pd.DataFrame):
                        y_test_time = y_test_time.iloc[:, 0]
                else:
                    X_test = X_test.drop(columns=[target_variable])
                    y_test_event = df_test[target_variable]
                    y_test_time = None
            else:
                raise ValueError(f"Target variable '{target_variable}' not found in test dataset")
            
            # Convert categorical columns in test set
            for col in actual_categorical:
                if col in X_test.columns:
                    X_test[col] = X_test[col].astype(str)
            
            # Apply same feature selection to test set
            if self.feature_selector is not None:
                selected_features = self.feature_selector.get_selected_features()
                X_test = X_test[[col for col in selected_features if col in X_test.columns]]
            
            # Process targets and split training data only
            if task_type == "survival":
                self.log("Processing survival analysis task...")
                self.log(f"y_event type: {type(y_event)}, shape: {y_event.shape if hasattr(y_event, 'shape') else 'N/A'}")
                self.log(f"y_time type: {type(y_time)}, shape: {y_time.shape if hasattr(y_time, 'shape') else 'N/A'}")
                
                # Process survival target for training data
                y_processed, target_encoder = self._process_survival_target(
                    y_event, y_time, X
                )
                self.log(f"Survival target processed. y_processed shape: {y_processed.shape}")
                
                # Process test set survival target
                y_test = self._process_survival_target(y_test_event, y_test_time, X_test)[0]
                
                # Use all training data for CV (no validation split) in two cases:
                # 1. Separate test set provided (df_test is not None)
                # 2. Preset CV groups provided (cv_groups is not None)
                # This maximizes training samples while still validating via CV
                if cv_groups is not None:
                    self.log("Using preset CV groups - no validation split, will use all training data with preset CV")
                    X_train, y_train = X, y_processed
                    X_val, y_val = None, None
                    cv_groups_train = cv_groups.values if hasattr(cv_groups, 'values') else cv_groups
                    cv_groups_val = None
                else:
                    # Since separate test set is provided, use all training data for CV
                    self.log(f"Separate test set provided ({len(X_test)} samples) - using all {len(X)} training samples with CV (no validation split)")
                    X_train, y_train = X, y_processed
                    X_val, y_val = None, None
                    cv_groups_train = None
                    cv_groups_val = None
            else:
                # Standard processing for classification/regression
                y_processed, target_encoder = self._process_target(y_event, task_type)
                
                # Process test set target
                y_test, _ = self._process_target(y_test_event, task_type)
                
                # Use all training data for CV (no validation split) in two cases:
                # 1. Separate test set provided (df_test is not None)
                # 2. Preset CV groups provided (cv_groups is not None)
                # This maximizes training samples while still validating via CV
                if cv_groups is not None:
                    self.log("Using preset CV groups - no validation split, will use all training data with preset CV")
                    X_train, y_train = X, y_processed
                    X_val, y_val = None, None
                    cv_groups_train = cv_groups.values if hasattr(cv_groups, 'values') else cv_groups
                    cv_groups_val = None
                else:
                    # Since separate test set is provided, use all training data for CV
                    self.log(f"Separate test set provided ({len(X_test)} samples) - using all {len(X)} training samples with CV (no validation split)")
                    X_train, y_train = X, y_processed
                    X_val, y_val = None, None
                    cv_groups_train = None
                    cv_groups_val = None
        else:
            # Auto-split from main dataset (current behavior)
            if task_type == "survival":
                self.log("Processing survival analysis task...")
                self.log(f"y_event type: {type(y_event)}, shape: {y_event.shape if hasattr(y_event, 'shape') else 'N/A'}")
                self.log(f"y_time type: {type(y_time)}, shape: {y_time.shape if hasattr(y_time, 'shape') else 'N/A'}")
                
                # Process survival target
                y_processed, target_encoder = self._process_survival_target(
                    y_event, y_time, X
                )
                self.log(f"Survival target processed. y_processed shape: {y_processed.shape}")
                
                # Risk-stratified splitting for survival analysis
                X_train, X_test, y_train, y_test = self._risk_stratified_split(
                    X, y_processed, y_event, y_time,
                    test_size=self.config.data.test_size,
                    random_state=self.config.data.random_state
                )
                
                # For validation split, we need to extract the training indices first
                # Then use those to get the corresponding event/time from original data
                train_indices = X_train.index
                y_event_train = y_event.loc[train_indices]
                y_time_train = y_time.loc[train_indices]
                
                # Validation split (also risk-stratified)
                X_train, X_val, y_train, y_val = self._risk_stratified_split(
                    X_train, y_train, 
                    y_event_train, y_time_train,
                    test_size=self.config.data.val_size / (1 - self.config.data.test_size),
                    random_state=self.config.data.random_state
                )
                
                # Split CV groups if present
                cv_groups_train = None
                cv_groups_val = None
                if cv_groups is not None:
                    train_idx = X_train.index if hasattr(X_train, 'index') else np.arange(len(X_train))
                    val_idx = X_val.index if hasattr(X_val, 'index') else np.arange(len(X_val))
                    cv_groups_train = cv_groups.loc[train_idx].values if hasattr(cv_groups, 'loc') else cv_groups[train_idx]
                    cv_groups_val = cv_groups.loc[val_idx].values if hasattr(cv_groups, 'loc') else cv_groups[val_idx]
            else:
                # Standard processing for classification/regression
                y_processed, target_encoder = self._process_target(y_event, task_type)
                
                # Standard split
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y_processed, 
                    test_size=self.config.data.test_size,
                    random_state=self.config.data.random_state,
                    stratify=y_processed if task_type == "classification" else None
                )
                
                # Split training into train and validation
                X_train, X_val, y_train, y_val = train_test_split(
                    X_train, y_train,
                    test_size=self.config.data.val_size / (1 - self.config.data.test_size),
                    random_state=self.config.data.random_state,
                    stratify=y_train if task_type == "classification" else None
                )
                
                # Split CV groups if present
                cv_groups_train = None
                cv_groups_val = None
                if cv_groups is not None:
                    train_idx = X_train.index if hasattr(X_train, 'index') else np.arange(len(X_train))
                    val_idx = X_val.index if hasattr(X_val, 'index') else np.arange(len(X_val))
                    cv_groups_train = cv_groups.loc[train_idx].values if hasattr(cv_groups, 'loc') else cv_groups[train_idx]
                    cv_groups_val = cv_groups.loc[val_idx].values if hasattr(cv_groups, 'loc') else cv_groups[val_idx]
        
        # Save engineered dataframes before preprocessing (encoding/scaling)
        # For survival tasks, pass the original time variables if available
        train_time = y_time if (task_type == "survival" and 'y_time' in locals()) else None
        test_time = y_test_time if (task_type == "survival" and use_presplit_test and 'y_test_time' in locals()) else None
        self._save_engineered_data(X_train, y_train, X_test, y_test, task_type, time_variable, 
                                   train_time, test_time)
        
        # Fit preprocessor on training data
        self.log("Fitting preprocessing pipeline...")
        X_train_processed = preprocessor.fit_transform(X_train)
        
        # Transform validation set if it exists (not needed when using preset CV)
        if X_val is not None:
            X_val_processed = preprocessor.transform(X_val)
        else:
            X_val_processed = None
            
        X_test_processed = preprocessor.transform(X_test)
        
        # Get feature names after preprocessing
        feature_names = self._get_feature_names(preprocessor, actual_categorical, actual_numerical)
        
        # Store preprocessor and feature names
        self.preprocessor = preprocessor
        self.feature_names = feature_names
        
        # Create result
        result = {
            "n_features": X_train_processed.shape[1],
            "n_samples_train": X_train_processed.shape[0],
            "n_samples_val": X_val_processed.shape[0] if X_val_processed is not None else 0,
            "n_samples_test": X_test_processed.shape[0],
            "feature_names": feature_names,
            "categorical_features": actual_categorical,
            "numerical_features": actual_numerical,
            "target_variable": target_variable,
            "task_type": data_analysis.get("task_type"),
            "data_splits": {
                "X_train": X_train_processed,
                "X_val": X_val_processed,
                "X_test": X_test_processed,
                "y_train": y_train,
                "y_val": y_val,
                "y_test": y_test,
                "cv_groups": cv_groups_train  # Preset CV groups for cross-validation (if available)
            },
            "preprocessor": preprocessor,
            "target_encoder": target_encoder,
            "original_features": feature_columns,
            "preprocessing_steps": self._get_preprocessing_summary(llm_recommendations),
            "data_quality_report": self._generate_quality_report(X_train, y_train),
            "llm_recommendations": llm_recommendations,
            # NEW: Feature engineering reports
            "feature_creation_report": self.feature_creator.get_feature_report() if self.feature_creator else None,
            "feature_selection_report": self.feature_selector._generate_selection_report(feature_columns) if self.feature_selector else None
        }
        
        # Track this iteration for refinement
        self.feature_engineering_history.append({
            "config": llm_recommendations,
            "n_features_created": len(self.feature_creator.get_created_features()) if self.feature_creator else 0,
            "n_features_selected": len(self.feature_selector.get_selected_features()) if self.feature_selector else len(feature_columns),
            "final_n_features": result["n_features"]
        })
        
        self.log(f"Feature engineering completed: {result['n_features']} features, {result['n_samples_train']} training samples")
        
        return result
    
    def _create_preprocessing_pipeline(
        self, 
        categorical_features: List[str], 
        numerical_features: List[str],
        llm_recommendations: Dict[str, Any],
        df: pd.DataFrame = None,
        cardinality_threshold: int = 15
    ) -> Tuple[ColumnTransformer, List[str]]:
        """Create preprocessing pipeline based on LLM recommendations
        
        Args:
            categorical_features: All categorical features
            numerical_features: All numerical features
            llm_recommendations: LLM recommendations for preprocessing
            df: DataFrame to analyze feature cardinality (optional)
            cardinality_threshold: Max unique values for one-hot encoding (default: 15)
            
        Returns:
            Tuple of (preprocessor, kept_categorical_features)
        """
        
        # Get recommendations
        num_strategy = llm_recommendations.get("numerical_transformations", {})
        cat_strategy = llm_recommendations.get("categorical_encoding", {})
        
        # Numerical pipeline
        num_imputer_strategy = num_strategy.get("imputation_strategy", "median")
        scaling_strategy = num_strategy.get("scaling_strategy", "standard")
        
        if scaling_strategy == "standard":
            scaler = StandardScaler()
        elif scaling_strategy == "minmax":
            scaler = MinMaxScaler()
        elif scaling_strategy == "robust":
            scaler = RobustScaler()
        else:
            scaler = StandardScaler()
        
        numerical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy=num_imputer_strategy)),
            ('scaler', scaler)
        ])
        
        # Split categorical features by cardinality if DataFrame is provided
        # Drop high-cardinality features (>threshold) to avoid feature explosion and arbitrary encoding
        low_cardinality_features = []
        dropped_high_cardinality = []
        
        if df is not None:
            for col in categorical_features:
                if col in df.columns:
                    # Handle duplicate column names (df[col] returns DataFrame if duplicates exist)
                    col_data = df[col]
                    if isinstance(col_data, pd.DataFrame):
                        # Multiple columns with same name - use first one
                        col_data = col_data.iloc[:, 0]
                    
                    n_unique = col_data.nunique()
                    
                    # Ensure n_unique is a scalar (sometimes returns Series)
                    if hasattr(n_unique, 'item'):
                        n_unique = n_unique.item()
                    
                    if n_unique <= cardinality_threshold:
                        low_cardinality_features.append(col)
                    else:
                        # Drop high-cardinality features instead of encoding them
                        dropped_high_cardinality.append((col, n_unique))
                else:
                    # If column not in df, default to low cardinality
                    low_cardinality_features.append(col)
            
            if dropped_high_cardinality:
                self.log(f"📊 Cardinality-based feature filtering:")
                self.log(f"  - Low cardinality (≤{cardinality_threshold}, one-hot): {len(low_cardinality_features)} features")
                self.log(f"  - High cardinality (>{cardinality_threshold}, DROPPED): {len(dropped_high_cardinality)} features")
                self.log(f"    Dropped features: {[f'{name} ({n} unique)' for name, n in dropped_high_cardinality[:5]]}")
                if len(dropped_high_cardinality) > 5:
                    self.log(f"    ... and {len(dropped_high_cardinality) - 5} more")
        else:
            # If no DataFrame provided, use all as low cardinality (backwards compatibility)
            low_cardinality_features = categorical_features
        
        # Categorical pipeline (only for low cardinality features)
        cat_imputer_strategy = cat_strategy.get("imputation_strategy", "most_frequent")
        
        # Pipeline for low cardinality features (one-hot encoding)
        # High cardinality features are dropped, not encoded
        categorical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy=cat_imputer_strategy)),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        # Combine pipelines based on what features we have
        transformers = [
            ('num', numerical_pipeline, numerical_features)
        ]
        
        if low_cardinality_features:
            transformers.append(('cat', categorical_pipeline, low_cardinality_features))
        
        preprocessor = ColumnTransformer(transformers)
        
        # Return both preprocessor and the list of categorical features that were kept
        return preprocessor, low_cardinality_features
    
    def _process_target(self, y: pd.Series, task_type: str) -> Tuple[np.ndarray, Optional[LabelEncoder]]:
        """Process target variable"""
        
        if task_type == "classification":
            if y.dtype == 'object' or y.dtype.name == 'category':
                # Encode categorical target
                encoder = LabelEncoder()
                y_processed = encoder.fit_transform(y.fillna('missing'))
                self.log(f"Target encoded: {list(encoder.classes_)}")
                return y_processed, encoder
            else:
                # Already numerical
                return y.fillna(-1).values, None
        else:
            # Regression - ensure numerical
            return y.fillna(y.median()).values, None
    
    def _process_survival_target(
        self, 
        y_event: pd.Series, 
        y_time: pd.Series,
        X: pd.DataFrame
    ) -> Tuple[np.ndarray, None]:
        """
        Process survival target into structured array format required by scikit-survival
        
        Args:
            y_event: Event indicator (1=event occurred, 0=censored)
            y_time: Time to event or censoring
            X: Feature dataframe (for length validation)
            
        Returns:
            Structured array with dtype [('event', bool), ('time', float)]
        """
        
        # Ensure we're working with Series (not arrays or other types)
        if not isinstance(y_event, pd.Series):
            self.log(f"Converting y_event to Series (was {type(y_event)})")
            if isinstance(y_event, pd.DataFrame):
                y_event = y_event.iloc[:, 0]  # Take first column
            else:
                y_event = pd.Series(y_event, index=X.index)
        
        if not isinstance(y_time, pd.Series):
            self.log(f"Converting y_time to Series (was {type(y_time)})")
            if isinstance(y_time, pd.DataFrame):
                self.log(f"y_time is a DataFrame with columns: {y_time.columns.tolist()}")
                y_time = y_time.iloc[:, 0]  # Take first column
            else:
                y_time = pd.Series(y_time, index=X.index)
        
        self.log(f"Processing survival target: {len(y_event)} samples")
        self.log(f"y_event shape: {y_event.shape}, y_time shape: {y_time.shape}")
        
        # Calculate median as a scalar
        time_median = float(y_time.median())
        self.log(f"Time median: {time_median}")
        
        # Convert event to boolean numpy array
        event = y_event.fillna(0).astype(bool).values
        
        # Convert time to float numpy array, fill missing with median
        time = y_time.fillna(time_median).astype(float).values
        
        # Ensure positive times
        n_nonpositive = np.sum(time <= 0)
        if n_nonpositive > 0:
            self.log(f"WARNING: Found {n_nonpositive} non-positive survival times (≤0). These are invalid for survival analysis.", "WARNING")
            self.log(f"  Min time before fix: {np.min(time):.3f}, Max: {np.max(time):.3f}", "WARNING")
            self.log(f"  Setting non-positive times to 0.001 (very small positive value)", "WARNING")
            time = np.maximum(time, 0.001)
            self.log(f"  Min time after fix: {np.min(time):.3f}", "WARNING")
        
        # Create structured array for scikit-survival
        # Since event and time are 1D numpy arrays, we can create this directly
        y_survival = np.zeros(len(event), dtype=[('event', bool), ('time', float)])
        y_survival['event'] = event
        y_survival['time'] = time
        
        self.log(f"Survival target processed: {np.sum(event)} events, {len(event) - np.sum(event)} censored")
        self.log(f"Median survival time: {np.median(time):.2f}, Max: {np.max(time):.2f}")
        
        return y_survival, None
    
    def _risk_stratified_split(
        self,
        X: pd.DataFrame,
        y_survival: np.ndarray,
        y_event: pd.Series,
        y_time: pd.Series,
        test_size: float,
        random_state: int
    ) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
        """
        Perform risk-stratified train/test split for survival data.
        
        This ensures balanced distribution of:
        1. Event groups (event vs. censored)
        2. Risk bins (discrete risk stratification)
        
        As per the CHIMERA dataset methodology:
        - Splitting within each event group
        - Splitting within each discrete risk bin
        - Maintains balanced risk distribution across splits
        
        Args:
            X: Features
            y_survival: Structured survival array
            y_event: Event indicators
            y_time: Time to event
            test_size: Proportion for test set
            random_state: Random seed
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        
        self.log(f"Starting risk-stratified split (test_size={test_size})...")
        self.log(f"Input types - y_event: {type(y_event)}, y_time: {type(y_time)}")
        
        # Ensure y_time and y_event are pandas Series
        if not isinstance(y_time, pd.Series):
            self.log(f"Converting y_time to Series (was {type(y_time)})")
            if isinstance(y_time, pd.DataFrame):
                self.log(f"y_time is a DataFrame in risk split, using first column")
                y_time = y_time.iloc[:, 0]
            else:
                y_time = pd.Series(y_time, index=X.index)
        
        if not isinstance(y_event, pd.Series):
            self.log(f"Converting y_event to Series (was {type(y_event)})")
            if isinstance(y_event, pd.DataFrame):
                self.log(f"y_event is a DataFrame in risk split, using first column")
                y_event = y_event.iloc[:, 0]
            else:
                y_event = pd.Series(y_event, index=X.index)
        
        # Create risk bins based on survival time (quintiles)
        # This creates discrete risk strata
        n_bins = min(5, len(y_time.dropna().unique()))  # Use up to 5 bins, or fewer if limited unique times
        
        try:
            # Use only non-NaN values for binning
            risk_bins = pd.qcut(y_time, q=n_bins, labels=False, duplicates='drop')
        except (ValueError, TypeError) as e:
            # If qcut fails (e.g., too few unique values), use equal-width binning
            try:
                risk_bins = pd.cut(y_time, bins=n_bins, labels=False)
            except (ValueError, TypeError):
                # If all else fails, just use a single bin
                risk_bins = pd.Series(0, index=y_time.index)
        
        # Handle NaN values in risk_bins (fill with a separate category)
        risk_bins = risk_bins.fillna(-1).astype(int)
        
        # Ensure y_event is also clean (no NaNs)
        y_event_clean = y_event.fillna(0).astype(int)
        
        # Create stratification key: combination of event status and risk bin
        # This ensures both event groups and risk levels are balanced
        stratify_key = pd.Series([
            f"event_{e}_risk_{r}" 
            for e, r in zip(y_event_clean.values, risk_bins.values)
        ], index=X.index)
        
        # Count samples per stratum
        stratum_counts = stratify_key.value_counts()
        min_samples = stratum_counts.min()
        
        self.log(f"Risk stratification: {n_bins} risk bins, {len(stratum_counts)} strata")
        self.log(f"Minimum samples per stratum: {min_samples}")
        
        # If some strata have too few samples, fall back to event-only stratification
        if min_samples < 2:
            self.log("Warning: Some risk strata have too few samples, using event-only stratification", "WARNING")
            stratify_key = y_event_clean.astype(str)
        
        try:
            # Perform stratified split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_survival,
                test_size=test_size,
                random_state=random_state,
                stratify=stratify_key
            )
            
            # Log split statistics
            train_events = np.sum(y_train['event'])
            test_events = np.sum(y_test['event'])
            
            self.log(f"Train set: {len(y_train)} samples ({train_events} events, "
                    f"{len(y_train) - train_events} censored)")
            self.log(f"Test set: {len(y_test)} samples ({test_events} events, "
                    f"{len(y_test) - test_events} censored)")
            
            return X_train, X_test, y_train, y_test
            
        except ValueError as e:
            # If stratification fails, do simple random split
            self.log(f"Risk stratification failed: {str(e)}, using simple random split", "WARNING")
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_survival,
                test_size=test_size,
                random_state=random_state
            )
            
            return X_train, X_test, y_train, y_test
    
    def _get_feature_names(
        self, 
        preprocessor: ColumnTransformer, 
        categorical_features: List[str], 
        numerical_features: List[str]
    ) -> List[str]:
        """Get feature names after preprocessing"""
        
        feature_names = []
        
        # Numerical features keep their names
        feature_names.extend(numerical_features)
        
        # Handle categorical features - now we may have two transformers: cat_low and cat_high
        try:
            # Try old format first (backwards compatibility)
            if 'cat' in preprocessor.named_transformers_:
                cat_transformer = preprocessor.named_transformers_['cat']
                if hasattr(cat_transformer, 'named_steps') and 'encoder' in cat_transformer.named_steps:
                    encoder = cat_transformer.named_steps['encoder']
                    if hasattr(encoder, 'get_feature_names_out'):
                        cat_names = encoder.get_feature_names_out(categorical_features)
                        feature_names.extend(cat_names)
                    else:
                        # Fallback for older sklearn versions
                        for cat_feature in categorical_features:
                            if hasattr(encoder, 'categories_'):
                                for cat in encoder.categories_[categorical_features.index(cat_feature)]:
                                    feature_names.append(f"{cat_feature}_{cat}")
                            else:
                                feature_names.append(cat_feature)
                else:
                    feature_names.extend(categorical_features)
            else:
                # New format with separate low/high cardinality transformers
                # Handle low cardinality (one-hot encoded)
                if 'cat_low' in preprocessor.named_transformers_:
                    cat_low_transformer = preprocessor.named_transformers_['cat_low']
                    if hasattr(cat_low_transformer, 'named_steps') and 'encoder' in cat_low_transformer.named_steps:
                        encoder = cat_low_transformer.named_steps['encoder']
                        # Get the actual features this transformer was fitted on
                        low_card_features = preprocessor.transformers_[
                            [i for i, (name, _, _) in enumerate(preprocessor.transformers_) if name == 'cat_low'][0]
                        ][2]
                        if hasattr(encoder, 'get_feature_names_out'):
                            cat_names = encoder.get_feature_names_out(low_card_features)
                            feature_names.extend(cat_names)
                        else:
                            feature_names.extend(low_card_features)
                
                # Handle high cardinality (label/ordinal encoded - keeps original names)
                if 'cat_high' in preprocessor.named_transformers_:
                    # Ordinal encoding keeps original feature names (one column per feature)
                    high_card_features = preprocessor.transformers_[
                        [i for i, (name, _, _) in enumerate(preprocessor.transformers_) if name == 'cat_high'][0]
                    ][2]
                    feature_names.extend(high_card_features)
        except Exception as e:
            self.log(f"Could not extract categorical feature names: {e}", "WARNING")
            # Fallback: just use original names
            feature_names.extend(categorical_features)
        
        return feature_names
    
    def _get_preprocessing_summary(self, llm_recommendations: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of preprocessing steps"""
        return {
            "numerical_preprocessing": {
                "imputation": llm_recommendations.get("numerical_transformations", {}).get("imputation_strategy", "median"),
                "scaling": llm_recommendations.get("numerical_transformations", {}).get("scaling_strategy", "standard")
            },
            "categorical_preprocessing": {
                "imputation": llm_recommendations.get("categorical_encoding", {}).get("imputation_strategy", "most_frequent"),
                "encoding": llm_recommendations.get("categorical_encoding", {}).get("low_cardinality", "onehot")
            },
            "train_test_split": {
                "test_size": self.config.data.test_size,
                "val_size": self.config.data.val_size,
                "random_state": self.config.data.random_state
            }
        }
    
    def _save_engineered_data(
        self, 
        X_train: pd.DataFrame, 
        y_train: Any, 
        X_test: pd.DataFrame, 
        y_test: Any,
        task_type: str,
        time_variable: Optional[str] = None,
        y_time_train: Optional[pd.Series] = None,
        y_time_test: Optional[pd.Series] = None
    ):
        """
        Save engineered dataframes (after feature engineering, before encoding/scaling)
        
        Saves:
        - train_engineered.csv: Training data with all engineered features
        - test_engineered.csv: Test data with all engineered features
        
        For survival tasks, includes both event and time columns.
        """
        try:
            from datetime import datetime
            from pathlib import Path
            
            # Create output directory
            output_dir = Path("outputs/engineered_data")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Prepare training data
            df_train = X_train.copy()
            
            if task_type == "survival":
                # For survival, y_train is structured array with event and time
                if hasattr(y_train, 'dtype') and y_train.dtype.names:
                    df_train['event'] = y_train['event']
                    df_train['time'] = y_train['time']
                elif y_time_train is not None:
                    # Fallback: separate event and time
                    df_train['event'] = y_train
                    df_train['time'] = y_time_train
            else:
                # For classification/regression
                df_train['target'] = y_train
            
            # Prepare test data
            df_test = X_test.copy()
            
            if task_type == "survival":
                if hasattr(y_test, 'dtype') and y_test.dtype.names:
                    df_test['event'] = y_test['event']
                    df_test['time'] = y_test['time']
                elif y_time_test is not None:
                    df_test['event'] = y_test
                    df_test['time'] = y_time_test
            else:
                df_test['target'] = y_test
            
            # Save to CSV
            train_path = output_dir / f"train_engineered_{timestamp}.csv"
            test_path = output_dir / f"test_engineered_{timestamp}.csv"
            
            df_train.to_csv(train_path, index=False)
            df_test.to_csv(test_path, index=False)
            
            self.log(f"💾 Saved engineered data:")
            self.log(f"   Training: {train_path} ({len(df_train)} samples, {len(df_train.columns)} columns)")
            self.log(f"   Test:     {test_path} ({len(df_test)} samples, {len(df_test.columns)} columns)")
            
        except Exception as e:
            self.log(f"Warning: Could not save engineered data: {str(e)}", "WARNING")
    
    def _generate_quality_report(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, Any]:
        """Generate data quality report after preprocessing"""
        
        report = {
            "missing_values_before": X_train.isnull().sum().to_dict(),
            "data_types": X_train.dtypes.astype(str).to_dict(),
            "target_distribution": {},
            "feature_statistics": {}
        }
        
        # Target distribution - handle structured arrays (survival) differently
        if hasattr(y_train, 'value_counts'):
            report["target_distribution"] = y_train.value_counts().to_dict()
        elif hasattr(y_train, 'dtype') and y_train.dtype.names:  # Structured array (survival)
            report["target_distribution"] = {
                "n_events": int(np.sum(y_train['event'])),
                "n_censored": int(len(y_train) - np.sum(y_train['event'])),
                "time_mean": float(np.mean(y_train['time'])),
                "time_median": float(np.median(y_train['time'])),
                "time_min": float(np.min(y_train['time'])),
                "time_max": float(np.max(y_train['time']))
            }
        else:
            report["target_distribution"] = {
                "mean": float(np.mean(y_train)),
                "std": float(np.std(y_train)),
                "min": float(np.min(y_train)),
                "max": float(np.max(y_train))
            }
        
        # Basic feature statistics
        numeric_cols = X_train.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            report["feature_statistics"] = X_train[numeric_cols].describe().to_dict()
        
        return report
    
    def save_preprocessor(self, output_dir: str) -> str:
        """Save the fitted preprocessor"""
        if self.preprocessor is None:
            raise ValueError("No preprocessor to save. Run feature engineering first.")
        
        output_path = Path(output_dir) / "preprocessor.pkl"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'wb') as f:
            pickle.dump({
                'preprocessor': self.preprocessor,
                'feature_names': self.feature_names
            }, f)
        
        self.log(f"Preprocessor saved to {output_path}")
        return str(output_path)
    
    def load_preprocessor(self, preprocessor_path: str):
        """Load a saved preprocessor"""
        with open(preprocessor_path, 'rb') as f:
            data = pickle.load(f)
            self.preprocessor = data['preprocessor']
            self.feature_names = data['feature_names']
        
        self.log(f"Preprocessor loaded from {preprocessor_path}")
    
    def transform_new_data(self, X: pd.DataFrame) -> np.ndarray:
        """Transform new data using fitted preprocessor"""
        if self.preprocessor is None:
            raise ValueError("No fitted preprocessor available. Train first or load saved preprocessor.")
        
        return self.preprocessor.transform(X)
