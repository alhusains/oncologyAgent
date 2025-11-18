"""Prompt templates for LLM interactions"""

from typing import Dict, Any, List
from dataclasses import dataclass


@dataclass
class PromptTemplates:
    """Collection of prompt templates for different ML tasks"""
    
    @staticmethod
    def data_analysis_prompt(data_info: Dict[str, Any], user_objective: str) -> str:
        """Prompt for data analysis and schema understanding"""
        return f"""
You are an expert data scientist analyzing a tabular dataset. Based on the provided information, perform a comprehensive analysis.

Dataset Information:
{data_info}

User Objective:
{user_objective}

Please provide a detailed analysis including:

1. **Target Variable Identification**:
   - Suggest the most appropriate target variable(s) based on the user objective
   - Explain your reasoning

2. **Task Type Classification**:
   - Determine if this is a classification or regression problem
   - Justify your decision

3. **Feature Analysis**:
   - Categorize columns as features, identifiers, or metadata
   - Identify numerical vs categorical features
   - Note any temporal features

4. **Data Quality Assessment**:
   - Identify potential data quality issues
   - Note missing value patterns
   - Flag any obvious outliers or anomalies

5. **Preprocessing Recommendations**:
   - Suggest specific preprocessing steps
   - Recommend handling for categorical variables
   - Suggest feature scaling approaches

Respond in JSON format with the following structure:
{{
    "target_variable": "column_name",
    "task_type": "classification|regression",
    "feature_columns": ["col1", "col2", ...],
    "categorical_columns": ["col1", "col2", ...],
    "numerical_columns": ["col1", "col2", ...],
    "identifier_columns": ["col1", "col2", ...],
    "data_quality_issues": ["issue1", "issue2", ...],
    "preprocessing_recommendations": {{
        "missing_values": "recommendation",
        "categorical_encoding": "recommendation",
        "feature_scaling": "recommendation",
        "outlier_handling": "recommendation"
    }},
    "confidence_score": 0.85,
    "reasoning": "Detailed explanation of decisions"
}}
"""

    @staticmethod
    def feature_engineering_prompt(
        data_schema: Dict[str, Any], 
        target_info: Dict[str, Any],
        domain_context: str = ""
    ) -> str:
        """Prompt for feature engineering suggestions - simplified to match implementation"""
        return f"""
You are an expert feature engineer. Based on the dataset schema, suggest SPECIFIC feature engineering operations that will be automatically applied.

Dataset Schema:
{data_schema}

Target Information:
{target_info}

Domain Context:
{domain_context}

YOUR TASK: Suggest feature engineering operations that will be AUTOMATICALLY IMPLEMENTED.

Review the ACTUAL features listed above and suggest:

1. **Feature Interactions** (1-4 interactions):
   - Which specific features should be combined?
   - For NUMERICAL × NUMERICAL: use "multiply" operation (e.g., age × tumor_size)
   - For CATEGORICAL × CATEGORICAL: use "concat" operation (e.g., smoking_status × sex)
   - For MIXED (num × cat): typically use "multiply" for numeric interaction
   - Only suggest features that ACTUALLY EXIST in the data above

2. **Mathematical Transformations** (1-4 features):
   - Which features would benefit from log or square root transforms?
   - Log (log1p): Good for right-skewed features, biomarkers, counts
   - Sqrt: Good for reducing extreme values, stabilizing variance
   - Only suggest for features that ACTUALLY EXIST in the data above


DOMAIN HINTS (only if relevant to THIS dataset):
- Oncology: Age×Stage interaction, log transforms for biomarkers (PSA, CEA, CA-125), tumor measurements
- General: Log for skewed distributions, sqrt for count data
- Always check if features actually exist before suggesting!

CRITICAL: Respond with this EXACT JSON format:

{{
    "feature_creator_operations": {{
        "feature_interactions": [
            {{
                "name": "age_x_stage",
                "features": ["age_at_diagnosis", "tumor_stage"],
                "operation": "multiply",
                "feature_types": ["numerical", "numerical"]
            }},
            {{
                "name": "smoking_x_sex",
                "features": ["smoking_status", "sex"],
                "operation": "concat",
                "feature_types": ["categorical", "categorical"]
            }}
        ],
        "transformations": [
            {{
                "features": ["psa_level", "cea_marker", "tumor_size"],
                "transform_type": "log1p",
                "prefix": "log"
            }},
            {{
                "features": ["mutation_count", "lymph_nodes_positive"],
                "transform_type": "sqrt",
                "prefix": "sqrt"
            }}
        ]
    }},
    "reasoning": "Brief explanation of why these specific operations make sense for THIS dataset"
}}

FORMATTING RULES:
1. Use the EXACT structure above - do NOT add extra keys like "feature_engineering_techniques"
2. Use actual feature names from the dataset provided above
3. If you don't want to create certain features, use empty arrays: "feature_interactions": []
4. "operation" must be exactly "multiply" or "concat"
5. "transform_type" must be exactly "log1p" or "sqrt"
6. Keep it focused - only suggest what will genuinely help for THIS specific dataset

Return ONLY the JSON structure shown above. Every suggested feature will be automatically created.
"""

    @staticmethod
    def model_selection_prompt(
        task_type: str,
        data_characteristics: Dict[str, Any],
        performance_requirements: Dict[str, Any] = None
    ) -> str:
        """Prompt for model selection"""
        return f"""
You are an expert ML model selector. Based on the task characteristics, recommend the best models and their configurations.

Task Type: {task_type}

Data Characteristics:
{data_characteristics}

Performance Requirements:
{performance_requirements or "Standard performance expectations"}

Please recommend models considering:

1. **Dataset Size and Dimensionality**:
   - Number of samples vs features
   - Computational constraints
   - Memory requirements

2. **Data Properties**:
   - Linear vs non-linear relationships
   - Feature interactions
   - Noise levels

3. **Interpretability Requirements**:
   - Need for explainable models
   - Feature importance analysis
   - Model transparency

4. **Performance vs Complexity Trade-offs**:
   - Training time constraints
   - Prediction speed requirements
   - Model maintenance complexity

Respond in JSON format:
{{
    "recommended_models": [
        {{
            "name": "model_name",
            "priority": 1,
            "reasoning": "Why this model is suitable",
            "hyperparameters": {{
                "param1": {{"range": [min, max], "type": "int|float|categorical"}},
                "param2": {{"values": ["option1", "option2"], "type": "categorical"}}
            }},
            "pros": ["advantage1", "advantage2"],
            "cons": ["limitation1", "limitation2"]
        }}
    ],
    "evaluation_metrics": ["metric1", "metric2", "metric3"],
    "cross_validation": {{
        "method": "k_fold|stratified_k_fold|time_series",
        "n_splits": 5,
        "random_state": 42
    }},
    "optimization_strategy": {{
        "method": "optuna|grid_search|random_search",
        "n_trials": 100,
        "timeout_minutes": 30
    }},
    "confidence_score": 0.9,
    "reasoning": "Overall model selection rationale"
}}
"""

    @staticmethod
    def results_critique_prompt(
        performance_results: Dict[str, Any],
        experiment_context: Dict[str, Any],
        iteration_count: int = 0
    ) -> str:
        """Prompt for critiquing model results"""
        return f"""
You are an expert ML consultant reviewing model performance results. Provide a comprehensive critique and improvement suggestions.

Performance Results:
{performance_results}

Experiment Context:
{experiment_context}

Current Iteration: {iteration_count}

Please analyze the results focusing on:

1. **Performance Assessment**:
   - Are the metrics satisfactory for the task?
   - Is there evidence of overfitting or underfitting?
   - How do different models compare?
   - Are there concerning patterns in the results?

2. **Model Comparison Analysis**:
   - Which models performed best and why?
   - Are there surprising results that need investigation?
   - Is the performance consistent across validation folds?

3. **Potential Issues Identification**:
   - Data leakage concerns
   - Validation strategy problems
   - Feature engineering limitations
   - Model selection issues

4. **Improvement Opportunities**:
   - Feature engineering enhancements
   - Alternative model architectures
   - Hyperparameter optimization
   - Data augmentation strategies
   - Ensemble methods

5. **Next Steps Recommendation**:
   - Should we iterate with improvements?
   - Are the results ready for production?
   - What specific changes would have the highest impact?

Respond in JSON format:
{{
    "overall_assessment": {{
        "performance_level": "excellent|good|fair|poor",
        "main_concerns": ["concern1", "concern2"],
        "key_strengths": ["strength1", "strength2"]
    }},
    "model_analysis": {{
        "best_model": "model_name",
        "performance_summary": "Brief summary of results",
        "concerning_patterns": ["pattern1", "pattern2"]
    }},
    "improvement_suggestions": [
        {{
            "category": "feature_engineering|model_selection|hyperparameters|data_quality",
            "suggestion": "Specific improvement suggestion",
            "expected_impact": "high|medium|low",
            "implementation_effort": "low|medium|high"
        }}
    ],
    "should_iterate": true,
    "iteration_priority": {{
        "feature_engineering": "high|medium|low",
        "model_selection": "high|medium|low",
        "hyperparameter_tuning": "high|medium|low",
        "data_collection": "high|medium|low"
    }},
    "confidence_score": 0.85,
    "detailed_reasoning": "Comprehensive explanation of the critique and recommendations"
}}
"""

    @staticmethod
    def interpretability_prompt(
        model_info: Dict[str, Any],
        feature_importance: Dict[str, Any],
        predictions_sample: Dict[str, Any]
    ) -> str:
        """Prompt for model interpretability analysis"""
        return f"""
You are an expert in ML model interpretability. Analyze the model's behavior and provide comprehensive insights.

Model Information:
{model_info}

Feature Importance:
{feature_importance}

Sample Predictions:
{predictions_sample}

Please provide interpretability insights covering:

1. **Feature Importance Analysis**:
   - Most influential features and their effects
   - Surprising or counterintuitive feature rankings
   - Feature interactions and dependencies

2. **Model Behavior Patterns**:
   - How the model makes decisions
   - Key decision boundaries
   - Model biases or preferences

3. **Prediction Explanations**:
   - Explanation of sample predictions
   - Feature contributions to specific outcomes
   - Confidence and uncertainty patterns

4. **Business Insights**:
   - Actionable insights for stakeholders
   - Feature engineering implications
   - Data collection recommendations

Respond in JSON format:
{{
    "feature_insights": {{
        "top_features": [
            {{
                "feature": "feature_name",
                "importance": 0.15,
                "effect": "positive|negative",
                "interpretation": "What this feature tells us"
            }}
        ],
        "surprising_features": ["feature1", "feature2"],
        "feature_interactions": [
            {{
                "features": ["feature1", "feature2"],
                "interaction_type": "synergistic|antagonistic",
                "description": "How they interact"
            }}
        ]
    }},
    "model_behavior": {{
        "decision_patterns": ["pattern1", "pattern2"],
        "biases_detected": ["bias1", "bias2"],
        "uncertainty_patterns": "Description of when model is uncertain"
    }},
    "business_insights": [
        {{
            "insight": "Business-relevant insight",
            "actionable_recommendation": "What stakeholders should do",
            "priority": "high|medium|low"
        }}
    ],
    "prediction_explanations": {{
        "sample_explanations": [
            {{
                "prediction": "predicted_value",
                "confidence": 0.85,
                "key_factors": ["factor1", "factor2"],
                "explanation": "Why this prediction was made"
            }}
        ]
    }},
    "confidence_score": 0.8,
    "summary": "Overall interpretability summary and key takeaways"
}}
"""

    @staticmethod
    def report_generation_prompt(experiment_results: Dict[str, Any]) -> str:
        """Prompt for generating comprehensive ML experiment report"""
        return f"""
You are an expert data science report writer. Create a comprehensive, professional report from the ML experiment results.

Experiment Results:
{experiment_results}

Please create a detailed report with the following sections:

1. **Executive Summary**
   - Project objective and key findings
   - Model performance summary
   - Business recommendations
   - Key insights and impact

2. **Data Analysis Summary**
   - Dataset characteristics and quality
   - Target variable analysis
   - Feature distribution insights
   - Data preprocessing decisions

3. **Methodology**
   - Feature engineering approach
   - Model selection rationale
   - Validation strategy
   - Performance metrics used

4. **Results and Performance**
   - Model comparison and rankings
   - Performance metrics analysis
   - Cross-validation results
   - Best model details

5. **Model Interpretability**
   - Feature importance analysis
   - Key drivers of predictions
   - Model behavior insights
   - Prediction explanations

6. **Conclusions and Recommendations**
   - Key findings and insights
   - Actionable recommendations
   - Limitations and considerations
   - Future work suggestions

7. **Technical Appendix**
   - Detailed methodology
   - Hyperparameter configurations
   - Additional metrics and plots
   - Reproduction instructions

Format the report in clear, professional markdown with:
- Clear section headers
- Bullet points for key insights
- Technical details in appropriate sections
- Executive-friendly summary at the top
- Professional tone suitable for stakeholders

The report should be comprehensive yet accessible to both technical and non-technical audiences.
"""
