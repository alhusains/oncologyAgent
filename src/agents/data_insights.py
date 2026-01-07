"""
Data Insights Module

Provides comprehensive data analysis and insights for ML datasets.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class DataInsightsAnalyzer:
    """
    Comprehensive data analysis for ML datasets.
    
    Provides:
    - Dataset overview
    - Descriptive statistics
    - Missing data analysis
    - Target variable distribution
    - Correlation analysis
    - Data quality metrics
    - Clinical insights (for medical data)
    """
    
    def __init__(self):
        pass
    
    def analyze(
        self,
        dataset_path: Optional[str] = None,
        df: Optional[pd.DataFrame] = None,
        target_variable: Optional[str] = None,
        task_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Perform comprehensive data analysis.
        
        Args:
            dataset_path: Path to dataset file (if df not provided)
            df: DataFrame (if already loaded)
            target_variable: Name of target variable
            task_type: Type of ML task (classification, regression, survival)
            
        Returns:
            Dictionary with comprehensive insights
        """
        # Load data if not provided
        if df is None:
            if dataset_path is None:
                raise ValueError("Either dataset_path or df must be provided")
            df = pd.read_csv(dataset_path)
        else:
            df = df.copy()
        
        insights = {}
        
        # 1. Dataset Overview
        insights['overview'] = self._get_overview(df)
        
        # 2. Descriptive Statistics
        insights['statistics'] = self._get_statistics(df)
        
        # 3. Missing Data Analysis
        insights['missing_data'] = self._analyze_missing_data(df)
        
        # 4. Target Variable Analysis
        if target_variable and target_variable in df.columns:
            insights['target_analysis'] = self._analyze_target(
                df, target_variable, task_type
            )
        
        # 5. Correlation Analysis
        insights['correlations'] = self._analyze_correlations(df, target_variable)
        
        # 6. Feature Types
        insights['feature_types'] = self._categorize_features(df)
        
        # 7. Data Quality Metrics
        insights['data_quality'] = self._assess_data_quality(df)
        
        # 8. Outliers Detection
        insights['outliers'] = self._detect_outliers(df)
        
        # 9. Clinical Insights (if applicable)
        insights['clinical_insights'] = self._get_clinical_insights(df, target_variable)
        
        return insights
    
    def _get_overview(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get basic dataset overview"""
        return {
            'n_samples': len(df),
            'n_features': len(df.columns),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2,
            'columns': list(df.columns),
            'dtypes': df.dtypes.value_counts().to_dict()
        }
    
    def _get_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get descriptive statistics"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
        stats = {
            'numeric': {},
            'categorical': {}
        }
        
        # Numeric features
        if len(numeric_cols) > 0:
            desc = df[numeric_cols].describe()
            stats['numeric'] = {
                col: {
                    'mean': desc[col]['mean'],
                    'std': desc[col]['std'],
                    'min': desc[col]['min'],
                    'max': desc[col]['max'],
                    'median': desc[col]['50%'],
                    'q25': desc[col]['25%'],
                    'q75': desc[col]['75%']
                }
                for col in numeric_cols
            }
        
        # Categorical features
        if len(categorical_cols) > 0:
            stats['categorical'] = {
                col: {
                    'n_unique': df[col].nunique(),
                    'most_common': df[col].value_counts().head(5).to_dict(),
                    'missing_count': df[col].isna().sum()
                }
                for col in categorical_cols
            }
        
        return stats
    
    def _analyze_missing_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze missing data patterns"""
        missing_counts = df.isna().sum()
        missing_pcts = (missing_counts / len(df) * 100).round(2)
        
        missing_features = missing_counts[missing_counts > 0].to_dict()
        missing_pcts_dict = missing_pcts[missing_pcts > 0].to_dict()
        
        return {
            'total_missing_values': int(df.isna().sum().sum()),
            'features_with_missing': len(missing_features),
            'missing_by_feature': {
                feat: {
                    'count': int(missing_features[feat]),
                    'percentage': float(missing_pcts_dict[feat])
                }
                for feat in missing_features
            },
            'rows_with_any_missing': int(df.isna().any(axis=1).sum()),
            'percentage_rows_with_missing': round(
                df.isna().any(axis=1).sum() / len(df) * 100, 2
            )
        }
    
    def _analyze_target(
        self, df: pd.DataFrame, target: str, task_type: Optional[str]
    ) -> Dict[str, Any]:
        """Analyze target variable"""
        target_series = df[target]
        
        analysis = {
            'variable_name': target,
            'data_type': str(target_series.dtype),
            'missing_count': int(target_series.isna().sum()),
            'missing_percentage': round(target_series.isna().sum() / len(df) * 100, 2)
        }
        
        # Task-specific analysis
        if task_type == 'classification' or target_series.nunique() < 20:
            # Classification
            value_counts = target_series.value_counts()
            analysis['task_hint'] = 'classification'
            analysis['n_classes'] = int(target_series.nunique())
            analysis['class_distribution'] = value_counts.to_dict()
            analysis['class_balance'] = {
                'balanced': value_counts.min() / value_counts.max() > 0.5,
                'imbalance_ratio': float(value_counts.max() / value_counts.min())
            }
        elif task_type == 'survival':
            # Survival analysis
            analysis['task_hint'] = 'survival'
            analysis['mean_time'] = float(target_series.mean())
            analysis['median_time'] = float(target_series.median())
            analysis['range'] = [float(target_series.min()), float(target_series.max())]
        else:
            # Regression
            analysis['task_hint'] = 'regression'
            analysis['mean'] = float(target_series.mean())
            analysis['std'] = float(target_series.std())
            analysis['median'] = float(target_series.median())
            analysis['range'] = [float(target_series.min()), float(target_series.max())]
            analysis['skewness'] = float(target_series.skew())
        
        return analysis
    
    def _analyze_correlations(
        self, df: pd.DataFrame, target: Optional[str]
    ) -> Dict[str, Any]:
        """Analyze feature correlations"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) < 2:
            return {'message': 'Not enough numeric features for correlation analysis'}
        
        corr_matrix = df[numeric_cols].corr()
        
        # Find highly correlated features (excluding diagonal)
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.7:
                    high_corr_pairs.append({
                        'feature1': corr_matrix.columns[i],
                        'feature2': corr_matrix.columns[j],
                        'correlation': float(corr_val)
                    })
        
        result = {
            'high_correlation_pairs': high_corr_pairs,
            'n_high_correlations': len(high_corr_pairs)
        }
        
        # Target correlations
        if target and target in numeric_cols:
            target_corrs = corr_matrix[target].drop(target).sort_values(
                ascending=False, key=abs
            )
            result['target_correlations'] = {
                'top_positive': target_corrs.head(5).to_dict(),
                'top_negative': target_corrs.tail(5).to_dict()
            }
        
        return result
    
    def _categorize_features(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Categorize features by type"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Further categorize numeric
        binary_numeric = [
            col for col in numeric_cols 
            if df[col].nunique() == 2
        ]
        
        continuous = [
            col for col in numeric_cols 
            if col not in binary_numeric and df[col].nunique() > 10
        ]
        
        discrete = [
            col for col in numeric_cols 
            if col not in binary_numeric and col not in continuous
        ]
        
        return {
            'numeric': {
                'all': numeric_cols,
                'continuous': continuous,
                'discrete': discrete,
                'binary': binary_numeric,
                'count': len(numeric_cols)
            },
            'categorical': {
                'all': categorical_cols,
                'count': len(categorical_cols)
            }
        }
    
    def _assess_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Assess overall data quality"""
        total_cells = df.shape[0] * df.shape[1]
        missing_cells = df.isna().sum().sum()
        
        # Check for duplicates
        n_duplicates = df.duplicated().sum()
        
        # Check for constant features
        constant_features = [
            col for col in df.columns 
            if df[col].nunique() == 1
        ]
        
        # Check for high cardinality categoricals
        high_cardinality = []
        for col in df.select_dtypes(include=['object', 'category']).columns:
            if df[col].nunique() > 50:
                high_cardinality.append({
                    'feature': col,
                    'n_unique': int(df[col].nunique())
                })
        
        quality_score = 100.0
        issues = []
        
        if missing_cells / total_cells > 0.1:
            quality_score -= 20
            issues.append("High proportion of missing data (>10%)")
        
        if n_duplicates > 0:
            quality_score -= 10
            issues.append(f"{n_duplicates} duplicate rows found")
        
        if len(constant_features) > 0:
            quality_score -= 10
            issues.append(f"{len(constant_features)} constant features (no variation)")
        
        if len(high_cardinality) > 0:
            quality_score -= 5
            issues.append(f"{len(high_cardinality)} high-cardinality categorical features")
        
        return {
            'quality_score': max(0, quality_score),
            'completeness_percentage': round((total_cells - missing_cells) / total_cells * 100, 2),
            'n_duplicate_rows': int(n_duplicates),
            'constant_features': constant_features,
            'high_cardinality_features': high_cardinality,
            'issues': issues
        }
    
    def _detect_outliers(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect outliers in numeric features using IQR method"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        outlier_summary = {}
        
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
            
            if len(outliers) > 0:
                outlier_summary[col] = {
                    'n_outliers': int(len(outliers)),
                    'percentage': round(len(outliers) / len(df) * 100, 2),
                    'bounds': {
                        'lower': float(lower_bound),
                        'upper': float(upper_bound)
                    }
                }
        
        return {
            'features_with_outliers': len(outlier_summary),
            'outlier_details': outlier_summary
        }
    
    def _get_clinical_insights(
        self, df: pd.DataFrame, target: Optional[str]
    ) -> Dict[str, Any]:
        """Extract clinical insights if medical data is detected"""
        clinical_keywords = [
            'age', 'gender', 'sex', 'survival', 'death', 'alive', 'deceased',
            'tumor', 'cancer', 'stage', 'grade', 'mutation', 'treatment',
            'therapy', 'patient', 'diagnosis', 'prognosis'
        ]
        
        # Detect if this is likely clinical data
        column_names_lower = [col.lower() for col in df.columns]
        is_clinical = any(
            keyword in ' '.join(column_names_lower) 
            for keyword in clinical_keywords
        )
        
        if not is_clinical:
            return {'is_clinical_data': False}
        
        insights = {'is_clinical_data': True, 'detected_clinical_features': []}
        
        # Identify clinical features
        for col in df.columns:
            col_lower = col.lower()
            for keyword in clinical_keywords:
                if keyword in col_lower:
                    insights['detected_clinical_features'].append({
                        'feature': col,
                        'type': keyword,
                        'n_unique': int(df[col].nunique())
                    })
                    break
        
        # Age analysis if present
        age_cols = [col for col in df.columns if 'age' in col.lower()]
        if age_cols:
            age_col = age_cols[0]
            insights['age_distribution'] = {
                'mean': float(df[age_col].mean()),
                'median': float(df[age_col].median()),
                'range': [float(df[age_col].min()), float(df[age_col].max())],
                'std': float(df[age_col].std())
            }
        
        # Gender distribution if present
        gender_cols = [col for col in df.columns if any(g in col.lower() for g in ['gender', 'sex'])]
        if gender_cols:
            gender_col = gender_cols[0]
            insights['gender_distribution'] = df[gender_col].value_counts().to_dict()
        
        return insights
    
    def format_report(self, insights: Dict[str, Any]) -> str:
        """Format insights as a readable report"""
        report = []
        report.append("=" * 70)
        report.append("DATA INSIGHTS REPORT")
        report.append("=" * 70)
        report.append("")
        
        # Overview
        if 'overview' in insights:
            ov = insights['overview']
            report.append("📊 DATASET OVERVIEW")
            report.append("-" * 70)
            report.append(f"   Samples: {ov['n_samples']:,}")
            report.append(f"   Features: {ov['n_features']}")
            report.append(f"   Memory Usage: {ov['memory_usage_mb']:.2f} MB")
            report.append("")
        
        # Target Analysis
        if 'target_analysis' in insights:
            ta = insights['target_analysis']
            report.append("🎯 TARGET VARIABLE ANALYSIS")
            report.append("-" * 70)
            report.append(f"   Variable: {ta['variable_name']}")
            report.append(f"   Task Type: {ta.get('task_hint', 'unknown')}")
            
            if 'n_classes' in ta:
                report.append(f"   Classes: {ta['n_classes']}")
                report.append(f"   Balanced: {'Yes' if ta['class_balance']['balanced'] else 'No'}")
                report.append(f"   Imbalance Ratio: {ta['class_balance']['imbalance_ratio']:.2f}")
                report.append("   Class Distribution:")
                for cls, count in ta['class_distribution'].items():
                    report.append(f"      {cls}: {count}")
            elif 'mean' in ta:
                report.append(f"   Mean: {ta['mean']:.3f}")
                report.append(f"   Median: {ta['median']:.3f}")
                report.append(f"   Std: {ta['std']:.3f}")
                report.append(f"   Range: [{ta['range'][0]:.3f}, {ta['range'][1]:.3f}]")
            report.append("")
        
        # Missing Data
        if 'missing_data' in insights:
            md = insights['missing_data']
            report.append("❓ MISSING DATA")
            report.append("-" * 70)
            report.append(f"   Total Missing Values: {md['total_missing_values']:,}")
            report.append(f"   Features with Missing: {md['features_with_missing']}")
            report.append(f"   Rows with Missing: {md['rows_with_any_missing']:,} ({md['percentage_rows_with_missing']}%)")
            
            if md['features_with_missing'] > 0:
                report.append("   Top Features with Missing Data:")
                sorted_missing = sorted(
                    md['missing_by_feature'].items(),
                    key=lambda x: x[1]['percentage'],
                    reverse=True
                )[:5]
                for feat, info in sorted_missing:
                    report.append(f"      {feat}: {info['percentage']}%")
            report.append("")
        
        # Data Quality
        if 'data_quality' in insights:
            dq = insights['data_quality']
            report.append("✅ DATA QUALITY ASSESSMENT")
            report.append("-" * 70)
            report.append(f"   Quality Score: {dq['quality_score']:.1f}/100")
            report.append(f"   Completeness: {dq['completeness_percentage']}%")
            report.append(f"   Duplicate Rows: {dq['n_duplicate_rows']}")
            report.append(f"   Constant Features: {len(dq['constant_features'])}")
            
            if dq['issues']:
                report.append("   Issues Found:")
                for issue in dq['issues']:
                    report.append(f"      ⚠️  {issue}")
            report.append("")
        
        # Feature Types
        if 'feature_types' in insights:
            ft = insights['feature_types']
            report.append("🔢 FEATURE TYPES")
            report.append("-" * 70)
            report.append(f"   Numeric Features: {ft['numeric']['count']}")
            report.append(f"      Continuous: {len(ft['numeric']['continuous'])}")
            report.append(f"      Discrete: {len(ft['numeric']['discrete'])}")
            report.append(f"      Binary: {len(ft['numeric']['binary'])}")
            report.append(f"   Categorical Features: {ft['categorical']['count']}")
            report.append("")
        
        # Correlations
        if 'correlations' in insights and 'target_correlations' in insights['correlations']:
            corr = insights['correlations']
            report.append("🔗 TOP CORRELATIONS WITH TARGET")
            report.append("-" * 70)
            
            if 'top_positive' in corr['target_correlations']:
                report.append("   Positive Correlations:")
                for feat, val in list(corr['target_correlations']['top_positive'].items())[:5]:
                    report.append(f"      {feat}: {val:.3f}")
            report.append("")
        
        # Outliers
        if 'outliers' in insights and insights['outliers']['features_with_outliers'] > 0:
            out = insights['outliers']
            report.append("📈 OUTLIERS DETECTED")
            report.append("-" * 70)
            report.append(f"   Features with Outliers: {out['features_with_outliers']}")
            
            sorted_outliers = sorted(
                out['outlier_details'].items(),
                key=lambda x: x[1]['percentage'],
                reverse=True
            )[:5]
            for feat, info in sorted_outliers:
                report.append(f"      {feat}: {info['n_outliers']} ({info['percentage']}%)")
            report.append("")
        
        # Clinical Insights
        if 'clinical_insights' in insights and insights['clinical_insights'].get('is_clinical_data'):
            ci = insights['clinical_insights']
            report.append("🏥 CLINICAL INSIGHTS")
            report.append("-" * 70)
            report.append(f"   Clinical Data Detected: Yes")
            report.append(f"   Clinical Features: {len(ci['detected_clinical_features'])}")
            
            if 'age_distribution' in ci:
                age = ci['age_distribution']
                report.append(f"   Age: Mean={age['mean']:.1f}, Range=[{age['range'][0]:.0f}, {age['range'][1]:.0f}]")
            
            if 'gender_distribution' in ci:
                report.append("   Gender Distribution:")
                for gender, count in ci['gender_distribution'].items():
                    report.append(f"      {gender}: {count}")
            report.append("")
        
        report.append("=" * 70)
        
        return "\n".join(report)

