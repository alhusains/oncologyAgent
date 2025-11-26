"""
Interpretability Report Generator

Generates clinician-friendly PDF reports with model explanations,
SHAP values, and visualizations for classification, regression, and survival analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("Warning: SHAP not installed. Install with: pip install shap")


class InterpretabilityReportGenerator:
    """
    Generate comprehensive interpretability reports for ML models.
    
    Supports:
    - Classification (binary/multiclass)
    - Regression
    - Survival analysis
    
    Includes:
    - SHAP values and plots
    - Feature importance
    - Prediction distributions
    - Model performance metrics
    - Clinical decision guidance
    """
    
    def __init__(self):
        # Set plotting style
        sns.set_style("whitegrid")
        plt.rcParams['figure.facecolor'] = 'white'
        plt.rcParams['font.size'] = 10
    
    def generate_report(
        self,
        model_name: str,
        model_obj: Any,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        y_pred: np.ndarray,
        task_type: str,
        metrics: Dict[str, Any],
        feature_importance: Optional[Dict[str, float]] = None,
        output_path: Optional[str] = None,
        additional_info: Optional[Dict[str, Any]] = None,
        y_pred_proba: Optional[np.ndarray] = None
    ) -> str:
        """
        Generate comprehensive interpretability PDF report.
        
        Args:
            model_name: Name of the model
            model_obj: Trained model object
            X_test: Test features
            y_test: True test labels/values
            y_pred: Model predictions
            task_type: Type of task
            metrics: Performance metrics
            feature_importance: Feature importance dict
            output_path: Custom output path
            additional_info: Additional information
            y_pred_proba: Prediction probabilities (for ROC curve in classification)
            y_pred: Model predictions
            task_type: 'classification', 'regression', or 'survival'
            metrics: Performance metrics dictionary
            feature_importance: Feature importance scores
            output_path: Path to save PDF (if None, auto-generated)
            additional_info: Additional information to include
            
        Returns:
            Path to generated PDF report
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = Path("outputs/interpretability_reports")
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = str(output_dir / f"{model_name}_{task_type}_{timestamp}.pdf")
        
        # Generate SHAP explainer if available
        shap_values = None
        explainer = None
        
        if SHAP_AVAILABLE:
            try:
                shap_values, explainer = self._compute_shap_values(
                    model_obj, X_test, task_type
                )
            except Exception as e:
                print(f"Could not compute SHAP values: {e}")
        
        # Create PDF with multiple pages
        with PdfPages(output_path) as pdf:
            # Page 1: Title and Overview
            self._create_title_page(
                pdf, model_name, task_type, metrics, additional_info
            )
            
            # Page 2: Performance Metrics
            self._create_metrics_page(
                pdf, metrics, task_type, y_test, y_pred, y_pred_proba
            )
            
            # Page 3: Feature Importance
            if feature_importance:
                self._create_feature_importance_page(
                    pdf, feature_importance, task_type
                )
            
            # Page 4-5: SHAP Analysis
            if shap_values is not None:
                self._create_shap_pages(
                    pdf, shap_values, explainer, X_test, task_type
                )
            
            # Page 6: Prediction Distribution
            self._create_prediction_distribution_page(
                pdf, y_test, y_pred, task_type
            )
            
            # Page 7: Clinical Decision Guidance (if classification or survival)
            if task_type in ['classification', 'survival']:
                self._create_clinical_guidance_page(
                    pdf, model_name, metrics, task_type, feature_importance
                )
            
            # Add metadata
            d = pdf.infodict()
            d['Title'] = f'Interpretability Report: {model_name}'
            d['Author'] = 'Oncology ML Agent'
            d['Subject'] = f'{task_type.capitalize()} Model Interpretation'
            d['CreationDate'] = datetime.now()
        
        print(f"\n✅ Interpretability report saved: {output_path}")
        return output_path
    
    def _compute_shap_values(
        self, model, X_test: pd.DataFrame, task_type: str
    ) -> Tuple[Any, Any]:
        """Compute SHAP values for the model"""
        # Limit samples for SHAP (computational cost)
        X_sample = X_test.sample(min(100, len(X_test)), random_state=42)
        
        # Check if this is an AutoGluon model
        model_type = str(type(model).__name__)
        is_autogluon = 'TabularPredictor' in model_type or 'autogluon' in str(type(model)).lower()
        
        # Choose appropriate explainer
        try:
            if is_autogluon:
                # AutoGluon models need special handling
                print("     Using model-agnostic explainer for AutoGluon...")
                
                # Create a prediction function wrapper
                def predict_fn(X):
                    # AutoGluon expects DataFrame
                    if not isinstance(X, pd.DataFrame):
                        X = pd.DataFrame(X, columns=X_test.columns)
                    return model.predict(X, as_pandas=False)
                
                # Use KernelExplainer with smaller background dataset
                background = shap.sample(X_test, min(30, len(X_test)))
                explainer = shap.KernelExplainer(predict_fn, background)
                
                # Compute SHAP values (this will take longer)
                shap_values = explainer.shap_values(X_sample)
                
            else:
                # Try TreeExplainer first (fast for tree models)
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_sample)
                
        except Exception as e:
            print(f"     TreeExplainer failed: {e}")
            try:
                # Fall back to KernelExplainer (slower but works for any model)
                print("     Trying KernelExplainer (this may take a moment)...")
                background = shap.sample(X_test, min(50, len(X_test)))
                explainer = shap.KernelExplainer(model.predict, background)
                shap_values = explainer.shap_values(X_sample)
            except Exception as e2:
                print(f"     Could not create SHAP explainer: {e2}")
                return None, None
        
        return shap_values, explainer
    
    def _create_title_page(
        self,
        pdf: PdfPages,
        model_name: str,
        task_type: str,
        metrics: Dict[str, Any],
        additional_info: Optional[Dict[str, Any]]
    ):
        """Create title page with overview"""
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis('off')
        
        # Title
        title_text = "Model Interpretability Report"
        ax.text(0.5, 0.9, title_text, ha='center', va='top', 
                fontsize=24, fontweight='bold', transform=ax.transAxes)
        
        # Subtitle
        subtitle = f"{task_type.capitalize()} Model: {model_name}"
        ax.text(0.5, 0.82, subtitle, ha='center', va='top',
                fontsize=16, transform=ax.transAxes, style='italic')
        
        # Horizontal line
        ax.plot([0.1, 0.9], [0.78, 0.78], 'k-', lw=2, transform=ax.transAxes)
        
        # Report details
        details_y = 0.70
        ax.text(0.5, details_y, "Report Details", ha='center', va='top',
                fontsize=14, fontweight='bold', transform=ax.transAxes)
        
        details_y -= 0.05
        report_date = datetime.now().strftime("%B %d, %Y at %I:%M %p")
        ax.text(0.5, details_y, f"Generated: {report_date}", ha='center', va='top',
                fontsize=11, transform=ax.transAxes)
        
        # Key metrics overview
        metrics_y = 0.55
        ax.text(0.5, metrics_y, "Performance Summary", ha='center', va='top',
                fontsize=14, fontweight='bold', transform=ax.transAxes)
        
        metrics_y -= 0.08
        
        if task_type == 'classification':
            key_metrics = [
                f"Accuracy: {metrics.get('accuracy', 0):.3f}",
                f"F1-Score: {metrics.get('f1', 0):.3f}",
                f"ROC-AUC: {metrics.get('roc_auc', 0):.3f}" if metrics.get('roc_auc') else ""
            ]
        elif task_type == 'regression':
            key_metrics = [
                f"R² Score: {metrics.get('r2', 0):.3f}",
                f"MAE: {metrics.get('mae', 0):.3f}",
                f"RMSE: {metrics.get('rmse', 0):.3f}"
            ]
        else:  # survival
            key_metrics = [
                f"C-Index: {metrics.get('concordance_index', 0):.3f}",
                f"IBS: {metrics.get('integrated_brier_score', 0):.3f}" if metrics.get('integrated_brier_score') else ""
            ]
        
        for metric in [m for m in key_metrics if m]:
            ax.text(0.5, metrics_y, metric, ha='center', va='top',
                    fontsize=12, transform=ax.transAxes)
            metrics_y -= 0.05
        
        # Additional info
        if additional_info:
            info_y = 0.30
            ax.text(0.5, info_y, "Dataset Information", ha='center', va='top',
                    fontsize=14, fontweight='bold', transform=ax.transAxes)
            info_y -= 0.05
            
            if 'n_samples_test' in additional_info:
                ax.text(0.5, info_y, f"Test Samples: {additional_info['n_samples_test']}", 
                       ha='center', va='top', fontsize=11, transform=ax.transAxes)
                info_y -= 0.04
            
            if 'n_features' in additional_info:
                ax.text(0.5, info_y, f"Features: {additional_info['n_features']}", 
                       ha='center', va='top', fontsize=11, transform=ax.transAxes)
                info_y -= 0.04
        
        # Footer
        footer_text = "This report provides model interpretations to support clinical decision-making.\n" \
                      "Always combine model predictions with clinical expertise."
        ax.text(0.5, 0.1, footer_text, ha='center', va='top',
                fontsize=9, style='italic', transform=ax.transAxes,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
    
    def _create_metrics_page(
        self,
        pdf: PdfPages,
        metrics: Dict[str, Any],
        task_type: str,
        y_test: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: Optional[np.ndarray] = None
    ):
        """Create page with detailed performance metrics"""
        fig = plt.figure(figsize=(11, 8.5))
        
        if task_type == 'classification':
            self._plot_classification_metrics(fig, metrics, y_test, y_pred, y_pred_proba)
        elif task_type == 'regression':
            self._plot_regression_metrics(fig, metrics, y_test, y_pred)
        else:  # survival
            self._plot_survival_metrics(fig, metrics, y_test, y_pred)
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
    
    def _plot_classification_metrics(
        self, fig, metrics: Dict[str, Any], y_test: np.ndarray, y_pred: np.ndarray,
        y_pred_proba: Optional[np.ndarray] = None
    ):
        """Plot classification-specific metrics"""
        from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
        
        # Title
        fig.suptitle('Classification Performance Metrics', fontsize=16, fontweight='bold', y=0.98)
        
        # Create subplots
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3,
                             left=0.1, right=0.95, top=0.92, bottom=0.08)
        
        # 1. Confusion Matrix
        ax1 = fig.add_subplot(gs[0, 0])
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1, cbar=False)
        ax1.set_title('Confusion Matrix', fontweight='bold')
        ax1.set_ylabel('True Label')
        ax1.set_xlabel('Predicted Label')
        
        # 2. Metrics Table
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.axis('off')
        metrics_data = [
            ['Metric', 'Value'],
            ['Accuracy', f"{metrics.get('accuracy', 0):.3f}"],
            ['Precision', f"{metrics.get('precision', 0):.3f}"],
            ['Recall', f"{metrics.get('recall', 0):.3f}"],
            ['F1-Score', f"{metrics.get('f1', 0):.3f}"],
            ['ROC-AUC', f"{metrics.get('roc_auc', 0):.3f}"]
        ]
        table = ax2.table(cellText=metrics_data, cellLoc='left', loc='center',
                         colWidths=[0.5, 0.3])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Style header
        for i in range(2):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        ax2.set_title('Performance Metrics', fontweight='bold', pad=20)
        
        # 3. ROC Curve (if binary classification and probabilities available)
        if len(np.unique(y_test)) == 2 and metrics.get('roc_auc'):
            ax3 = fig.add_subplot(gs[1, :])
            
            # Plot actual ROC curve if probabilities are available
            if y_pred_proba is not None:
                # For binary classification, use probabilities of positive class
                if y_pred_proba.ndim == 2:
                    # Two columns: [prob_class_0, prob_class_1]
                    y_scores = y_pred_proba[:, 1]
                else:
                    # Single column: probability of positive class
                    y_scores = y_pred_proba
                
                fpr, tpr, _ = roc_curve(y_test, y_scores)
                roc_auc = auc(fpr, tpr)
                
                ax3.plot(fpr, tpr, color='darkorange', lw=2, 
                        label=f'ROC curve (AUC = {roc_auc:.3f})')
            
            # Plot random baseline
            ax3.plot([0, 1], [0, 1], 'k--', lw=2, label='Random')
            ax3.set_xlabel('False Positive Rate')
            ax3.set_ylabel('True Positive Rate')
            ax3.set_title(f'ROC Curve (AUC = {metrics.get("roc_auc", 0):.3f})', fontweight='bold')
            ax3.legend(loc='lower right')
            ax3.grid(True, alpha=0.3)
            ax3.set_xlim([0.0, 1.0])
            ax3.set_ylim([0.0, 1.05])
        
        # 4. Class Distribution
        ax4 = fig.add_subplot(gs[2, :])
        unique_classes, pred_counts = np.unique(y_pred, return_counts=True)
        _, true_counts = np.unique(y_test, return_counts=True)
        
        x = np.arange(len(unique_classes))
        width = 0.35
        ax4.bar(x - width/2, true_counts, width, label='True', alpha=0.8)
        ax4.bar(x + width/2, pred_counts, width, label='Predicted', alpha=0.8)
        ax4.set_xlabel('Class')
        ax4.set_ylabel('Count')
        ax4.set_title('True vs Predicted Class Distribution', fontweight='bold')
        ax4.set_xticks(x)
        ax4.set_xticklabels(unique_classes)
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis='y')
    
    def _plot_regression_metrics(
        self, fig, metrics: Dict[str, Any], y_test: np.ndarray, y_pred: np.ndarray
    ):
        """Plot regression-specific metrics"""
        fig.suptitle('Regression Performance Metrics', fontsize=16, fontweight='bold', y=0.98)
        
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3,
                             left=0.1, right=0.95, top=0.92, bottom=0.08)
        
        # 1. Predicted vs Actual
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.scatter(y_test, y_pred, alpha=0.5, s=30)
        
        # Add perfect prediction line
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
        
        ax1.set_xlabel('True Values')
        ax1.set_ylabel('Predicted Values')
        ax1.set_title('Predicted vs Actual', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Residuals Plot
        ax2 = fig.add_subplot(gs[0, 1])
        residuals = y_test - y_pred
        ax2.scatter(y_pred, residuals, alpha=0.5, s=30)
        ax2.axhline(y=0, color='r', linestyle='--', lw=2)
        ax2.set_xlabel('Predicted Values')
        ax2.set_ylabel('Residuals')
        ax2.set_title('Residual Plot', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # 3. Residual Distribution
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.hist(residuals, bins=30, edgecolor='black', alpha=0.7)
        ax3.axvline(x=0, color='r', linestyle='--', lw=2)
        ax3.set_xlabel('Residuals')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Residual Distribution', fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')
        
        # 4. Metrics Table
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.axis('off')
        metrics_data = [
            ['Metric', 'Value'],
            ['R² Score', f"{metrics.get('r2', 0):.3f}"],
            ['MAE', f"{metrics.get('mae', 0):.3f}"],
            ['RMSE', f"{metrics.get('rmse', 0):.3f}"],
            ['MAPE', f"{metrics.get('mape', 0):.2f}%" if 'mape' in metrics else 'N/A']
        ]
        table = ax4.table(cellText=metrics_data, cellLoc='left', loc='center',
                         colWidths=[0.5, 0.3])
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 2.5)
        
        for i in range(2):
            table[(0, i)].set_facecolor('#2196F3')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        ax4.set_title('Performance Metrics', fontweight='bold', pad=20)
    
    def _plot_survival_metrics(
        self, fig, metrics: Dict[str, Any], y_test: np.ndarray, y_pred: np.ndarray
    ):
        """Plot survival-specific metrics"""
        fig.suptitle('Survival Analysis Performance Metrics', fontsize=16, fontweight='bold', y=0.98)
        
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3,
                             left=0.1, right=0.95, top=0.92, bottom=0.08)
        
        # 1. Metrics Table
        ax1 = fig.add_subplot(gs[0, :])
        ax1.axis('off')
        
        metrics_data = [['Metric', 'Value', 'Interpretation']]
        
        if 'concordance_index' in metrics:
            c_index = metrics['concordance_index']
            interp = 'Excellent' if c_index > 0.8 else 'Good' if c_index > 0.7 else 'Fair'
            metrics_data.append(['C-Index', f"{c_index:.3f}", interp])
        
        if 'integrated_brier_score' in metrics:
            ibs = metrics['integrated_brier_score']
            interp = 'Good' if ibs < 0.15 else 'Fair' if ibs < 0.25 else 'Poor'
            metrics_data.append(['IBS', f"{ibs:.3f}", interp])
        
        table = ax1.table(cellText=metrics_data, cellLoc='left', loc='center',
                         colWidths=[0.35, 0.25, 0.3])
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 3)
        
        for i in range(3):
            table[(0, i)].set_facecolor('#FF9800')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        ax1.set_title('Survival Model Performance', fontweight='bold', pad=20)
        
        # 2. Time-dependent AUC (if available)
        if 'time_dependent_auc' in metrics and isinstance(metrics['time_dependent_auc'], dict):
            ax2 = fig.add_subplot(gs[1, :])
            
            auc_dict = metrics['time_dependent_auc']
            
            # Parse timepoints - handle formats like '6mo', '12mo', '24mo' or numeric strings
            timepoints = []
            auc_values = []
            
            for key, value in auc_dict.items():
                # Extract numeric value from key
                if isinstance(key, (int, float)):
                    t = float(key)
                elif isinstance(key, str):
                    # Try to parse strings like '6mo', '12mo', '24mo'
                    import re
                    match = re.search(r'(\d+\.?\d*)', key)
                    if match:
                        t = float(match.group(1))
                    else:
                        continue  # Skip if we can't parse
                else:
                    continue
                
                timepoints.append(t)
                auc_values.append(value)
            
            # Sort by timepoint
            if timepoints:
                sorted_pairs = sorted(zip(timepoints, auc_values))
                timepoints, auc_values = zip(*sorted_pairs)
                
                ax2.plot(timepoints, auc_values, marker='o', linewidth=2, markersize=8, color='#2196F3')
                ax2.axhline(y=0.5, color='r', linestyle='--', label='Random', alpha=0.7)
                ax2.set_xlabel('Time Point (months)')
                ax2.set_ylabel('AUC')
                ax2.set_title('Time-Dependent AUC', fontweight='bold')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
                ax2.set_ylim([0.4, 1.0])
    
    def _create_feature_importance_page(
        self,
        pdf: PdfPages,
        feature_importance: Dict[str, float],
        task_type: str
    ):
        """Create page with feature importance analysis"""
        fig, axes = plt.subplots(2, 1, figsize=(11, 8.5))
        fig.suptitle('Feature Importance Analysis', fontsize=16, fontweight='bold', y=0.98)
        
        # Sort features by importance
        sorted_features = sorted(
            feature_importance.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        
        # Top 20 features
        top_n = min(20, len(sorted_features))
        top_features = sorted_features[:top_n]
        
        features = [f[0] for f in top_features]
        importances = [f[1] for f in top_features]
        
        # 1. Horizontal bar chart
        ax1 = axes[0]
        colors = ['#1f77b4' if x > 0 else '#ff7f0e' for x in importances]
        y_pos = np.arange(len(features))
        
        ax1.barh(y_pos, importances, color=colors, alpha=0.8)
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(features, fontsize=9)
        ax1.invert_yaxis()
        ax1.set_xlabel('Importance Score')
        ax1.set_title(f'Top {top_n} Most Important Features', fontweight='bold', pad=10)
        ax1.grid(True, alpha=0.3, axis='x')
        
        # 2. Cumulative importance
        ax2 = axes[1]
        cumsum = np.cumsum([abs(x) for x in importances])
        cumsum_normalized = cumsum / cumsum[-1] * 100
        
        ax2.plot(range(1, len(cumsum_normalized) + 1), cumsum_normalized,
                marker='o', linewidth=2, markersize=6)
        ax2.axhline(y=80, color='r', linestyle='--', label='80% threshold')
        ax2.axhline(y=95, color='orange', linestyle='--', label='95% threshold')
        ax2.set_xlabel('Number of Features')
        ax2.set_ylabel('Cumulative Importance (%)')
        ax2.set_title('Cumulative Feature Importance', fontweight='bold', pad=10)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0, 105])
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
    
    def _create_shap_pages(
        self,
        pdf: PdfPages,
        shap_values: Any,
        explainer: Any,
        X_test: pd.DataFrame,
        task_type: str
    ):
        """Create pages with SHAP analysis"""
        X_sample = X_test.sample(min(100, len(X_test)), random_state=42)
        
        # Page 1: SHAP Summary Plot
        fig, ax = plt.subplots(figsize=(11, 8.5))
        fig.suptitle('SHAP Summary Plot', fontsize=16, fontweight='bold', y=0.98)
        
        try:
            # Handle multiclass case
            if isinstance(shap_values, list):
                shap_values_plot = shap_values[0]  # Use first class
            else:
                shap_values_plot = shap_values
            
            shap.summary_plot(
                shap_values_plot, X_sample, show=False, max_display=20
            )
            
            # Add explanation text
            plt.figtext(0.5, 0.02, 
                       "Each dot represents a sample. Red = high feature value, Blue = low feature value.\n"
                       "Position shows impact on prediction.",
                       ha='center', fontsize=9, style='italic',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        except Exception as e:
            ax.text(0.5, 0.5, f"Could not generate SHAP summary plot:\n{str(e)}",
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # Page 2: SHAP Feature Importance (bar plot)
        fig, ax = plt.subplots(figsize=(11, 8.5))
        fig.suptitle('SHAP Feature Importance', fontsize=16, fontweight='bold', y=0.98)
        
        try:
            if isinstance(shap_values, list):
                shap_values_plot = shap_values[0]
            else:
                shap_values_plot = shap_values
            
            shap.summary_plot(
                shap_values_plot, X_sample, plot_type="bar", show=False, max_display=20
            )
            
            plt.figtext(0.5, 0.02,
                       "Mean absolute SHAP value shows average impact of each feature on predictions.",
                       ha='center', fontsize=9, style='italic',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        except Exception as e:
            ax.text(0.5, 0.5, f"Could not generate SHAP bar plot:\n{str(e)}",
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
    
    def _create_prediction_distribution_page(
        self,
        pdf: PdfPages,
        y_test: np.ndarray,
        y_pred: np.ndarray,
        task_type: str
    ):
        """Create page showing prediction distributions"""
        fig = plt.figure(figsize=(11, 8.5))
        fig.suptitle('Prediction Distribution Analysis', fontsize=16, fontweight='bold', y=0.98)
        
        if task_type == 'survival':
            # Special handling for survival data
            self._plot_survival_distributions(fig, y_test, y_pred)
            
        elif task_type == 'classification':
            # Class distribution comparison
            gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3,
                                 left=0.1, right=0.95, top=0.92, bottom=0.08)
            
            ax1 = fig.add_subplot(gs[0, :])
            unique_classes = np.unique(np.concatenate([y_test, y_pred]))
            true_dist = [np.sum(y_test == c) for c in unique_classes]
            pred_dist = [np.sum(y_pred == c) for c in unique_classes]
            
            x = np.arange(len(unique_classes))
            width = 0.35
            ax1.bar(x - width/2, true_dist, width, label='True', alpha=0.8, color='#2196F3')
            ax1.bar(x + width/2, pred_dist, width, label='Predicted', alpha=0.8, color='#FF9800')
            ax1.set_xlabel('Class')
            ax1.set_ylabel('Count')
            ax1.set_title('Class Distribution Comparison', fontweight='bold')
            ax1.set_xticks(x)
            ax1.set_xticklabels(unique_classes)
            ax1.legend()
            ax1.grid(True, alpha=0.3, axis='y')
            
            # Agreement analysis
            ax2 = fig.add_subplot(gs[1, :])
            agreement = (y_test == y_pred)
            agreement_pct = agreement.sum() / len(agreement) * 100
            
            labels = ['Correct', 'Incorrect']
            sizes = [agreement.sum(), (~agreement).sum()]
            colors = ['#4CAF50', '#F44336']
            explode = (0.1, 0)
            
            ax2.pie(sizes, explode=explode, labels=labels, colors=colors,
                   autopct='%1.1f%%', shadow=True, startangle=90, textprops={'fontsize': 12})
            ax2.set_title(f'Prediction Accuracy: {agreement_pct:.1f}%', fontweight='bold')
            
        else:  # regression
            gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3,
                                 left=0.1, right=0.95, top=0.92, bottom=0.08)
            
            # True values distribution
            ax1 = fig.add_subplot(gs[0, 0])
            ax1.hist(y_test, bins=30, alpha=0.7, color='#2196F3', edgecolor='black')
            ax1.set_xlabel('Value')
            ax1.set_ylabel('Frequency')
            ax1.set_title('True Values Distribution', fontweight='bold')
            ax1.grid(True, alpha=0.3, axis='y')
            
            # Predicted values distribution
            ax2 = fig.add_subplot(gs[0, 1])
            ax2.hist(y_pred, bins=30, alpha=0.7, color='#FF9800', edgecolor='black')
            ax2.set_xlabel('Value')
            ax2.set_ylabel('Frequency')
            ax2.set_title('Predicted Values Distribution', fontweight='bold')
            ax2.grid(True, alpha=0.3, axis='y')
            
            # Error distribution
            ax3 = fig.add_subplot(gs[1, :])
            errors = y_test - y_pred
            ax3.hist(errors, bins=30, alpha=0.7, color='#9C27B0', edgecolor='black')
            ax3.axvline(x=0, color='r', linestyle='--', linewidth=2, label='Zero Error')
            ax3.set_xlabel('Prediction Error (True - Predicted)')
            ax3.set_ylabel('Frequency')
            ax3.set_title('Prediction Error Distribution', fontweight='bold')
            ax3.legend()
            ax3.grid(True, alpha=0.3, axis='y')
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
    
    def _plot_survival_distributions(
        self, fig, y_test: np.ndarray, y_pred: np.ndarray
    ):
        """Plot survival-specific distributions"""
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3,
                             left=0.1, right=0.95, top=0.92, bottom=0.08)
        
        # Extract time and event from structured array
        if hasattr(y_test, 'dtype') and y_test.dtype.names:
            # Structured array with ('event', 'time') fields
            times = y_test['time']
            events = y_test['event']
        else:
            # Fallback: assume it's just times
            times = y_test
            events = np.ones(len(y_test), dtype=bool)
        
        # 1. Survival Time Distribution (censored vs events)
        ax1 = fig.add_subplot(gs[0, 0])
        event_times = times[events]
        censored_times = times[~events]
        
        bins = np.linspace(times.min(), times.max(), 30)
        ax1.hist(event_times, bins=bins, alpha=0.7, color='#F44336', 
                edgecolor='black', label=f'Events (n={len(event_times)})')
        ax1.hist(censored_times, bins=bins, alpha=0.7, color='#2196F3',
                edgecolor='black', label=f'Censored (n={len(censored_times)})')
        ax1.set_xlabel('Survival Time')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Survival Time Distribution', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
        
        # 2. Risk Score Distribution
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.hist(y_pred, bins=30, alpha=0.7, color='#FF9800', edgecolor='black')
        ax2.set_xlabel('Risk Score')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Predicted Risk Score Distribution', fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # 3. Risk Score by Event Status
        ax3 = fig.add_subplot(gs[1, 0])
        event_risks = y_pred[events]
        censored_risks = y_pred[~events]
        
        ax3.boxplot([event_risks, censored_risks], 
                   labels=['Event', 'Censored'],
                   patch_artist=True,
                   boxprops=dict(facecolor='#FF9800', alpha=0.7),
                   medianprops=dict(color='red', linewidth=2))
        ax3.set_ylabel('Risk Score')
        ax3.set_title('Risk Scores by Event Status', fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')
        
        # 4. Event Rate Summary
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.axis('off')
        
        n_events = events.sum()
        n_censored = (~events).sum()
        event_rate = n_events / len(events) * 100
        
        summary_data = [
            ['Metric', 'Value'],
            ['Total Patients', f'{len(events)}'],
            ['Events', f'{n_events}'],
            ['Censored', f'{n_censored}'],
            ['Event Rate', f'{event_rate:.1f}%'],
            ['Median Time', f'{np.median(times):.1f}'],
            ['Mean Risk Score', f'{np.mean(y_pred):.3f}']
        ]
        
        table = ax4.table(cellText=summary_data, cellLoc='left', loc='center',
                         colWidths=[0.5, 0.3])
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 2.5)
        
        for i in range(2):
            table[(0, i)].set_facecolor('#FF9800')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        ax4.set_title('Summary Statistics', fontweight='bold', pad=20)
    
    def _create_clinical_guidance_page(
        self,
        pdf: PdfPages,
        model_name: str,
        metrics: Dict[str, Any],
        task_type: str,
        feature_importance: Optional[Dict[str, float]]
    ):
        """Create clinical decision guidance page"""
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis('off')
        
        # Title
        ax.text(0.5, 0.95, 'Clinical Decision Support Guidance',
               ha='center', va='top', fontsize=18, fontweight='bold',
               transform=ax.transAxes)
        
        y_pos = 0.88
        
        # Model Overview
        ax.text(0.05, y_pos, '📋 Model Overview', fontsize=14, fontweight='bold',
               transform=ax.transAxes)
        y_pos -= 0.05
        
        overview_text = f"Model Type: {model_name}\n"
        overview_text += f"Task: {task_type.capitalize()}\n"
        
        if task_type == 'classification':
            overview_text += f"Accuracy: {metrics.get('accuracy', 0):.1%}\n"
            overview_text += f"F1-Score: {metrics.get('f1', 0):.3f}"
        elif task_type == 'survival':
            overview_text += f"C-Index: {metrics.get('concordance_index', 0):.3f}\n"
        
        ax.text(0.08, y_pos, overview_text, fontsize=11, va='top',
               transform=ax.transAxes, family='monospace',
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
        y_pos -= 0.15
        
        # Key Predictors
        if feature_importance:
            ax.text(0.05, y_pos, '🔑 Key Predictive Features', fontsize=14,
                   fontweight='bold', transform=ax.transAxes)
            y_pos -= 0.05
            
            top_features = sorted(feature_importance.items(),
                                key=lambda x: abs(x[1]), reverse=True)[:5]
            
            features_text = "Most influential factors for predictions:\n"
            for i, (feat, imp) in enumerate(top_features, 1):
                features_text += f"  {i}. {feat} (importance: {abs(imp):.3f})\n"
            
            ax.text(0.08, y_pos, features_text, fontsize=11, va='top',
                   transform=ax.transAxes, family='monospace',
                   bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
            y_pos -= 0.15
        
        # Clinical Recommendations
        ax.text(0.05, y_pos, '⚕️ Clinical Recommendations', fontsize=14,
               fontweight='bold', transform=ax.transAxes)
        y_pos -= 0.05
        
        if task_type == 'classification':
            recommendations = """
1. Use predictions as decision support, not sole decision criteria
2. Review misclassified cases for potential data quality issues
3. Consider model confidence when making high-stakes decisions
4. Regularly validate model performance on new data
5. Document all clinical decisions informed by model predictions
            """
        elif task_type == 'survival':
            recommendations = """
1. Use survival predictions to inform treatment planning
2. Consider individual risk stratification for patient management
3. Integrate predictions with clinical staging and biomarkers
4. Monitor model calibration over time
5. Use confidence intervals when discussing prognosis with patients
            """
        else:
            recommendations = """
1. Validate predictions against clinical expectations
2. Use model to identify patients who may benefit from intervention
3. Consider prediction uncertainty in clinical decisions
4. Regular model retraining with new data is recommended
            """
        
        ax.text(0.08, y_pos, recommendations, fontsize=10, va='top',
               transform=ax.transAxes,
               bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.3))
        y_pos -= 0.22
        
        # Limitations
        ax.text(0.05, y_pos, '⚠️ Important Limitations', fontsize=14,
               fontweight='bold', transform=ax.transAxes, color='darkred')
        y_pos -= 0.05
        
        limitations = """
• Model trained on specific population - may not generalize to all patients
• Predictions are probabilistic and subject to uncertainty
• Cannot replace clinical expertise and judgment
• Should be used as one tool among many in clinical decision-making
• Regular performance monitoring and updates are essential
        """
        
        ax.text(0.08, y_pos, limitations, fontsize=10, va='top',
               transform=ax.transAxes,
               bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.3))
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

