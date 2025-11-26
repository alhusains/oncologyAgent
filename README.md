# Oncology ML Agent

An AI-powered conversational agent for automated machine learning analysis of oncology datasets. Built on LangChain and GPT-4, this system provides interactive, step-by-step ML pipeline execution with comprehensive interpretability reporting.

## Overview

The Oncology ML Agent is a conversational AI assistant that guides users through machine learning workflows for clinical data analysis. It supports classification, regression, and survival analysis tasks with full interpretability and clinical decision support.

### Key Features

- **Interactive Conversation**: Multi-turn dialogue allowing precise control over each analysis step
- **Comprehensive Data Analysis**: Automated dataset profiling with quality assessment and clinical insights
- **Automated ML Pipeline**: Feature engineering, model selection, training, and evaluation
- **Interpretability Reports**: PDF reports with SHAP analysis, feature importance, and clinical guidance
- **Multiple Task Types**: Classification, regression, and survival analysis support
- **Session Management**: Save and resume analysis sessions

## Installation

### Requirements

- Python >= 3.10
- OpenAI API key

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd oncologyAgent

# Install dependencies
pip install -e .
```

### Dependencies

The system requires:
- **LangChain**: Agent framework and tool orchestration
- **OpenAI GPT-4**: Language model for reasoning and decision-making
- **Scikit-learn**: Machine learning algorithms
- **Scikit-survival**: Survival analysis models
- **AutoGluon**: AutoML ensemble models
- **SHAP**: Model interpretability
- **Matplotlib/Seaborn**: Visualization and reporting

Full dependency list available in `pyproject.toml`.

## Quick Start

### Running the Agent

```bash
python main.py
```

You will be prompted for:
1. OpenAI API key
2. Training dataset path
3. Test dataset path (optional)
4. Analysis objective

### Example Session

```
You: analyze the data

Agent: [Provides comprehensive dataset analysis]

You: implement feature engineering

Agent: [Engineers features and reports results]

You: train classification models

Agent: [Trains multiple models and shows performance]

You: generate interpretability report

Agent: [Creates PDF report with SHAP analysis]
```

## Usage

### Basic Workflow

1. **Start Interactive Session**
   ```bash
   python main.py
   ```

2. **Configure Dataset**
   - Provide training data path
   - Optionally provide separate test set
   - Specify analysis objective

3. **Interact with Agent**
   - Request specific analysis steps
   - Execute full pipeline
   - Generate reports

4. **Save Results**
   - Type `save` to save session
   - Type `summary` to view progress
   - Type `exit` to end session

### Programmatic Usage

```python
from src.agents import ConversationalMLAgent
from src.core.config import Config
import os

# Configure
os.environ["OPENAI_API_KEY"] = "your-key"
config = Config.from_env()

# Initialize agent
agent = ConversationalMLAgent(config)
agent.set_dataset("data/train.csv", "data/test.csv", "Classification analysis")

# Interact
response = await agent.chat("Give me data insights")
response = await agent.chat("Train classification models")
response = await agent.chat("Generate interpretability report")

# Save session
agent.save_session()
```

## Features

### Data Analysis

The agent provides comprehensive dataset analysis including:
- Dataset overview and statistics
- Missing data patterns and quality assessment
- Target variable distribution and balance
- Feature correlations and relationships
- Outlier detection
- Clinical data insights

Request with: `"analyze the data"` or `"give me data insights"`

### Feature Engineering

Automated feature engineering with:
- Missing value imputation
- Categorical encoding (one-hot, label)
- Numerical scaling (standard, min-max, robust)
- Feature selection
- Class balancing (for classification)
- Risk-stratified splitting (for survival)

Request with: `"implement feature engineering"`

### Model Training

Supports multiple model types:

**Classification:**
- AutoGluon (AutoML ensemble)
- XGBoost, CatBoost, LightGBM
- Random Forest
- Logistic Regression

**Regression:**
- Linear Regression, Ridge
- XGBoost, LightGBM
- Random Forest

**Survival Analysis:**
- Cox Proportional Hazards
- Random Survival Forest
- Gradient Boosting Survival Analysis

Request with: `"train a random forest model"` or `"train all classification models"`

### Model Evaluation

Task-specific evaluation metrics:
- **Classification**: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- **Regression**: R², MAE, RMSE, MAPE
- **Survival**: C-Index, Integrated Brier Score, Time-Dependent AUC

Request with: `"evaluate the model"`

### Interpretability Reports

Comprehensive PDF reports including:
- Model performance metrics and visualizations
- Feature importance rankings
- SHAP analysis (summary plots, feature contributions)
- Prediction distributions
- ROC curves (classification)
- Time-dependent AUC (survival)
- Clinical decision support guidance

Request with: `"generate interpretability report"`

Reports are saved to `outputs/interpretability_reports/`

## Architecture

### System Components

```
oncologyAgent/
├── src/
│   ├── agents/
│   │   ├── conversational_agent.py  # Multi-turn chat agent
│   │   ├── tools.py                 # ML pipeline tools
│   │   ├── data_insights.py         # Data analysis
│   │   └── interpretability.py      # Report generation
│   ├── core/
│   │   ├── config.py                # Configuration management
│   │   └── state.py                 # State tracking
│   ├── llm/
│   │   └── client.py                # LLM interface
│   ├── data/
│   │   └── analyzer.py              # Dataset analysis
│   └── ml/
│       ├── feature_engineer.py      # Feature engineering
│       ├── model_selector.py        # Model selection
│       └── trainer.py               # Model training
├── main.py                          # Entry point
└── outputs/
    ├── interpretability_reports/    # PDF reports
    └── chat_sessions/               # Saved sessions
```

### Agent Tools

The agent has access to 10 specialized tools:

1. **analyze_data**: Dataset structure and target identification
2. **engineer_features**: Preprocessing and feature transformation
3. **select_models**: Model recommendations
4. **train_model**: Model training with hyperparameter optimization
5. **evaluate_model**: Test set evaluation
6. **analyze_errors**: Error pattern analysis
7. **get_feature_importance**: Feature importance rankings
8. **get_current_state**: Progress tracking
9. **get_data_insights**: Comprehensive data analysis
10. **generate_interpretability_report**: PDF report generation

## Configuration

### Environment Variables

```bash
export OPENAI_API_KEY="your-key-here"
export LLM_MODEL="gpt-4o-mini"
```

### Configuration File

Edit `configs/default.yaml`:

```yaml
llm:
  model: "gpt-4o-mini"
  temperature: 0.1

ml:
  cv_folds: 5
  optuna_trials: 100

data:
  test_size: 0.2
  val_size: 0.2
  random_state: 42
```

## Advanced Features

### Session Management

Sessions can be saved and include:
- Full conversation history
- ML pipeline state
- Trained models and results
- Analysis metadata

Sessions are saved to `outputs/chat_sessions/`

### Pre-split Test Sets

```python
agent.set_dataset(
    dataset_path="train.csv",
    testset_path="test.csv",  # Optional separate test set
    objective="Classification"
)
```

### Preset Cross-Validation

Use preset CV folds from dataset:

```python
# Dataset must have 'CV' column with fold numbers
agent.set_dataset(
    dataset_path="data_with_cv.csv",
    objective="Classification",
    use_preset_CV=True
)
```

## Output Files

### Interpretability Reports

Location: `outputs/interpretability_reports/`

Format: `{model_name}_{task_type}_{timestamp}.pdf`

Content:
- Title page with model overview
- Performance metrics dashboard
- Feature importance plots
- SHAP analysis visualizations
- Prediction distributions
- Clinical decision guidance

### Session Files

Location: `outputs/chat_sessions/`

Format: `session_{id}_{timestamp}.json`

Content:
- Conversation history
- ML pipeline state
- Model results
- Session metadata

## Model Support

### Task Types

**Classification**: Binary and multi-class classification with:
- Confusion matrices
- ROC curves and AUC
- Precision, recall, F1-score

**Regression**: Continuous value prediction with:
- R² and error metrics
- Residual analysis
- Prediction vs actual plots

**Survival Analysis**: Time-to-event analysis with:
- C-index and concordance
- Integrated Brier Score
- Time-dependent AUC

### AutoGluon Integration

For classification tasks, AutoGluon provides:
- Automatic ensemble model creation
- Multiple model types trained in parallel
- Hyperparameter optimization
- Weighted ensemble predictions
- GPU acceleration support

## Clinical Decision Support

The interpretability reports include clinical guidance:
- Model reliability assessment
- Feature importance interpretation
- Prediction confidence analysis
- Usage recommendations
- Limitations and caveats

Reports are designed for clinical audiences with clear visualizations and actionable insights.

## Troubleshooting

### Common Issues

**Agent doesn't execute tools**
- Ensure dataset paths are correct
- Check that objective is clearly stated
- Verify OpenAI API key is valid

**SHAP analysis slow**
- Normal for AutoGluon models (30-60 seconds)
- Can disable with interpretability report options
- Faster for tree-based models (XGBoost, CatBoost)

**Missing dependencies**
- Run `pip install -e .` to install all requirements
- Check Python version >= 3.10

## Contributing

The agent framework is designed to be extensible:
- Add new tools in `src/agents/tools.py`
- Extend model support in `src/ml/trainer.py`
- Customize prompts in `src/agents/conversational_agent.py`
- Add new report sections in `src/agents/interpretability.py`

## License

MIT License - See LICENSE file for details

## Citation

If you use this work in your research, please cite:

```bibtex
@software{oncology_ml_agent,
  title={Oncology ML Agent: Interactive ML Pipeline for Clinical Data},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/oncologyAgent}
}
```

## Support

For issues, questions, or contributions:
- Open an issue on GitHub
- Contact: your.email@example.com

## Acknowledgments

Built with:
- LangChain for agent framework
- OpenAI GPT-4 for reasoning
- scikit-learn and scikit-survival for ML
- SHAP for interpretability
- AutoGluon for AutoML
