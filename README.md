# Tabular ML Agent Framework

An intelligent agentic framework for automated tabular machine learning using LangChain, GPT-4o, and ReAct architecture.

## Overview

This framework provides an agentic ML automation where the ReAct agent autonomously builds ML models by:
- Reasoning about what to do next based on results
- Calling tools dynamically via LangChain
- Analyzing errors and iterating to improve
- Stopping when performance is satisfactory
- Adapting strategy based on data characteristics

## Quick Start

### Installation

```bash
cd tabular_ml_agent
pip install -e .
```

### Basic Usage

**Run the Agent:**
```bash
python main.py
```

You'll be prompted for:
- OpenAI API key
- Dataset path
- ML objective
- Test set path (optional)
- Agent type (Standard or With Reflection)

### Example Usage

```python
import asyncio
from src.core.config import Config
from src.agents import ReActMLAgent

async def run_agent():
    config = Config.from_env()
    agent = ReActMLAgent(config)
    
    result = await agent.run(
        dataset_path="your_dataset.csv",
        objective="Predict target variable",
        max_iterations=20
    )
    
    print(f"Best model: {result['best_model']}")
    print(f"Best score: {result['best_score']:.3f}")
    print(f"Models trained: {result['trained_models']}")

asyncio.run(run_agent())
```

## Features

### Supported Task Types
- **Classification**: Binary and multi-class with accuracy, F1, ROC-AUC
  - **AutoGluon Integration**: Uses AutoML ensemble by default (trains and combines multiple models automatically)
- **Regression**: Continuous prediction with R², MAE, RMSE
- **Survival Analysis**: Time-to-event with C-index, IBS, time-dependent AUC

### Agent Capabilities
- **Autonomous decision-making**: Decides which models to train and when to stop
- **LangChain integration**: Standardized tool calling and observability
- **Error analysis**: Deep dive into prediction failures
- **Feature importance**: Understand what drives predictions
- **Reflection mode**: Additional reasoning step for improvement
- **Conversation logging**: Full trace of agent reasoning

### ML Components
- **Data analysis**: Automatic target detection and task type identification
- **Feature engineering**: Preprocessing, encoding, scaling
- **Model selection**: Task-appropriate model recommendations
- **Hyperparameter optimization**: Optuna-based tuning
- **Cross-validation**: Stratified or preset CV groups
- **Risk stratification**: For survival analysis

## Architecture

```
tabular_ml_agent/
├── src/
│   ├── agents/                     # ReAct Agent (LangChain)
│   │   ├── langchain_react_agent.py   # Main agent orchestrator
│   │   ├── langchain_tools.py         # Tool wrappers for LangChain
│   │   ├── tools.py                   # ML toolkit (8 tools)
│   │   └── error_analyzer.py          # Error analysis
│   ├── core/                       # Configuration & state
│   ├── llm/                        # GPT-4o interface
│   ├── data/                       # Data analysis
│   │   └── analyzer.py
│   └── ml/                         # ML pipeline
│       ├── feature_engineer.py
│       ├── model_selector.py
│       └── trainer.py
├── configs/                        # Configuration files
├── main.py                         # Main entry point
└── outputs/
    ├── react_conversations/        # Agent logs
    └── evaluations/                # Model metrics
```

## How It Works

### ReAct Loop

The agent follows a Think → Act → Observe cycle:

1. **Think**: LLM reasons about current state → decides next action
2. **Act**: Agent calls a tool (e.g., `train_model`)
3. **Observe**: Agent sees result → updates strategy
4. **Repeat**: Until goal achieved or max iterations

### Process Flow

```
1. analyze_data       → Identify target, task type, features
2. engineer_features  → Preprocess, encode, scale, split data
3. select_models      → Choose appropriate models for task
4. train_model        → Train with hyperparameter optimization
5. evaluate_model     → Test on held-out set
6. get_feature_importance → Understand predictions
7. analyze_errors     → (if needed) Identify failure patterns
8. iterate or finish  → Based on performance
```

### Available Tools

| Tool | Purpose |
|------|---------|
| `analyze_data` | Dataset structure, target identification, task type |
| `engineer_features` | Preprocessing, encoding, scaling, data splitting |
| `select_models` | Model recommendations based on task & data |
| `train_model` | Train specific model with hyperparameter tuning |
| `evaluate_model` | Test set evaluation with task-specific metrics |
| `analyze_errors` | Deep dive into prediction errors |
| `get_feature_importance` | Feature importance rankings |
| `get_current_state` | Check pipeline progress |

## Configuration

### Environment Variables

```bash
export OPENAI_API_KEY="your-key-here"
export LLM_MODEL="gpt-4o-mini"  # or "gpt-4o"
export LOG_LEVEL="INFO"
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
  models_to_try:
    - autogluon      # AutoML ensemble (classification only)
    - catboost
    - xgboost
    - random_forest
    - cox_ph         # for survival

data:
  test_size: 0.2
  val_size: 0.2
  random_state: 42
  use_preset_CV: false
```

## AutoGluon for Classification

For classification tasks, the agent uses **AutoGluon** by default, which:
- Trains multiple models in parallel (LightGBM, CatBoost, XGBoost, Random Forest, etc.)
- Automatically tunes hyperparameters
- Creates weighted ensembles for better performance
- Requires no manual configuration

### Installation

```bash
pip install autogluon.tabular
```

### Benefits
- Better performance: Ensemble of models typically outperforms individual models
- Less work: No need to train and compare multiple models manually
- Automatic tuning: Handles hyperparameter optimization internally

### Fallback to Individual Models
To train individual models instead of AutoGluon:
```python
# Select specific models (skips autogluon)
result = await toolkit._select_models(prefer_simple=True)
# Then train: xgboost, catboost, logistic_regression, etc.
```

## Output Files

| Output | Location | Content |
|--------|----------|---------|
| Conversation log | `outputs/react_conversations/` | Full agent reasoning trace |
| Evaluation metrics | `outputs/evaluations/` | Performance metrics for all models |
| Trained models | In memory via `feature_result` | Serialized models & preprocessors |

## Model Support

### Classification
- **AutoGluon** (default) - AutoML ensemble that automatically trains and combines multiple models
- Logistic Regression
- Random Forest
- XGBoost
- LightGBM
- CatBoost

### Regression
- Linear Regression
- Ridge
- Random Forest
- XGBoost
- LightGBM

### Survival Analysis
- Cox Proportional Hazards
- Random Survival Forest
- Gradient Boosting Survival

## Agent Variants

### Standard Agent
```python
from src.agents import ReActMLAgent

agent = ReActMLAgent(config)
result = await agent.run(dataset_path, objective)
```

### Reflection Agent
Adds reflection step when performance < 0.75:

```python
from src.agents import ReActAgentWithReflection

agent = ReActAgentWithReflection(config)
result = await agent.run(dataset_path, objective)
```

## Advanced Features

### Pre-split Test Set
```python
result = await agent.run(
    dataset_path="train.csv",
    testset_path="test.csv",
    objective="Predict survival"
)
```

### Preset Cross-Validation Groups
```python
# Dataset must have 'CV' column with fold numbers (0,1,2,3,4)
result = await agent.run(
    dataset_path="data_with_cv_column.csv",
    objective="Predict outcome",
    use_preset_CV=True
)
```

### Control Iterations
```python
result = await agent.run(
    dataset_path="data.csv",
    objective="Classify",
    max_iterations=10
)
```

## LangChain Benefits

The agent uses LangChain for:
- Standardized tools: Compatible with LangChain ecosystem
- Better error handling: Automatic parsing error recovery
- Observability ready: Easy LangSmith integration
- Callbacks: Custom monitoring and logging
- Extensibility: Easy to add new tools

### Enable LangSmith Tracing

```bash
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_API_KEY="your-langsmith-key"
export LANGCHAIN_PROJECT="tabular-ml-agent"

python main.py
```

Visit https://smith.langchain.com to see execution traces, token usage, and costs.

## Troubleshooting

### Agent doesn't call certain tools

Tools are called based on LLM reasoning. To make tools more likely to be called, edit the prompt in `src/agents/langchain_react_agent.py` under `_create_prompt()`.

### Performance issues

- Reduce `max_iterations` for faster runs
- Use `quick_mode=true` in `train_model` tool
- Reduce `optuna_trials` in config

### API errors

- Check OpenAI API key is set
- Verify model name is correct (`gpt-4o-mini` or `gpt-4o`)
- Check rate limits

## Contributing

The agent is designed to be extensible:

1. **Add new tools**: Create in `langchain_tools.py`
2. **Modify prompts**: Edit `_create_prompt()` in `langchain_react_agent.py`
3. **Add models**: Extend `trainer.py` with new model types
4. **Custom callbacks**: Add to agent initialization


