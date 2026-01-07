# Oncology ML Agent

An AI-powered conversational agent for automated machine learning analysis of oncology datasets. The agent provides interactive, step-by-step ML pipeline execution with self-improvement capabilities through the ACE (Agentic Context Engineering) framework.

## Features

### Core Capabilities
- **Interactive ML Pipeline**: Conversational interface for data analysis, feature engineering, model training, and evaluation
- **Multiple Task Types**: Supports classification, regression, and survival analysis
- **Automated Model Selection**: Trains and compares multiple models (XGBoost, Random Forest, LightGBM, CatBoost, AutoGluon)
- **Comprehensive Analysis**: Includes data insights, feature importance, SHAP interpretability, and error analysis
- **Persistent State**: Maintains context and progress across conversation turns

### ACE Framework (Self-Improvement)
The ACE framework enables the agent to learn from experience and continuously improve:

- **Trajectory Tracking**: Records all actions, decisions, and outcomes from experiments
- **Playbook Learning**: Builds a knowledge base of strategies that worked (or didn't work) for specific contexts
- **Self-Improvement Loop**: Can automatically iterate to find better model configurations
- **Context-Aware Decisions**: Uses learned knowledge to inform future recommendations
- **Ablation Testing**: Validates improvements through rigorous A/B testing

The ACE framework makes the agent truly agentic - it progressively learns what works for different cancer types, dataset sizes, and task types, rather than relying on hardcoded heuristics.

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd oncologyAgent

# Install dependencies
pip install -r requirements.txt  # or use pyproject.toml
```

## Quick Start

### Basic Usage (Conversational Agent)

```bash
python main.py
```

The agent will prompt you for:
- OpenAI API key
- Training dataset path
- Optional test dataset path
- Analysis objective

Then you can interact naturally:
```
You: Analyze the data and show me insights
Agent: [Performs data analysis and presents findings]

You: Create some relevant features
Agent: [Engineers features based on data characteristics]

You: Train and evaluate models
Agent: [Trains multiple models, compares performance]
```

### With ACE Framework (Self-Improvement)

When ACE is enabled (default), the agent can improve itself:

```
You: Try to improve the model performance
Agent: [Runs improvement loop, tests changes, learns from results]
```

The agent will:
1. Analyze the current baseline model
2. Generate improvement hypotheses based on learned knowledge
3. Test changes through ablation experiments
4. Update its playbook with successful strategies

### Configuration

Edit `configs/default.yaml` to customize:

```yaml
llm:
  model: "gpt-4o-mini"  # or gpt-4, gpt-4-turbo, etc.
  temperature: 0.1

ml:
  cv_folds: 5
  optuna_trials: 100
  models_to_try: ["xgboost", "random_forest", "lightgbm", "catboost", "autogluon"]

ace:
  enabled: true
  max_improvement_iterations: 3
  auto_reflect: true
```

Or use environment variables:
```bash
export OPENAI_API_KEY="your-key"
export LLM_MODEL="gpt-4o-mini"
export ML_N_JOBS=4
```

## Project Structure

```
oncologyAgent/
├── src/
│   ├── agents/          # Agent implementations
│   │   ├── conversational_agent.py
│   │   ├── ace_agent.py
│   │   └── tools.py
│   ├── ace/             # ACE framework components
│   │   ├── controller.py
│   │   ├── generator.py
│   │   ├── reflector.py
│   │   ├── curator.py
│   │   └── schemas.py
│   ├── core/            # Base classes and configuration
│   ├── llm/             # LLM client and prompts
│   └── ml/              # ML pipeline components
├── configs/             # Configuration files
├── knowledge/           # Playbook storage (auto-generated)
├── main.py              # Entry point
└── README.md
```

## How ACE Works

The ACE framework consists of four components:

1. **Generator**: Records trajectories of all agent actions and experiments
2. **Reflector**: Analyzes trajectories to extract lessons about what worked
3. **Curator**: Maintains a playbook of learned strategies with confidence scores
4. **Controller**: Orchestrates improvement loops using playbook knowledge

The agent starts with minimal seed knowledge and progressively builds expertise through experimentation. For example:
- Initially: 6 general bootstrap strategies
- After 5 breast cancer experiments: 15+ cancer-specific strategies learned
- Knowledge includes: successful feature interactions, model configurations, common pitfalls

## API Key

You'll need an OpenAI API key. Set it via:
- Interactive prompt (entered during execution)
- Environment variable: `export OPENAI_API_KEY="your-key"`
- Config file: Add to `configs/default.yaml`

## Requirements

- Python 3.8+
- OpenAI API access
- Key packages: langchain, langgraph, pandas, scikit-learn, xgboost, lightgbm, catboost, autogluon, shap, optuna

## Use Cases

- Exploratory analysis of oncology clinical trial data
- Automated model development for cancer biomarker prediction
- Survival analysis for treatment outcome studies
- Rapid prototyping of ML pipelines for medical research
- Learning optimal strategies for specific cancer types
