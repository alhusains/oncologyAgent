# Oncology ML Agent

An AI-powered conversational agent for automated machine learning analysis of oncology datasets. The agent provides an interactive, step-by-step ML pipeline with self-improvement capabilities through the ACE (Agentic Context Engineering) framework.

## 🎯 Overview

This project implements an intelligent agent that conducts comprehensive machine learning analysis through natural language conversations. It's specifically designed for oncology research but applicable to any tabular ML task.

**Key Capabilities:**
- 💬 **Interactive ML Pipeline**: Conversational interface for complete ML workflow
- 🤖 **Self-Improving Agent**: Learns from experience and continuously improves (ACE framework)
- 📊 **Multiple Task Types**: Classification, regression, and survival analysis
- 🔬 **Comprehensive Analysis**: Data insights, feature engineering, model selection, and interpretability
- 🌐 **Web Interface**: Optional Gradio-based web UI for easy interaction

## ✨ Features

### Core ML Pipeline

The agent guides you through the complete ML workflow:

1. **Exploratory Data Analysis (EDA)**
   - Distribution analysis with outlier detection
   - Survival-specific insights (log-rank tests, event rate variance)
   - Feature importance estimation
   - Interaction detection (numerical pairs, categorical-numerical)
   - Clinical relevance assessment

2. **Feature Engineering**
   - Smart transformations based on EDA insights
   - Domain-specific features for oncology data
   - Polynomial interactions and binning strategies
   - Handles missing data intelligently

3. **Model Training & Selection**
   - Multiple algorithms: XGBoost, Random Forest, LightGBM, CatBoost, AutoGluon
   - Automated hyperparameter tuning with Optuna
   - Cross-validation with stratification
   - Model comparison and selection

4. **Advanced Ensembling**
   - Voting and weighted averaging
   - Stacking and blending (Kaggle-inspired)
   - Survival-specific ensemble strategies
   - Risk score combinations

5. **Model Evaluation & Interpretability**
   - Comprehensive performance metrics
   - SHAP analysis for feature importance
   - Error analysis and prediction insights
   - PDF report generation

### ACE Framework (Self-Improvement) ⚡

The ACE (Agentic Context Engineering) framework enables the agent to learn from experience:

- **Trajectory Tracking**: Records all actions, decisions, and outcomes
- **Playbook Learning**: Builds a knowledge base of successful strategies
- **Self-Improvement Loop**: Automatically iterates to find better configurations
- **Context-Aware Decisions**: Uses learned knowledge for future recommendations
- **Transfer Learning**: Knowledge transfers across cancer types and datasets

**How it Works:**

1. **Generator**: Records trajectories of experiments
2. **Reflector**: Analyzes results and extracts lessons
3. **Curator**: Maintains playbook with confidence scores and semantic similarity matching
4. **Controller**: Orchestrates improvement loops using accumulated knowledge

The agent starts with minimal seed knowledge and progressively builds expertise through experimentation. See experiments section for validation.

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd oncologyAgent

# Install dependencies
pip install -e .
# or
pip install -r requirements.txt
```

### Basic Usage

**Option 1: Command-Line Interface**

```bash
python main.py
```

The agent will prompt you for:
- OpenAI API key
- Training dataset path (.csv or .xlsx)
- Optional test dataset path
- Analysis objective

Then interact naturally:
```
You: Analyze the data and show me insights
Agent: [Performs comprehensive EDA with survival-specific analysis]

You: Create relevant features based on the insights
Agent: [Engineers features informed by EDA findings]

You: Train and compare multiple models
Agent: [Trains XGBoost, Random Forest, LightGBM, etc. and compares]

You: Generate an interpretability report
Agent: [Creates PDF with SHAP values and feature importance]
```

**Option 2: Web Interface (Gradio)**

```bash
python gradio_app.py
```

Then open your browser to `http://localhost:7860` for a ChatGPT-like interface.

### With ACE Self-Improvement

When ACE is enabled (default), the agent can improve itself:

```
You: Try to improve the model performance
Agent: [Runs improvement loop, tests changes, learns from results]
```

The agent will:
1. Analyze current baseline performance
2. Generate improvement hypotheses from playbook
3. Test changes through ablation experiments
4. Update playbook with successful strategies

## 🧪 Experiments & Validation

The project includes a comprehensive experimental framework to validate ACE's effectiveness through cross-dataset transfer learning.

### Experimental Design

**Phase 1: Baseline**
- Basic preprocessing (no feature engineering)
- Single model selected by LLM
- Test set evaluation

**Phase 2: Self-Improvement** (N iterations, default 5)
- Agent chooses strategy: Feature Engineering, Model Selection, or Ensembling
- Implement → Evaluate → ACE Reflects → Record Lessons
- Playbook accumulates knowledge

**Phase 3: Transfer Learning**
- Sequential learning on 4 cancer datasets (breast, prostate, lung, pancreas)
- Transfer evaluation on 5th dataset (colorectal)
- Control (empty playbook) vs Experimental (accumulated playbook)

### Running Experiments

```bash
# Quick test on one dataset (~1-2 hours)
python experiments/quick_test.py

# Full cross-dataset transfer experiment (~4-8 hours)
python experiments/cross_dataset_transfer.py

# Analyze results and generate figures
python experiments/analyze_results.py --plot
```

**Key Metrics Tracked:**
- Initial performance boost from transfer learning
- Final performance improvement
- Convergence speed (iterations to plateau)
- Knowledge accumulation (playbook entries)

## ⚙️ Configuration

### Using Config File

Edit `configs/default.yaml`:

```yaml
llm:
  model: "gpt-4o-mini"  # or gpt-4, gpt-4o, gpt-4-turbo
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

### Using Environment Variables

```bash
export OPENAI_API_KEY="your-key"
export LLM_MODEL="gpt-4o-mini"
export ML_N_JOBS=4
export ACE_ENABLED=true
```

## 📁 Project Structure

```
oncologyAgent/
├── src/
│   ├── agents/              # Agent implementations
│   │   ├── conversational_agent.py    # Main conversational agent
│   │   ├── ace_agent.py               # ACE-enhanced agent
│   │   ├── tools.py                   # Agent tool library
│   │   └── langchain_tools.py         # LangChain tool wrappers
│   ├── ace/                 # ACE framework components
│   │   ├── controller.py              # Orchestrates improvement loops
│   │   ├── generator.py               # Records trajectories
│   │   ├── reflector.py               # Extracts lessons
│   │   ├── curator.py                 # Manages playbook
│   │   └── schemas.py                 # Data structures
│   ├── ml/                  # ML pipeline components
│   │   ├── feature_engineer.py        # Feature engineering
│   │   ├── model_selector.py          # Model training & selection
│   │   ├── ensemble_builder.py        # Ensemble methods
│   │   └── trainer.py                 # Training orchestration
│   ├── llm/                 # LLM client and prompts
│   │   ├── client.py                  # OpenAI client wrapper
│   │   └── prompts.py                 # Prompt templates
│   ├── core/                # Base classes and config
│   │   ├── config.py                  # Configuration management
│   │   └── state.py                   # Agent state management
│   └── data/                # Data analysis utilities
│       └── analyzer.py                # EDA and data insights
├── experiments/             # Experimental framework
│   ├── quick_test.py                  # Single-dataset test
│   ├── cross_dataset_transfer.py      # Transfer learning experiment
│   └── analyze_results.py             # Results analysis & plotting
├── configs/                 # Configuration files
│   └── default.yaml
├── main.py                  # CLI entry point
├── gradio_app.py            # Web interface
├── pyproject.toml           # Package dependencies
└── README.md
```

## 🔧 Requirements

- Python 3.10+
- OpenAI API access

**Key Dependencies:**
- LangChain & LangChain-OpenAI
- Pandas, NumPy, Scikit-learn
- XGBoost, LightGBM, CatBoost, AutoGluon
- Scikit-survival (for survival analysis)
- SHAP, LIME (for interpretability)
- Optuna (hyperparameter optimization)
- Gradio (web interface)

See `pyproject.toml` for complete list.

## 💡 Use Cases

- **Clinical Trial Analysis**: Automated analysis of oncology clinical trial data
- **Biomarker Discovery**: ML pipeline for cancer biomarker prediction
- **Survival Analysis**: Treatment outcome studies and time-to-event modeling
- **Rapid Prototyping**: Quick ML pipeline development for medical research
- **Transfer Learning**: Learn optimal strategies for specific cancer types

## 📝 Example Workflows

### Full Classification Pipeline

```
You: Conduct full classification analysis
Agent: [Runs complete pipeline: EDA → Feature Engineering → Model Training → Evaluation → Report]
```

### Step-by-Step Survival Analysis

```
You: Give me comprehensive data insights
Agent: [Performs EDA with survival-specific analysis]

You: Implement feature engineering based on these insights
Agent: [Creates features informed by EDA findings]

You: Train survival models
Agent: [Trains Cox PH, Random Survival Forest, etc.]

You: Evaluate on test set
Agent: [Computes C-index, time-dependent AUC, etc.]

You: Generate interpretability report
Agent: [Creates PDF with SHAP analysis and survival curves]
```

### Self-Improvement Loop

```
You: Try to improve performance
Agent: 
  Iteration 1: Trying feature engineering...
  → Improvement: +2.3% C-index
  Iteration 2: Trying model selection...
  → Improvement: +1.8% C-index
  Iteration 3: Trying ensemble methods...
  → Improvement: +3.1% C-index
  
  Playbook updated with 3 new strategies!
```

## 🤝 Contributing

Contributions are welcome! This project is designed for extensibility:

- Add new ML models in `src/ml/model_selector.py`
- Extend feature engineering in `src/ml/feature_engineer.py`
- Add new agent tools in `src/agents/tools.py`
- Improve ACE strategies in `src/ace/`

## 📄 License

MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- Built with LangChain for agent orchestration
- Powered by OpenAI's GPT models
- Inspired by Kaggle grandmaster ensemble techniques
- ACE framework adapted from recent advances in agentic AI

## 📞 Contact

For questions or collaboration opportunities, please open an issue on GitHub.

---

**Note**: This agent uses LLM-based decision making. While it strives for accuracy, always validate results critically, especially for clinical applications.
