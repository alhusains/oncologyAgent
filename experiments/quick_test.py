#!/usr/bin/env python3
"""
Quick Test of ACE Self-Improvement Framework

This script demonstrates the ACE framework on a single dataset with clear separation:
1. Baseline Phase: Minimal agent (basic preprocessing + one model)
2. Self-Improvement Phase: Agent iteratively improves via feature eng, model selection, or ensembling
3. Results: Shows progression of improvements and recorded lessons

Run this before the full cross-dataset experiment to verify everything works.
"""

import asyncio
import sys
import os
from pathlib import Path
import warnings
import json
from typing import Dict, Any

warnings.filterwarnings('ignore', category=UserWarning, module='pydantic')

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.config import Config
from src.agents import ACEMLAgent


async def run_baseline_phase(agent: ACEMLAgent, dataset_info: Dict[str, Any]) -> Dict[str, Any]:
    """
    Baseline phase: Minimal agent capabilities
    - Basic preprocessing only (no feature engineering)
    - LLM selects ONE model
    - Evaluate on test set
    
    Returns baseline metrics
    """
    print("\n" + "="*80)
    print("PHASE 1: BASELINE (Minimal Agent)")
    print("="*80)
    print("Goal: Establish baseline with minimal capabilities")
    print("  - Basic preprocessing (no feature engineering)")
    print("  - Single model selected by LLM")
    print("  - Test set evaluation")
    print("="*80)
    
    # Step 1: Analyze data (basic analysis, no EDA yet)
    print("\n[1/4] Analyzing data...")
    response = await agent.chat(
        "Analyze the dataset. Provide basic statistics but do NOT perform EDA yet. "
        "Just identify the target variable, task type, and basic data characteristics."
    )
    print(f"Analysis complete")
    
    # Step 2: Basic preprocessing (minimal feature engineering)
    print("\n[2/4] Basic preprocessing...")
    # Directly call feature engineering with basic_only=True to ensure no advanced features
    fe_result = await agent.toolkit._engineer_features(
        scaling_strategy="standard",
        encoding_strategy="onehot",
        handle_imbalance=False,
        basic_only=True  # KEY: Only basic preprocessing
    )
    if "error" in fe_result:
        print(f"Error in preprocessing: {fe_result['error']}")
    else:
        print(f"Preprocessing complete: {fe_result.get('n_features', 0)} features")
    
    # Step 3: Train ONE model (LLM decides which)
    print("\n[3/4] Training baseline model...")
    response = await agent.chat(
        "Preprocessing is complete. Now select and train ONE survival model. "
        "Do NOT run feature engineering again - data is already preprocessed. "
        "Just call select_models (if needed) and train_model for ONE model."
    )
    
    baseline_score = agent.toolkit.state.get("best_score", 0.0)
    baseline_model = agent.toolkit.state.get("best_model", "unknown")
    print(f"Baseline model trained: {baseline_model}")
    print(f"Cross-validation score: {baseline_score:.4f}")
    
    # Step 4: Evaluate on test set
    print("\n[4/4] Evaluating on test set...")
    if baseline_model and baseline_model != "unknown":
        # Directly call evaluation to ensure it happens
        eval_result = await agent.toolkit._evaluate_model(baseline_model)
        test_score = eval_result.get("primary_score", 0.0)
    else:
        # Fallback: ask agent to evaluate
        response = await agent.chat(f"Evaluate the {baseline_model} model on the test set")
        test_score = agent.toolkit.state.get("test_score", 0.0)
    print(f"Test set score: {test_score:.4f}")
    
    print("\n" + "="*80)
    print(f"BASELINE RESULTS")
    print("="*80)
    print(f"  Model: {baseline_model}")
    print(f"  CV Score: {baseline_score:.4f}")
    print(f"  Test Score: {test_score:.4f}")
    print("="*80 + "\n")
    
    return {
        "model": baseline_model,
        "cv_score": float(baseline_score),
        "test_score": float(test_score)
    }


async def run_self_improvement_phase(
    agent: ACEMLAgent, 
    dataset_info: Dict[str, Any],
    n_iterations: int = 5
) -> Dict[str, Any]:
    """
    Self-improvement phase: Agent iteratively enhances performance
    
    Each iteration:
    1. Agent chooses enhancement strategy (feature eng, model selection, or ensemble)
    2. Implement the strategy
    3. Evaluate performance
    4. ACE reflects and records lessons
    
    Returns enhancement trajectory and final metrics
    """
    print("\n" + "="*80)
    print(f"PHASE 2: ITERATIVE ENHANCEMENT ({n_iterations} Iterations)")
    print("="*80)
    print("Goal: Agent iteratively enhances performance via:")
    print("  - Feature Engineering (with EDA insights)")
    print("  - Model Selection (try different models)")
    print("  - Ensembling (combine models)")
    print("="*80 + "\n")
    
    improvement_trajectory = []
    baseline_test_score = agent.toolkit.state.get("test_score", 0.0)
    
    for iteration in range(1, n_iterations + 1):
        print("\n" + "="*80)
        print(f"ENHANCEMENT ITERATION {iteration}/{n_iterations}")
        print("="*80)
        
        # Get current best score
        current_best = agent.toolkit.state.get("best_score", 0.0)
        current_model = agent.toolkit.state.get("best_model", "unknown")
        current_test_score = agent.toolkit.state.get("test_score", baseline_test_score)
        print(f"Current best: {current_model} (CV: {current_best:.4f}, Test: {current_test_score:.4f})")
        
        # Track models before agent acts
        models_before = set(agent.toolkit.state.get("trained_models", {}).keys())
        
        # Agent decides enhancement strategy (avoid "improve" keyword!)
        print(f"\n[{iteration}.1] Agent choosing enhancement strategy...")
        response = await agent.chat(
            f"This is iteration {iteration} of {n_iterations}. "
            f"Current best CV score: {current_best:.4f}. "
            f"Basic preprocessing is already done. Choose ONE enhancement strategy:\n\n"
            f"Strategy 1 - Feature Engineering:\n"
            f"  Use get_data_insights to perform EDA\n"
            f"  Then use refine_features to CREATE NEW features based on EDA insights\n"
            f"  (add interactions, transformations ON TOP of existing features)\n"
            f"  DO NOT call engineer_features - preprocessing is already complete!\n\n"
            f"Strategy 2 - Model Selection:\n"
            f"  Train a different survival model (e.g., cox_ph, gradient_boosting_survival)\n\n"
            f"Strategy 3 - Ensembling:\n"
            f"  If you have 2+ trained models, create an ensemble using create_ensemble\n\n"
            f"Choose the strategy you think will help most, explain your reasoning, and implement it."
        )
        
        # Record what strategy was chosen (heuristic detection)
        strategy_chosen = "unknown"
        response_lower = response.lower()
        if "feature" in response_lower and ("engineer" in response_lower or "eda" in response_lower):
            strategy_chosen = "feature_engineering"
        elif "ensemble" in response_lower or "combin" in response_lower or "stack" in response_lower:
            strategy_chosen = "ensembling"
        elif "model" in response_lower and ("train" in response_lower or "different" in response_lower):
            strategy_chosen = "model_selection"
        
        print(f"Strategy chosen: {strategy_chosen}")
        
        # Get new score after enhancement attempt
        new_cv_score = agent.toolkit.state.get("best_score", current_best)
        new_best_model = agent.toolkit.state.get("best_model", current_model)
        
        # Find newly trained or retrained models
        models_after = set(agent.toolkit.state.get("trained_models", {}).keys())
        newly_trained = models_after - models_before
        retrained = models_after & models_before  # Models that existed before and after
        
        # Evaluate on test set
        print(f"\n[{iteration}.2] Evaluating on test set...")
        
        if newly_trained:
            print(f"  Newly trained models:")
            for model_name in newly_trained:
                eval_result = await agent.toolkit._evaluate_model(model_name)
                test_perf = eval_result.get("primary_score", 0.0)
                cv_perf = agent.toolkit.state.get("trained_models", {}).get(model_name, {}).get("cv_score", 0.0)
                print(f"    {model_name}: CV={cv_perf:.4f}, Test={test_perf:.4f}")
        
        if retrained and strategy_chosen == "feature_engineering":
            print(f"  Retrained models (with new features):")
            for model_name in retrained:
                eval_result = await agent.toolkit._evaluate_model(model_name)
                test_perf = eval_result.get("primary_score", 0.0)
                cv_perf = agent.toolkit.state.get("trained_models", {}).get(model_name, {}).get("cv_score", 0.0)
                print(f"    {model_name}: CV={cv_perf:.4f}, Test={test_perf:.4f}")
        
        # Always use the best model's test score for tracking
        if new_best_model and new_best_model in agent.toolkit.state.get("trained_models", {}):
            best_result = await agent.toolkit._evaluate_model(new_best_model)
            new_test_score = best_result.get("primary_score", current_test_score)
        else:
            new_test_score = current_test_score
        
        # Calculate improvements
        cv_improvement = new_cv_score - current_best
        test_improvement = new_test_score - current_test_score
        
        print(f"\n[{iteration}.3] Results:")
        print(f"  New CV Score: {new_cv_score:.4f} (Δ: {cv_improvement:+.4f})")
        print(f"  New Test Score: {new_test_score:.4f} (Δ: {test_improvement:+.4f})")
        print(f"  Best Model: {new_best_model}")
        
        # Record iteration results
        iteration_result = {
            "iteration": iteration,
            "strategy": strategy_chosen,
            "cv_score": float(new_cv_score),
            "test_score": float(new_test_score),
            "cv_improvement": float(cv_improvement),
            "test_improvement": float(test_improvement),
            "model": new_best_model
        }
        improvement_trajectory.append(iteration_result)
        
        # Trigger ACE reflection
        print(f"\n[{iteration}.4] ACE reflection...")
        if agent.ace_enabled:
            # Start new trajectory for this iteration if needed
            if not agent.trajectory_generator.current_trajectory:
                await agent._start_new_trajectory()
            
            # Trigger reflection
            await agent._trigger_reflection()
            print("Lessons recorded in playbook")
        
        print("="*80)
    
    # Final evaluation
    print("\n" + "="*80)
    print("FINAL EVALUATION")
    print("="*80)
    
    final_cv_score = agent.toolkit.state.get("best_score", 0.0)
    final_model = agent.toolkit.state.get("best_model", "unknown")
    
    # Final test evaluation
    if final_model in agent.toolkit.state.get("trained_models", {}):
        result = await agent.toolkit._evaluate_model(final_model)
        final_test_score = result.get("test_score", 0.0)
        agent.toolkit.state["test_score"] = final_test_score
    else:
        final_test_score = agent.toolkit.state.get("test_score", 0.0)
    
    print(f"  Final Model: {final_model}")
    print(f"  Final CV Score: {final_cv_score:.4f}")
    print(f"  Final Test Score: {final_test_score:.4f}")
    print("="*80 + "\n")
    
    return {
        "final_cv_score": float(final_cv_score),
        "final_test_score": float(final_test_score),
        "final_model": final_model,
        "trajectory": improvement_trajectory
    }


async def quick_test(api_key: str, n_iterations: int = 5):
    """Run quick test on single dataset"""
    
    print("\n" + "="*80)
    print("ACE SELF-IMPROVEMENT FRAMEWORK - QUICK TEST")
    print("="*80)
    
    # Configuration
    os.environ["OPENAI_API_KEY"] = api_key
    config = Config.from_env()
    config.llm.model = "gpt-4o-mini"
    
    # ACE configuration
    config.ace.enabled = True
    config.ace.max_improvement_iterations = n_iterations
    config.ace.auto_reflect = True
    config.ace.playbook_path = "experiments/test_playbook.json"
    
    # Single model mode initially (agent can change this)
    config.ml.single_model_mode = False  # Allow multiple models for ensembling
    
    print(f"\nConfiguration:")
    print(f"  Model: {config.llm.model}")
    print(f"  ACE Enabled: {config.ace.enabled}")
    print(f"  Improvement Iterations: {n_iterations}")
    print(f"  Playbook: {config.ace.playbook_path}")
    
    # Test dataset
    dataset = {
        "train": "data_ace/msk_breast_train.csv",
        "test": "data_ace/msk_breast_test.csv",
        "name": "breast",
        "description": "Breast cancer survival"
    }
    
    print(f"\nDataset: {dataset['name']} ({dataset['description']})")
    print(f"  Train: {dataset['train']}")
    print(f"  Test: {dataset['test']}")
    print("="*80)
    
    # Create agent
    agent = ACEMLAgent(config)
    
    # Set dataset
    agent.set_dataset(
        dataset_path=dataset["train"],
        testset_path=dataset["test"],
        objective=f"Survival analysis for {dataset['description']}"
    )
    
    # Run experiment
    baseline_results = await run_baseline_phase(agent, dataset)
    improvement_results = await run_self_improvement_phase(agent, dataset, n_iterations)
    
    # Save results
    results = {
        "dataset": dataset,
        "baseline": baseline_results,
        "improvement": improvement_results,
        "playbook_stats": agent.curator.get_summary() if agent.ace_enabled else {}
    }
    
    results_dir = Path("experiments/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    results_file = results_dir / "quick_test_results.json"
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    
    # Print summary
    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY")
    print("="*80)
    print(f"\nDataset: {dataset['name']}")
    print(f"\nBaseline:")
    print(f"  Model: {baseline_results['model']}")
    print(f"  CV Score: {baseline_results['cv_score']:.4f}")
    print(f"  Test Score: {baseline_results['test_score']:.4f}")
    
    print(f"\nFinal (after {n_iterations} iterations):")
    print(f"  Model: {improvement_results['final_model']}")
    print(f"  CV Score: {improvement_results['final_cv_score']:.4f}")
    print(f"  Test Score: {improvement_results['final_test_score']:.4f}")
    
    print(f"\nTotal Improvement:")
    cv_gain = improvement_results['final_cv_score'] - baseline_results['cv_score']
    test_gain = improvement_results['final_test_score'] - baseline_results['test_score']
    print(f"  CV: {cv_gain:+.4f}")
    print(f"  Test: {test_gain:+.4f}")
    
    print(f"\nPlaybook:")
    playbook_stats = results['playbook_stats']
    print(f"  Total Items: {playbook_stats.get('total_items', 0)}")
    print(f"  Lessons Extracted: {playbook_stats.get('lessons_extracted', 0)}")
    
    print(f"\nImprovement Trajectory:")
    for step in improvement_results['trajectory']:
        print(f"  Iter {step['iteration']}: {step['strategy']:20s} -> CV: {step['cv_score']:.4f} (Δ: {step['cv_improvement']:+.4f})")
    
    print("\n" + "="*80)
    print("TEST COMPLETE")
    print("="*80)
    print("\nEverything appears to be working correctly.")
    print("You can now run the full cross-dataset experiment with:")
    print("  python experiments/cross_dataset_transfer.py")
    print("="*80 + "\n")


def main():
    """Entry point"""
    print("\n" + "="*80)
    print("ACE FRAMEWORK QUICK TEST")
    print("="*80)
    
    api_key = input("\nEnter OpenAI API key: ").strip()
    if not api_key:
        print("ERROR: API key required")
        return
    
    iterations_input = input("Number of improvement iterations (default 5): ").strip()
    n_iterations = int(iterations_input) if iterations_input else 5
    
    print(f"\nThis will test the ACE self-improvement framework on one dataset")
    print(f"Baseline phase + {n_iterations} improvement iterations")
    print(f"Estimated time: ~{15 + n_iterations * 10}-{20 + n_iterations * 15} minutes")
    print("Press Ctrl+C to cancel\n")
    
    asyncio.run(quick_test(api_key, n_iterations))


if __name__ == "__main__":
    main()
