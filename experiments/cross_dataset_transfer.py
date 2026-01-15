#!/usr/bin/env python3
"""
Cross-Dataset Transfer Learning Experiment for ACE Framework

This experiment demonstrates that ACE learns generalizable strategies across cancer types
and these strategies transfer to improve performance on new datasets.

Experimental Design:
====================

Phase 1: Sequential Learning (Datasets 1-4)
  For each dataset:
    1. Baseline: Basic preprocessing + single model
    2. Self-Improvement: N iterations of improvements (feature eng, model selection, ensembling)
    3. ACE Reflection: Record lessons in playbook
    4. Playbook accumulates knowledge across datasets

Phase 2: Transfer Evaluation (Dataset 5)
  A. Control Condition:
     - Fresh agent with EMPTY playbook
     - Run baseline + self-improvement
     - Measures performance WITHOUT transfer learning
  
  B. Experimental Condition:
     - Fresh agent with ACCUMULATED playbook from datasets 1-4
     - Run baseline + self-improvement
     - Measures performance WITH transfer learning
  
  C. Comparison:
     - Initial performance (baseline)
     - Final performance (after self-improvement)
     - Convergence speed (iterations to plateau)
     - Strategies used (which improvements were tried)

All datasets: Survival analysis on 5 different cancer types
"""

import asyncio
import sys
import os
from pathlib import Path
import json
import time
import shutil
from datetime import datetime
from typing import Dict, Any, List, Optional
import warnings

warnings.filterwarnings('ignore', category=UserWarning, module='pydantic')

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.config import Config
from src.agents import ACEMLAgent


# ============================================================================
# EXPERIMENT CONFIGURATION
# ============================================================================

DATASETS = [
    {
        "name": "breast",
        "train": "data_ace/msk_breast_train.csv",
        "test": "data_ace/msk_breast_test.csv",
        "cancer_type": "breast cancer",
        "description": "Breast cancer survival"
    },
    {
        "name": "prostate",
        "train": "data_ace/msk_prostate_train.csv",
        "test": "data_ace/msk_prostate_test.csv",
        "cancer_type": "prostate cancer",
        "description": "Prostate cancer survival"
    },
    {
        "name": "lung",
        "train": "data_ace/msk_lung_train.csv",
        "test": "data_ace/msk_lung_test.csv",
        "cancer_type": "lung cancer",
        "description": "Lung cancer survival"
    },
    {
        "name": "pancreas",
        "train": "data_ace/msk_pancreas_train.csv",
        "test": "data_ace/msk_pancreas_test.csv",
        "cancer_type": "pancreatic cancer",
        "description": "Pancreatic cancer survival"
    },
    {
        "name": "colorectal",
        "train": "data_ace/msk_colo_train.csv",
        "test": "data_ace/msk_colo_test.csv",
        "cancer_type": "colorectal cancer",
        "description": "Colorectal cancer survival"
    }
]

EXPERIMENT_CONFIG = {
    "n_improvement_iterations": 5,
    "model": "gpt-4o-mini",
    "results_dir": "experiments/results",
    "playbook_checkpoint_dir": "experiments/playbook_checkpoints"
}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

async def run_baseline_phase(agent: ACEMLAgent, dataset: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run baseline phase: Minimal agent capabilities
    Returns baseline metrics
    """
    print("\n" + "-"*80)
    print("BASELINE PHASE")
    print("-"*80)
    
    # Analyze data
    print("[1/4] Analyzing data...")
    await agent.chat(
        "Analyze the dataset. Identify target variable, task type, and basic characteristics. "
        "Do NOT perform detailed EDA yet."
    )
    
    # Basic preprocessing
    print("[2/4] Basic preprocessing...")
    fe_result = await agent.toolkit._engineer_features(
        scaling_strategy="standard",
        encoding_strategy="onehot",
        handle_imbalance=False,
        basic_only=True  # KEY: Only basic preprocessing
    )
    if "error" not in fe_result:
        print(f"  {fe_result.get('n_features', 0)} features after preprocessing")
    
    # Train single model
    print("[3/4] Training baseline model...")
    await agent.chat(
        "Preprocessing is complete. Select and train ONE survival model. "
        "Do NOT run feature engineering again - just train ONE model."
    )
    
    baseline_score = agent.toolkit.state.get("best_score", 0.0)
    baseline_model = agent.toolkit.state.get("best_model", "unknown")
    
    # Evaluate on test
    print("[4/4] Evaluating baseline...")
    if baseline_model and baseline_model != "unknown":
        eval_result = await agent.toolkit._evaluate_model(baseline_model)
        test_score = eval_result.get("primary_score", 0.0)
    else:
        await agent.chat("Evaluate the model on test set")
        test_score = agent.toolkit.state.get("test_score", 0.0)
    
    print(f"\nBaseline: {baseline_model}, CV={baseline_score:.4f}, Test={test_score:.4f}")
    
    return {
        "cv_score": float(baseline_score),
        "test_score": float(test_score),
        "model": baseline_model
    }


async def run_self_improvement_phase(
    agent: ACEMLAgent,
    dataset: Dict[str, Any],
    n_iterations: int
) -> Dict[str, Any]:
    """
    Run iterative enhancement phase: Agent iteratively enhances performance
    Returns enhancement trajectory and final metrics
    """
    print("\n" + "-"*80)
    print(f"ITERATIVE ENHANCEMENT PHASE ({n_iterations} iterations)")
    print("-"*80)
    
    trajectory = []
    current_test_score = agent.toolkit.state.get("test_score", 0.0)
    
    for iteration in range(1, n_iterations + 1):
        print(f"\n--- Iteration {iteration}/{n_iterations} ---")
        
        current_best = agent.toolkit.state.get("best_score", 0.0)
        current_model = agent.toolkit.state.get("best_model", "unknown")
        
        # Track models before agent acts
        models_before = set(agent.toolkit.state.get("trained_models", {}).keys())
        
        # Agent chooses and implements enhancement strategy (avoid "improve" keyword!)
        await agent.chat(
            f"Iteration {iteration}: Current best CV={current_best:.4f}. "
            f"Preprocessing is done. Choose ONE enhancement:\n"
            f"1. Feature Engineering: Use get_data_insights + refine_features to ADD new features\n"
            f"   (DO NOT call engineer_features - it redoes preprocessing)\n"
            f"2. Model Selection: Train a different survival model\n"
            f"3. Ensembling: Create ensemble if you have 2+ models\n"
            f"Explain your choice and implement it."
        )
        
        new_score = agent.toolkit.state.get("best_score", current_best)
        new_best_model = agent.toolkit.state.get("best_model", current_model)
        
        # Find newly trained or retrained models
        models_after = set(agent.toolkit.state.get("trained_models", {}).keys())
        newly_trained = models_after - models_before
        retrained = models_after & models_before
        
        # Evaluate models
        if newly_trained:
            for model_name in newly_trained:
                eval_result = await agent.toolkit._evaluate_model(model_name)
                test_perf = eval_result.get("primary_score", 0.0)
                cv_perf = agent.toolkit.state.get("trained_models", {}).get(model_name, {}).get("cv_score", 0.0)
                print(f"  {model_name}: CV={cv_perf:.4f}, Test={test_perf:.4f}")
        
        if retrained:
            for model_name in retrained:
                eval_result = await agent.toolkit._evaluate_model(model_name)
                test_perf = eval_result.get("primary_score", 0.0)
                cv_perf = agent.toolkit.state.get("trained_models", {}).get(model_name, {}).get("cv_score", 0.0)
                print(f"  {model_name} (retrained): CV={cv_perf:.4f}, Test={test_perf:.4f}")
        
        # Use best model's test score
        if new_best_model and new_best_model in agent.toolkit.state.get("trained_models", {}):
            best_result = await agent.toolkit._evaluate_model(new_best_model)
            new_test_score = best_result.get("primary_score", current_test_score)
        else:
            new_test_score = current_test_score
        
        cv_improvement = new_score - current_best
        test_improvement = new_test_score - current_test_score
        print(f"Result: CV={new_score:.4f} (Δ{cv_improvement:+.4f}), Test={new_test_score:.4f} (Δ{test_improvement:+.4f})")
        
        trajectory.append({
            "iteration": iteration,
            "cv_score": float(new_score),
            "test_score": float(new_test_score),
            "cv_improvement": float(cv_improvement),
            "test_improvement": float(test_improvement),
            "model": new_best_model
        })
        
        current_test_score = new_test_score
        
        # Trigger reflection
        if agent.ace_enabled:
            if not agent.trajectory_generator.current_trajectory:
                await agent._start_new_trajectory()
            await agent._trigger_reflection()
    
    # Final evaluation
    final_score = agent.toolkit.state.get("best_score", 0.0)
    final_model = agent.toolkit.state.get("best_model", "unknown")
    
    if final_model and final_model in agent.toolkit.state.get("trained_models", {}):
        eval_result = await agent.toolkit._evaluate_model(final_model)
        final_test_score = eval_result.get("primary_score", current_test_score)
    else:
        final_test_score = current_test_score
    
    print(f"\nFinal: {final_model}, CV={final_score:.4f}, Test={final_test_score:.4f}")
    
    return {
        "final_cv_score": float(final_score),
        "final_test_score": float(final_test_score),
        "final_model": final_model,
        "trajectory": trajectory
    }


# ============================================================================
# EXPERIMENT RUNNER
# ============================================================================

class CrossDatasetExperiment:
    """Manages the cross-dataset transfer learning experiment"""
    
    def __init__(self, api_key: str, seed: int = 42):
        self.api_key = api_key
        self.seed = seed
        self.n_iterations = EXPERIMENT_CONFIG["n_improvement_iterations"]
        
        # Directories
        self.results_dir = Path(EXPERIMENT_CONFIG["results_dir"]) / f"seed_{seed}"
        self.checkpoint_dir = Path(EXPERIMENT_CONFIG["playbook_checkpoint_dir"]) / f"seed_{seed}"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Results storage
        self.sequential_results = []
        self.control_results = None
        self.experimental_results = None
        
        self.experiment_start = datetime.now()
    
    def create_agent(self, playbook_path: Optional[str] = None) -> ACEMLAgent:
        """Create fresh ACE agent with specified playbook"""
        os.environ["OPENAI_API_KEY"] = self.api_key
        
        config = Config.from_env()
        config.llm.model = EXPERIMENT_CONFIG["model"]
        
        # ACE configuration
        config.ace.enabled = True
        config.ace.max_improvement_iterations = self.n_iterations
        config.ace.auto_reflect = True
        config.ace.auto_save_playbook = True
        
        if playbook_path:
            config.ace.playbook_path = playbook_path
        
        config.ml.single_model_mode = False  # Allow multiple models
        config.data.random_state = self.seed
        
        return ACEMLAgent(config)
    
    async def run_dataset(
        self,
        dataset: Dict[str, Any],
        agent: ACEMLAgent,
        phase: str
    ) -> Dict[str, Any]:
        """
        Run complete pipeline on one dataset: baseline + self-improvement
        """
        dataset_name = dataset["name"]
        print(f"\n{'='*80}")
        print(f"DATASET: {dataset_name.upper()} | PHASE: {phase}")
        print(f"{'='*80}")
        
        # Set dataset
        agent.set_dataset(
            dataset_path=dataset["train"],
            testset_path=dataset["test"],
            objective=f"Survival analysis for {dataset['description']}"
        )
        
        start_time = time.time()
        
        # Run baseline
        baseline_results = await run_baseline_phase(agent, dataset)
        
        # Run self-improvement
        improvement_results = await run_self_improvement_phase(agent, dataset, self.n_iterations)
        
        total_time = time.time() - start_time
        
        # Get playbook stats
        playbook_stats = agent.curator.get_summary() if agent.ace_enabled else {}
        
        # Print summary
        print(f"\n{'='*80}")
        print(f"RESULTS: {dataset_name.upper()}")
        print(f"{'='*80}")
        print(f"Baseline: CV={baseline_results['cv_score']:.4f}, Test={baseline_results['test_score']:.4f}")
        print(f"Final:    CV={improvement_results['final_cv_score']:.4f}, Test={improvement_results['final_test_score']:.4f}")
        cv_gain = improvement_results['final_cv_score'] - baseline_results['cv_score']
        test_gain = improvement_results['final_test_score'] - baseline_results['test_score']
        print(f"Gain:     CV={cv_gain:+.4f}, Test={test_gain:+.4f}")
        print(f"Time:     {total_time:.1f}s")
        print(f"Playbook: {playbook_stats.get('total_items', 0)} items")
        print(f"{'='*80}\n")
        
        return {
            "dataset": dataset_name,
            "phase": phase,
            "baseline": baseline_results,
            "improvement": improvement_results,
            "total_improvement": {
                "cv_gain": float(cv_gain),
                "test_gain": float(test_gain)
            },
            "time_seconds": float(total_time),
            "playbook": playbook_stats,
            "seed": self.seed
        }
    
    async def run_phase1_sequential_learning(self):
        """Phase 1: Sequential learning on datasets 1-4"""
        print("\n" + "="*80)
        print("PHASE 1: SEQUENTIAL LEARNING (Datasets 1-4)")
        print("="*80)
        print("Goal: Accumulate knowledge in playbook across datasets")
        print("="*80 + "\n")
        
        # Start with empty playbook
        playbook_path = str(self.checkpoint_dir / "playbook_sequential.json")
        if Path(playbook_path).exists():
            Path(playbook_path).unlink()
        
        agent = self.create_agent(playbook_path=playbook_path)
        
        # Run on first 4 datasets
        for i, dataset in enumerate(DATASETS[:4], 1):
            print(f"\n>>> SEQUENTIAL DATASET {i}/4: {dataset['name'].upper()} <<<")
            
            result = await self.run_dataset(
                dataset=dataset,
                agent=agent,
                phase=f"sequential_{i}"
            )
            
            self.sequential_results.append(result)
            
            # Save checkpoint
            agent.save_session()
            checkpoint_file = self.checkpoint_dir / f"checkpoint_after_{dataset['name']}.json"
            shutil.copy(playbook_path, checkpoint_file)
            print(f"Playbook checkpoint saved: {checkpoint_file}")
            
            # Save intermediate results
            self._save_results("phase1_partial")
        
        final_playbook_items = agent.curator.playbook.total_items
        print("\n" + "="*80)
        print("PHASE 1 COMPLETE")
        print("="*80)
        print(f"Trained on: {[d['name'] for d in DATASETS[:4]]}")
        print(f"Playbook contains: {final_playbook_items} items")
        print("="*80 + "\n")
    
    async def run_phase2a_control(self):
        """Phase 2A: Control - Dataset 5 with empty playbook"""
        print("\n" + "="*80)
        print("PHASE 2A: CONTROL (Dataset 5 - No Transfer Learning)")
        print("="*80)
        print("Goal: Baseline performance WITHOUT learned knowledge")
        print("="*80 + "\n")
        
        # Create agent with empty playbook
        control_playbook_path = str(self.checkpoint_dir / "playbook_control.json")
        if Path(control_playbook_path).exists():
            Path(control_playbook_path).unlink()
        
        agent = self.create_agent(playbook_path=control_playbook_path)
        
        dataset = DATASETS[4]  # 5th dataset
        print(f">>> CONTROL: {dataset['name'].upper()} <<<")
        
        result = await self.run_dataset(
            dataset=dataset,
            agent=agent,
            phase="control"
        )
        
        self.control_results = result
        agent.save_session()
        self._save_results("phase2a_with_control")
        
        print("\n" + "="*80)
        print("CONTROL COMPLETE")
        print("="*80 + "\n")
    
    async def run_phase2b_experimental(self):
        """Phase 2B: Experimental - Dataset 5 with accumulated playbook"""
        print("\n" + "="*80)
        print("PHASE 2B: EXPERIMENTAL (Dataset 5 - With Transfer Learning)")
        print("="*80)
        print("Goal: Test if learned knowledge improves performance")
        print("="*80 + "\n")
        
        # Load accumulated playbook from sequential learning
        sequential_playbook_path = str(self.checkpoint_dir / "playbook_sequential.json")
        
        if not Path(sequential_playbook_path).exists():
            raise FileNotFoundError(f"Sequential playbook not found: {sequential_playbook_path}")
        
        agent = self.create_agent(playbook_path=sequential_playbook_path)
        
        playbook_items = agent.curator.playbook.total_items
        print(f"Loaded playbook with {playbook_items} items from datasets 1-4")
        
        dataset = DATASETS[4]  # 5th dataset
        print(f"\n>>> EXPERIMENTAL: {dataset['name'].upper()} <<<")
        
        result = await self.run_dataset(
            dataset=dataset,
            agent=agent,
            phase="experimental"
        )
        
        self.experimental_results = result
        agent.save_session()
        self._save_results("phase2b_final")
        
        print("\n" + "="*80)
        print("EXPERIMENTAL COMPLETE")
        print("="*80 + "\n")
    
    def _save_results(self, stage: str):
        """Save results to JSON"""
        results = {
            "experiment_config": EXPERIMENT_CONFIG,
            "datasets": DATASETS,
            "seed": self.seed,
            "experiment_start": self.experiment_start.isoformat(),
            "stage": stage,
            "sequential_results": self.sequential_results,
            "control_results": self.control_results,
            "experimental_results": self.experimental_results
        }
        
        output_file = self.results_dir / f"results_{stage}.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"Results saved: {output_file}")
    
    def print_final_comparison(self):
        """Print final comparison table"""
        print("\n" + "="*80)
        print("FINAL COMPARISON: CONTROL vs EXPERIMENTAL (Dataset 5)")
        print("="*80)
        
        if not self.control_results or not self.experimental_results:
            print("ERROR: Missing results")
            return
        
        ctrl = self.control_results
        exp = self.experimental_results
        
        dataset = DATASETS[4]
        print(f"\nDataset: {dataset['name']} ({dataset['description']})")
        
        # Table header
        print(f"\n{'Metric':<30} {'Control':<15} {'Experimental':<15} {'Benefit':<15}")
        print("-" * 80)
        
        # Baseline scores
        print(f"{'Baseline CV':<30} {ctrl['baseline']['cv_score']:<15.4f} "
              f"{exp['baseline']['cv_score']:<15.4f} "
              f"{exp['baseline']['cv_score'] - ctrl['baseline']['cv_score']:<+15.4f}")
        print(f"{'Baseline Test':<30} {ctrl['baseline']['test_score']:<15.4f} "
              f"{exp['baseline']['test_score']:<15.4f} "
              f"{exp['baseline']['test_score'] - ctrl['baseline']['test_score']:<+15.4f}")
        
        # Final scores
        print(f"{'Final CV':<30} {ctrl['improvement']['final_cv_score']:<15.4f} "
              f"{exp['improvement']['final_cv_score']:<15.4f} "
              f"{exp['improvement']['final_cv_score'] - ctrl['improvement']['final_cv_score']:<+15.4f}")
        print(f"{'Final Test':<30} {ctrl['improvement']['final_test_score']:<15.4f} "
              f"{exp['improvement']['final_test_score']:<15.4f} "
              f"{exp['improvement']['final_test_score'] - ctrl['improvement']['final_test_score']:<+15.4f}")
        
        # Total improvement achieved
        print(f"{'Total CV Gain':<30} {ctrl['total_improvement']['cv_gain']:<+15.4f} "
              f"{exp['total_improvement']['cv_gain']:<+15.4f} "
              f"{exp['total_improvement']['cv_gain'] - ctrl['total_improvement']['cv_gain']:<+15.4f}")
        print(f"{'Total Test Gain':<30} {ctrl['total_improvement']['test_gain']:<+15.4f} "
              f"{exp['total_improvement']['test_gain']:<+15.4f} "
              f"{exp['total_improvement']['test_gain'] - ctrl['total_improvement']['test_gain']:<+15.4f}")
        
        # Time
        print(f"{'Time (seconds)':<30} {ctrl['time_seconds']:<15.1f} "
              f"{exp['time_seconds']:<15.1f} "
              f"{exp['time_seconds'] - ctrl['time_seconds']:<+15.1f}")
        
        # Playbook
        ctrl_items = ctrl['playbook'].get('total_items', 0)
        exp_items = exp['playbook'].get('total_items', 0)
        print(f"{'Playbook Items':<30} {ctrl_items:<15} "
              f"{exp_items:<15} {exp_items - ctrl_items:<+15}")
        
        print("=" * 80)
        
        # Key findings
        baseline_benefit = exp['baseline']['test_score'] - ctrl['baseline']['test_score']
        final_benefit = exp['improvement']['final_test_score'] - ctrl['improvement']['final_test_score']
        convergence_benefit = exp['total_improvement']['test_gain'] - ctrl['total_improvement']['test_gain']
        
        print(f"\nKEY FINDINGS:")
        print(f"  1. Transfer Learning Effect (Baseline):")
        print(f"     Experimental started {baseline_benefit:+.4f} better than control")
        print(f"     This demonstrates knowledge transfer from prior datasets")
        
        print(f"\n  2. Final Performance Benefit:")
        print(f"     Experimental finished {final_benefit:+.4f} better than control")
        print(f"     Final test scores: {exp['improvement']['final_test_score']:.4f} vs "
              f"{ctrl['improvement']['final_test_score']:.4f}")
        
        print(f"\n  3. Improvement Efficiency:")
        print(f"     Experimental improved {convergence_benefit:+.4f} {'more' if convergence_benefit > 0 else 'less'} than control")
        print(f"     Total improvements: {exp['total_improvement']['test_gain']:+.4f} vs "
              f"{ctrl['total_improvement']['test_gain']:+.4f}")
        
        print(f"\n  4. Knowledge Accumulation:")
        print(f"     Playbook grew from 0 to {exp_items} items through 5 datasets")
        print(f"     Control accumulated {ctrl_items} items (single dataset)")
        
        print("=" * 80 + "\n")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

async def run_full_experiment(api_key: str, seed: int = 42):
    """Run complete experiment: sequential learning + control + experimental"""
    
    experiment = CrossDatasetExperiment(api_key, seed)
    
    print("\n" + "="*80)
    print("CROSS-DATASET TRANSFER LEARNING EXPERIMENT")
    print("="*80)
    print(f"Seed: {seed}")
    print(f"Model: {EXPERIMENT_CONFIG['model']}")
    print(f"Improvement Iterations: {EXPERIMENT_CONFIG['n_improvement_iterations']}")
    print(f"Results Directory: {experiment.results_dir}")
    print("="*80 + "\n")
    
    try:
        # Phase 1: Sequential learning on datasets 1-4
        await experiment.run_phase1_sequential_learning()
        
        # Phase 2A: Control on dataset 5 (no transfer)
        await experiment.run_phase2a_control()
        
        # Phase 2B: Experimental on dataset 5 (with transfer)
        await experiment.run_phase2b_experimental()
        
        # Final comparison
        experiment.print_final_comparison()
        
        print("\nEXPERIMENT COMPLETE!")
        print(f"Results saved to: {experiment.results_dir}")
        
    except Exception as e:
        print(f"\nEXPERIMENT FAILED: {e}")
        import traceback
        traceback.print_exc()
        raise


def main():
    """Entry point"""
    print("\n" + "="*80)
    print("ACE CROSS-DATASET TRANSFER LEARNING EXPERIMENT")
    print("="*80)
    
    api_key = input("\nEnter OpenAI API key: ").strip()
    if not api_key:
        print("ERROR: API key required")
        return
    
    seed_input = input("Enter random seed (default 42): ").strip()
    seed = int(seed_input) if seed_input else 42
    
    print(f"\nStarting experiment with seed {seed}...")
    print(f"This will train 5 datasets with {EXPERIMENT_CONFIG['n_improvement_iterations']} improvement iterations each")
    print("Estimated time: 4-8 hours (depending on dataset sizes and improvements)")
    print("\nDatasets:")
    for i, ds in enumerate(DATASETS, 1):
        print(f"  {i}. {ds['name']} ({ds['description']})")
    print("\nPress Ctrl+C to cancel\n")
    
    asyncio.run(run_full_experiment(api_key, seed))


if __name__ == "__main__":
    main()
