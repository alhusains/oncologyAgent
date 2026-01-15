#!/usr/bin/env python3
"""
Analyze Results from Cross-Dataset Transfer Learning Experiments

This script loads the experimental results and generates:
1. Performance comparison tables
2. Improvement trajectory visualizations
3. Statistical significance tests
4. Strategy effectiveness analysis
5. Summary statistics for ICML paper

Usage:
    python experiments/analyze_results.py [--seed SEED]
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List
import sys

# Try to import plotting libraries (optional)
try:
import matplotlib.pyplot as plt
    import numpy as np
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    print("Note: matplotlib not available. Plots will be skipped.")


def load_results(results_dir: Path) -> Dict[str, Any]:
    """Load experimental results from JSON file"""
    final_results_file = results_dir / "results_phase2b_final.json"
    
    if not final_results_file.exists():
        raise FileNotFoundError(f"Results file not found: {final_results_file}")
    
    with open(final_results_file, 'r') as f:
        return json.load(f)


def print_sequential_learning_summary(sequential_results: List[Dict[str, Any]]):
    """Print summary of sequential learning phase"""
    print("\n" + "="*80)
    print("SEQUENTIAL LEARNING SUMMARY (Datasets 1-4)")
    print("="*80)
    
    print(f"\n{'Dataset':<15} {'Baseline CV':<12} {'Final CV':<12} {'CV Gain':<12} "
          f"{'Test Gain':<12} {'Playbook Items':<15}")
    print("-" * 80)
    
    for result in sequential_results:
        dataset = result['dataset']
        baseline_cv = result['baseline']['cv_score']
        final_cv = result['improvement']['final_cv_score']
        cv_gain = result['total_improvement']['cv_gain']
        test_gain = result['total_improvement']['test_gain']
        playbook_items = result['playbook'].get('total_items', 0)
        
        print(f"{dataset:<15} {baseline_cv:<12.4f} {final_cv:<12.4f} "
              f"{cv_gain:<+12.4f} {test_gain:<+12.4f} {playbook_items:<15}")
    
    # Averages
    avg_cv_gain = sum(r['total_improvement']['cv_gain'] for r in sequential_results) / len(sequential_results)
    avg_test_gain = sum(r['total_improvement']['test_gain'] for r in sequential_results) / len(sequential_results)
    
    print("-" * 80)
    print(f"{'AVERAGE':<15} {'':<12} {'':<12} {avg_cv_gain:<+12.4f} {avg_test_gain:<+12.4f}")
    print("="*80)


def print_transfer_learning_analysis(control: Dict[str, Any], experimental: Dict[str, Any]):
    """Print detailed transfer learning analysis"""
    print("\n" + "="*80)
    print("TRANSFER LEARNING ANALYSIS (Dataset 5: Control vs Experimental)")
    print("="*80)
    
    # Baseline comparison
    print("\n1. INITIAL PERFORMANCE (Baseline Phase)")
    print("-" * 80)
    ctrl_baseline_cv = control['baseline']['cv_score']
    exp_baseline_cv = experimental['baseline']['cv_score']
    ctrl_baseline_test = control['baseline']['test_score']
    exp_baseline_test = experimental['baseline']['test_score']
    
    baseline_cv_benefit = exp_baseline_cv - ctrl_baseline_cv
    baseline_test_benefit = exp_baseline_test - ctrl_baseline_test
    
    print(f"Control Baseline:       CV={ctrl_baseline_cv:.4f}, Test={ctrl_baseline_test:.4f}")
    print(f"Experimental Baseline:  CV={exp_baseline_cv:.4f}, Test={exp_baseline_test:.4f}")
    print(f"Transfer Benefit:       CV={baseline_cv_benefit:+.4f}, Test={baseline_test_benefit:+.4f}")
    print(f"\nInterpretation: Experimental agent started {baseline_test_benefit:+.4f} better")
    print(f"                {'POSITIVE' if baseline_test_benefit > 0 else 'NEGATIVE'} transfer learning effect")
    
    # Final performance comparison
    print("\n2. FINAL PERFORMANCE (After Self-Improvement)")
    print("-" * 80)
    ctrl_final_cv = control['improvement']['final_cv_score']
    exp_final_cv = experimental['improvement']['final_cv_score']
    ctrl_final_test = control['improvement']['final_test_score']
    exp_final_test = experimental['improvement']['final_test_score']
    
    final_cv_benefit = exp_final_cv - ctrl_final_cv
    final_test_benefit = exp_final_test - ctrl_final_test
    
    print(f"Control Final:          CV={ctrl_final_cv:.4f}, Test={ctrl_final_test:.4f}")
    print(f"Experimental Final:     CV={exp_final_cv:.4f}, Test={exp_final_test:.4f}")
    print(f"Transfer Benefit:       CV={final_cv_benefit:+.4f}, Test={final_test_benefit:+.4f}")
    print(f"\nInterpretation: Experimental agent finished {final_test_benefit:+.4f} better")
    
    # Improvement trajectory comparison
    print("\n3. IMPROVEMENT EFFICIENCY")
    print("-" * 80)
    ctrl_cv_gain = control['total_improvement']['cv_gain']
    exp_cv_gain = experimental['total_improvement']['cv_gain']
    ctrl_test_gain = control['total_improvement']['test_gain']
    exp_test_gain = experimental['total_improvement']['test_gain']
    
    gain_difference = exp_test_gain - ctrl_test_gain
    
    print(f"Control Improvement:        CV={ctrl_cv_gain:+.4f}, Test={ctrl_test_gain:+.4f}")
    print(f"Experimental Improvement:   CV={exp_cv_gain:+.4f}, Test={exp_test_gain:+.4f}")
    print(f"Efficiency Difference:      CV={exp_cv_gain - ctrl_cv_gain:+.4f}, Test={gain_difference:+.4f}")
    print(f"\nInterpretation: Experimental {'improved' if gain_difference > 0 else 'did not improve'} "
          f"{abs(gain_difference):.4f} {'more' if gain_difference > 0 else 'less'}")
    
    # Time comparison
    print("\n4. COMPUTATIONAL EFFICIENCY")
    print("-" * 80)
    ctrl_time = control['time_seconds']
    exp_time = experimental['time_seconds']
    time_diff = exp_time - ctrl_time
    
    print(f"Control Time:        {ctrl_time:.1f}s ({ctrl_time/60:.1f} min)")
    print(f"Experimental Time:   {exp_time:.1f}s ({exp_time/60:.1f} min)")
    print(f"Time Difference:     {time_diff:+.1f}s ({time_diff/60:+.1f} min)")
    print(f"\nInterpretation: Experimental was {abs(time_diff):.1f}s "
          f"{'slower' if time_diff > 0 else 'faster'}")
    
    # Playbook knowledge
    print("\n5. KNOWLEDGE ACCUMULATION")
    print("-" * 80)
    ctrl_items = control['playbook'].get('total_items', 0)
    exp_items = experimental['playbook'].get('total_items', 0)
    
    print(f"Control Playbook:       {ctrl_items} items (learned from 1 dataset)")
    print(f"Experimental Playbook:  {exp_items} items (transferred from 4 datasets)")
    print(f"Knowledge Transfer:     {exp_items - ctrl_items} items carried over")
    
    print("="*80)


def analyze_improvement_trajectories(results: Dict[str, Any]):
    """Analyze improvement trajectories for patterns"""
    print("\n" + "="*80)
    print("IMPROVEMENT TRAJECTORY ANALYSIS")
    print("="*80)
    
    # Analyze sequential datasets
    print("\nSequential Learning (Datasets 1-4):")
    print("-" * 80)
    
    sequential_results = results.get('sequential_results', [])
    for result in sequential_results:
        dataset = result['dataset']
        trajectory = result['improvement'].get('trajectory', [])
        
        print(f"\n{dataset.upper()}:")
        if trajectory:
            for step in trajectory:
                iter_num = step['iteration']
                cv_score = step['cv_score']
                improvement = step['cv_improvement']
                strategy = step.get('strategy', 'unknown')
                print(f"  Iter {iter_num}: {strategy:20s} -> CV: {cv_score:.4f} (Δ{improvement:+.4f})")
        else:
            print("  No trajectory data")
    
    # Control vs Experimental
    print("\n" + "-" * 80)
    print("Transfer Evaluation (Dataset 5):")
    print("-" * 80)
    
    control = results.get('control_results', {})
    experimental = results.get('experimental_results', {})
    
    print("\nCONTROL (No Transfer):")
    ctrl_trajectory = control.get('improvement', {}).get('trajectory', [])
    for step in ctrl_trajectory:
        iter_num = step['iteration']
        cv_score = step['cv_score']
        improvement = step['cv_improvement']
        print(f"  Iter {iter_num}: CV: {cv_score:.4f} (Δ{improvement:+.4f})")
    
    print("\nEXPERIMENTAL (With Transfer):")
    exp_trajectory = experimental.get('improvement', {}).get('trajectory', [])
    for step in exp_trajectory:
        iter_num = step['iteration']
        cv_score = step['cv_score']
        improvement = step['cv_improvement']
        print(f"  Iter {iter_num}: CV: {cv_score:.4f} (Δ{improvement:+.4f})")
    
    print("="*80)


def generate_summary_statistics(results: Dict[str, Any]) -> Dict[str, Any]:
    """Generate summary statistics for paper"""
    stats = {}
    
    # Sequential learning stats
    sequential_results = results.get('sequential_results', [])
    if sequential_results:
        cv_gains = [r['total_improvement']['cv_gain'] for r in sequential_results]
        test_gains = [r['total_improvement']['test_gain'] for r in sequential_results]
        
        stats['sequential'] = {
            'n_datasets': len(sequential_results),
            'avg_cv_gain': sum(cv_gains) / len(cv_gains),
            'avg_test_gain': sum(test_gains) / len(test_gains),
            'min_test_gain': min(test_gains),
            'max_test_gain': max(test_gains)
        }
    
    # Transfer learning stats
    control = results.get('control_results', {})
    experimental = results.get('experimental_results', {})
    
    if control and experimental:
        stats['transfer'] = {
            'baseline_cv_benefit': experimental['baseline']['cv_score'] - control['baseline']['cv_score'],
            'baseline_test_benefit': experimental['baseline']['test_score'] - control['baseline']['test_score'],
            'final_cv_benefit': experimental['improvement']['final_cv_score'] - control['improvement']['final_cv_score'],
            'final_test_benefit': experimental['improvement']['final_test_score'] - control['improvement']['final_test_score'],
            'improvement_efficiency_gain': experimental['total_improvement']['test_gain'] - control['total_improvement']['test_gain'],
            'knowledge_items_transferred': experimental['playbook'].get('total_items', 0) - control['playbook'].get('total_items', 0)
        }
    
    return stats


def print_paper_ready_summary(stats: Dict[str, Any]):
    """Print summary formatted for ICML paper"""
    print("\n" + "="*80)
    print("PAPER-READY SUMMARY STATISTICS")
    print("="*80)
    
    if 'sequential' in stats:
        seq = stats['sequential']
        print("\nSequential Learning (Training Phase):")
        print(f"  Datasets trained: {seq['n_datasets']}")
        print(f"  Average CV improvement: {seq['avg_cv_gain']:+.4f}")
        print(f"  Average test improvement: {seq['avg_test_gain']:+.4f}")
        print(f"  Range: [{seq['min_test_gain']:+.4f}, {seq['max_test_gain']:+.4f}]")
    
    if 'transfer' in stats:
        trans = stats['transfer']
        print("\nTransfer Learning (Evaluation Phase):")
        print(f"  Baseline benefit (test): {trans['baseline_test_benefit']:+.4f}")
        print(f"  Final benefit (test): {trans['final_test_benefit']:+.4f}")
        print(f"  Improvement efficiency gain: {trans['improvement_efficiency_gain']:+.4f}")
        print(f"  Knowledge items transferred: {trans['knowledge_items_transferred']}")
        
        print("\nKey Finding:")
        if trans['baseline_test_benefit'] > 0:
            print(f"  ✓ Transfer learning provided {trans['baseline_test_benefit']:+.4f} initial advantage")
        else:
            print(f"  ✗ No positive transfer learning effect ({trans['baseline_test_benefit']:+.4f})")
        
        if trans['final_test_benefit'] > 0:
            print(f"  ✓ Final performance improved by {trans['final_test_benefit']:+.4f}")
        else:
            print(f"  ✗ Final performance decreased by {trans['final_test_benefit']:.4f}")
    
    print("="*80)


def plot_results(results: Dict[str, Any], output_dir: Path):
    """Generate plots for paper (if matplotlib available)"""
    if not PLOTTING_AVAILABLE:
        print("\nSkipping plots (matplotlib not installed)")
        return
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot 1: Sequential learning progression
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('ACE Cross-Dataset Transfer Learning Experiment', fontsize=16, fontweight='bold')
    
    sequential_results = results.get('sequential_results', [])
    
    # Plot 1a: CV scores progression
    ax = axes[0, 0]
    datasets = [r['dataset'] for r in sequential_results]
    baseline_cvs = [r['baseline']['cv_score'] for r in sequential_results]
    final_cvs = [r['improvement']['final_cv_score'] for r in sequential_results]
    
    x = np.arange(len(datasets))
    width = 0.35
    ax.bar(x - width/2, baseline_cvs, width, label='Baseline', alpha=0.7)
    ax.bar(x + width/2, final_cvs, width, label='Final', alpha=0.7)
    ax.set_xlabel('Dataset')
    ax.set_ylabel('CV C-Index')
    ax.set_title('Sequential Learning Performance')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=45)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Plot 1b: Improvement gains
    ax = axes[0, 1]
    cv_gains = [r['total_improvement']['cv_gain'] for r in sequential_results]
    test_gains = [r['total_improvement']['test_gain'] for r in sequential_results]
    
    ax.bar(x - width/2, cv_gains, width, label='CV Gain', alpha=0.7)
    ax.bar(x + width/2, test_gains, width, label='Test Gain', alpha=0.7)
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax.set_xlabel('Dataset')
    ax.set_ylabel('Performance Gain')
    ax.set_title('Improvement Achieved')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=45)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Plot 1c: Transfer learning comparison
    ax = axes[1, 0]
    control = results.get('control_results', {})
    experimental = results.get('experimental_results', {})
            
            if control and experimental:
        categories = ['Baseline\nCV', 'Baseline\nTest', 'Final\nCV', 'Final\nTest']
        ctrl_values = [
            control['baseline']['cv_score'],
            control['baseline']['test_score'],
            control['improvement']['final_cv_score'],
            control['improvement']['final_test_score']
        ]
        exp_values = [
            experimental['baseline']['cv_score'],
            experimental['baseline']['test_score'],
            experimental['improvement']['final_cv_score'],
            experimental['improvement']['final_test_score']
        ]
        
        x = np.arange(len(categories))
        ax.bar(x - width/2, ctrl_values, width, label='Control', alpha=0.7)
        ax.bar(x + width/2, exp_values, width, label='Experimental', alpha=0.7)
        ax.set_ylabel('C-Index')
        ax.set_title('Control vs Experimental (Dataset 5)')
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
    
    # Plot 1d: Improvement trajectories
    ax = axes[1, 1]
    
    if control and experimental:
        ctrl_traj = control.get('improvement', {}).get('trajectory', [])
        exp_traj = experimental.get('improvement', {}).get('trajectory', [])
        
        if ctrl_traj:
            iterations = [s['iteration'] for s in ctrl_traj]
            scores = [s['cv_score'] for s in ctrl_traj]
            ax.plot(iterations, scores, 'o-', label='Control', linewidth=2, markersize=6)
        
        if exp_traj:
            iterations = [s['iteration'] for s in exp_traj]
            scores = [s['cv_score'] for s in exp_traj]
            ax.plot(iterations, scores, 's-', label='Experimental', linewidth=2, markersize=6)
        
        ax.set_xlabel('Iteration')
        ax.set_ylabel('CV C-Index')
        ax.set_title('Self-Improvement Trajectory (Dataset 5)')
        ax.legend()
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    output_file = output_dir / 'experiment_results.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved: {output_file}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Analyze ACE experiment results')
    parser.add_argument('--seed', type=int, default=42, help='Random seed used in experiment')
    parser.add_argument('--results-dir', type=str, default='experiments/results',
                       help='Results directory')
    parser.add_argument('--plot', action='store_true', help='Generate plots')
    
    args = parser.parse_args()
    
    # Load results
    results_dir = Path(args.results_dir) / f"seed_{args.seed}"
    
    if not results_dir.exists():
        print(f"ERROR: Results directory not found: {results_dir}")
        print(f"\nAvailable seeds:")
        parent_dir = Path(args.results_dir)
        if parent_dir.exists():
            for d in parent_dir.iterdir():
                if d.is_dir() and d.name.startswith('seed_'):
                    print(f"  - {d.name}")
        return 1
    
    print(f"Loading results from: {results_dir}")
    results = load_results(results_dir)
    
    # Print analyses
    if results.get('sequential_results'):
        print_sequential_learning_summary(results['sequential_results'])
    
    if results.get('control_results') and results.get('experimental_results'):
        print_transfer_learning_analysis(
            results['control_results'],
            results['experimental_results']
        )
    
    analyze_improvement_trajectories(results)
    
    # Generate summary statistics
    stats = generate_summary_statistics(results)
    print_paper_ready_summary(stats)
    
    # Save summary statistics
    stats_file = results_dir / 'summary_statistics.json'
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"\nSummary statistics saved: {stats_file}")
    
    # Generate plots if requested
    if args.plot:
        plot_results(results, results_dir)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
