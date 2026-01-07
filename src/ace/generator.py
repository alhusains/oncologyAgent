"""
ACE Generator Component

Wraps ML pipeline execution to capture detailed trajectories.
The Generator tracks what actions are taken and their outcomes.
"""

from typing import Dict, Any, Optional, List, Callable
from datetime import datetime
import uuid
import functools
import copy

from .schemas import (
    Trajectory, TrajectoryStep, ActionType, ActionOutcome,
    ImprovementExperiment, AblationResult, ImprovementType
)


class TrajectoryGenerator:
    """
    Tracks execution trajectories for ML pipeline operations.
    
    The Generator wraps tool executions to capture:
    - What actions were taken
    - What inputs were used
    - What outputs were produced
    - How performance changed
    """
    
    def __init__(self):
        self.current_trajectory: Optional[Trajectory] = None
        self.trajectory_history: List[Trajectory] = []
        
        # Performance tracking
        self._last_metric: Optional[float] = None
        self._metric_name: str = "score"
        self._baseline_metric: Optional[float] = None
        
        # Current improvement experiment
        self.current_experiment: Optional[ImprovementExperiment] = None
    
    def start_trajectory(
        self,
        experiment_id: str,
        dataset_info: Dict[str, Any],
        cancer_type: Optional[str] = None,
        task_type: str = "classification",
        run_number: int = 0,
        parent_trajectory_id: Optional[str] = None,
        baseline_score: Optional[float] = None
    ) -> Trajectory:
        """
        Start a new trajectory for an experiment run.
        
        Args:
            experiment_id: Unique experiment identifier
            dataset_info: Information about the dataset
            cancer_type: Type of cancer (for domain-specific knowledge)
            task_type: classification, regression, or survival
            run_number: 0 for baseline, 1+ for improvement iterations
            parent_trajectory_id: ID of previous trajectory (for improvement runs)
            baseline_score: Score to compare against (for improvement runs)
        """
        self.current_trajectory = Trajectory(
            trajectory_id=str(uuid.uuid4()),
            experiment_id=experiment_id,
            run_number=run_number,
            parent_trajectory_id=parent_trajectory_id,
            dataset_info=dataset_info,
            cancer_type=cancer_type,
            task_type=task_type,
            baseline_score=baseline_score
        )
        
        # Reset metric tracking
        self._last_metric = baseline_score
        
        return self.current_trajectory
    
    def end_trajectory(
        self,
        final_metrics: Dict[str, float],
        best_model: Optional[str] = None,
        best_score: float = 0.0
    ) -> Trajectory:
        """
        Finalize and return the current trajectory.
        
        Args:
            final_metrics: Dictionary of final performance metrics
            best_model: Name of the best performing model
            best_score: Best achieved score
        """
        if self.current_trajectory is None:
            raise ValueError("No active trajectory to end")
        
        self.current_trajectory.finalize(final_metrics, best_model, best_score)
        
        completed = self.current_trajectory
        self.trajectory_history.append(completed)
        self.current_trajectory = None
        
        return completed
    
    def record_step(
        self,
        action_type: ActionType,
        action_name: str,
        action_inputs: Dict[str, Any],
        action_outputs: Dict[str, Any],
        reasoning: str = "",
        context_snapshot: Optional[Dict[str, Any]] = None
    ) -> Optional[TrajectoryStep]:
        """
        Record an action step in the current trajectory.
        
        Args:
            action_type: Type of action (from ActionType enum)
            action_name: Name of the action/function
            action_inputs: Parameters used
            action_outputs: Results from the action
            reasoning: LLM's reasoning for this action
            context_snapshot: Current state context
        """
        if self.current_trajectory is None:
            return None  # No active trajectory, skip recording
        
        # Classify outcome
        outcome = self._classify_outcome(action_outputs)
        
        # Track metric changes
        metric_before = self._last_metric
        metric_after = self._extract_metric(action_outputs)
        metric_delta = None
        
        if metric_before is not None and metric_after is not None:
            metric_delta = metric_after - metric_before
        
        if metric_after is not None:
            self._last_metric = metric_after
        
        step = TrajectoryStep(
            step_id=len(self.current_trajectory.steps),
            action_type=action_type,
            action_name=action_name,
            action_inputs=self._sanitize_for_storage(action_inputs),
            action_outputs=self._sanitize_for_storage(action_outputs),
            outcome=outcome,
            reasoning=reasoning,
            metric_before=metric_before,
            metric_after=metric_after,
            metric_delta=metric_delta,
            context_snapshot=context_snapshot or {}
        )
        
        self.current_trajectory.add_step(step)
        return step
    
    def record_change(self, change_type: str, change_details: Dict[str, Any]):
        """Record a change from baseline for improvement runs"""
        if self.current_trajectory is None:
            return
        
        self.current_trajectory.changes_from_baseline.append({
            "type": change_type,
            "details": change_details,
            "timestamp": datetime.now().isoformat()
        })
    
    def get_trajectory_summary(self) -> Dict[str, Any]:
        """Get summary of current trajectory"""
        if self.current_trajectory is None:
            return {"active": False}
        
        traj = self.current_trajectory
        return {
            "active": True,
            "trajectory_id": traj.trajectory_id,
            "run_number": traj.run_number,
            "n_steps": len(traj.steps),
            "current_metric": self._last_metric,
            "baseline_score": traj.baseline_score,
            "changes_made": len(traj.changes_from_baseline)
        }
    
    def _classify_outcome(self, outputs: Dict[str, Any]) -> ActionOutcome:
        """Classify action outcome based on outputs"""
        if "error" in outputs:
            return ActionOutcome.ERROR
        
        if not outputs.get("success", True):
            return ActionOutcome.FAILURE
        
        # Check for metric improvement
        metric = self._extract_metric(outputs)
        if metric is not None and self._last_metric is not None:
            delta = metric - self._last_metric
            if delta > 0.005:  # 0.5% improvement threshold
                return ActionOutcome.SUCCESS
            elif delta < -0.005:
                return ActionOutcome.FAILURE
        
        return ActionOutcome.NEUTRAL
    
    def _extract_metric(self, outputs: Dict[str, Any]) -> Optional[float]:
        """Extract primary metric from outputs"""
        # Direct metric keys - covers classification, regression, and survival
        metric_keys = [
            # General
            "cv_score", "best_score", "primary_score",
            # Classification
            "accuracy", "f1", "roc_auc", "precision", "recall",
            # Survival
            "concordance_index", "c_index", "cindex",
            # Regression  
            "r2", "r2_score", "mae", "rmse"
        ]
        
        for key in metric_keys:
            if key in outputs:
                try:
                    value = float(outputs[key])
                    # For error metrics (mae, rmse), we want lower = better
                    # So negate them for consistent "higher = better" tracking
                    if key in ["mae", "rmse", "mse"]:
                        return -value  # Negate so improvement = positive delta
                    return value
                except (TypeError, ValueError):
                    continue
        
        # Nested in metrics dict
        if "metrics" in outputs and isinstance(outputs["metrics"], dict):
            metrics = outputs["metrics"]
            for key in metric_keys:
                if key in metrics:
                    try:
                        value = float(metrics[key])
                        if key in ["mae", "rmse", "mse"]:
                            return -value
                        return value
                    except (TypeError, ValueError):
                        continue
        
        # Primary metric/score
        if "primary_score" in outputs:
            try:
                return float(outputs["primary_score"])
            except (TypeError, ValueError):
                pass
        
        return None
    
    def _sanitize_for_storage(self, data: Any, max_depth: int = 3) -> Any:
        """Sanitize data for JSON storage, removing large objects"""
        if max_depth <= 0:
            return "<truncated>"
        
        if data is None:
            return None
        
        if isinstance(data, (str, int, float, bool)):
            if isinstance(data, str) and len(data) > 1000:
                return data[:1000] + "..."
            return data
        
        if isinstance(data, (list, tuple)):
            if len(data) > 20:
                return [self._sanitize_for_storage(x, max_depth - 1) for x in data[:20]] + ["..."]
            return [self._sanitize_for_storage(x, max_depth - 1) for x in data]
        
        if isinstance(data, dict):
            # Skip large/non-serializable keys
            skip_keys = {
                "model", "preprocessor", "X_train", "X_test", "y_train", "y_test",
                "data_splits", "pipeline", "transformer", "scaler", "encoder"
            }
            result = {}
            for k, v in data.items():
                if k in skip_keys:
                    result[k] = f"<{type(v).__name__}>"
                else:
                    result[k] = self._sanitize_for_storage(v, max_depth - 1)
            return result
        
        # Non-serializable types
        return f"<{type(data).__name__}>"


# Improvement experiment tracking
class ImprovementExperimentTracker:
    """
    Tracks self-improvement experiments with ablation testing.
    
    This manages the improvement loop:
    1. Establish baseline
    2. Try changes incrementally
    3. Track which changes helped
    4. Learn from results
    """
    
    def __init__(self, generator: TrajectoryGenerator):
        self.generator = generator
        self.current_experiment: Optional[ImprovementExperiment] = None
        self.experiment_history: List[ImprovementExperiment] = []
    
    def start_experiment(
        self,
        session_id: str,
        baseline_score: float,
        baseline_model: str,
        baseline_trajectory_id: str,
        max_iterations: int = 3
    ) -> ImprovementExperiment:
        """Start a new improvement experiment"""
        self.current_experiment = ImprovementExperiment(
            session_id=session_id,
            max_iterations=max_iterations,
            baseline_trajectory_id=baseline_trajectory_id,
            baseline_score=baseline_score,
            baseline_model=baseline_model
        )
        self.generator.current_experiment = self.current_experiment
        return self.current_experiment
    
    def record_ablation(
        self,
        change_type: ImprovementType,
        change_description: str,
        change_details: Dict[str, Any],
        score_with_change: float,
        execution_time: float = 0.0
    ) -> AblationResult:
        """Record result of trying a single change"""
        if self.current_experiment is None:
            raise ValueError("No active experiment")
        
        result = AblationResult(
            change_type=change_type,
            change_description=change_description,
            change_details=change_details,
            baseline_score=self.current_experiment.baseline_score,
            score_with_change=score_with_change,
            execution_time_seconds=execution_time
        )
        result.compute_improvement()
        
        self.current_experiment.add_ablation_result(result)
        return result
    
    def complete_iteration(self, trajectory_id: str, score: float):
        """Complete one improvement iteration"""
        if self.current_experiment is None:
            return
        
        self.current_experiment.iteration_trajectories.append(trajectory_id)
        self.current_experiment.current_iteration += 1
        
        # Update best score if improved
        if score > self.current_experiment.final_score:
            self.current_experiment.final_score = score
    
    def end_experiment(
        self,
        final_score: float,
        final_model: Optional[str] = None,
        stop_reason: Optional[str] = None
    ) -> ImprovementExperiment:
        """End the current improvement experiment"""
        if self.current_experiment is None:
            raise ValueError("No active experiment")
        
        exp = self.current_experiment
        exp.final_score = final_score
        exp.final_model = final_model
        exp.total_improvement = final_score - exp.baseline_score
        exp.completed = True
        
        if stop_reason:
            exp.stopped_early = True
            exp.stop_reason = stop_reason
        
        self.experiment_history.append(exp)
        self.current_experiment = None
        self.generator.current_experiment = None
        
        return exp
    
    def should_continue(self) -> bool:
        """Check if we should continue improving"""
        if self.current_experiment is None:
            return False
        
        exp = self.current_experiment
        
        # Check iteration limit
        if exp.current_iteration >= exp.max_iterations:
            return False
        
        # Check if we're making progress (at least some beneficial changes)
        if exp.current_iteration > 1:
            recent_results = exp.ablation_results[-3:]  # Last 3 changes
            beneficial = sum(1 for r in recent_results if r.is_beneficial)
            if beneficial == 0:
                return False  # No recent improvements
        
        return True
    
    def get_experiment_summary(self) -> Dict[str, Any]:
        """Get summary of current experiment"""
        if self.current_experiment is None:
            return {"active": False}
        
        return {
            "active": True,
            **self.current_experiment.get_improvement_summary()
        }

