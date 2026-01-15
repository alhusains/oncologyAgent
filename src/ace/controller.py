"""
ACE Improvement Controller

Orchestrates the self-improvement loop:
1. Establish baseline
2. Generate improvement candidates from playbook
3. Test changes incrementally (ablation)
4. Learn from results
5. Iterate until convergence or max iterations
"""

from typing import Dict, Any, List, Optional, Callable, Tuple
from datetime import datetime
import asyncio
import time

from .schemas import (
    Trajectory, ImprovementExperiment, AblationResult, ImprovementType,
    ActionType, Lesson
)
from .generator import TrajectoryGenerator, ImprovementExperimentTracker
from .reflector import TrajectoryReflector
from .curator import PlaybookCurator


class ImprovementController:
    """
    Controls the self-improvement loop for the ML agent.
    
    This is the main orchestrator that:
    1. Manages improvement experiments
    2. Generates improvement candidates from playbook + LLM
    3. Tests changes incrementally with ablation
    4. Triggers reflection and learning after each iteration
    5. Decides when to stop (convergence, max iterations, etc.)
    """
    
    def __init__(
        self,
        generator: TrajectoryGenerator,
        reflector: TrajectoryReflector,
        curator: PlaybookCurator,
        llm_client=None,
        config: Optional[Dict[str, Any]] = None
    ):
        self.generator = generator
        self.reflector = reflector
        self.curator = curator
        self.llm = llm_client
        
        # Configuration
        self.config = config or {}
        self.max_iterations = self.config.get("max_improvement_iterations", 3)
        self.min_improvement_threshold = self.config.get("min_improvement_threshold", 0.005)
        self.max_changes_per_iteration = self.config.get("max_changes_per_iteration", 3)
        
        # Experiment tracking
        self.experiment_tracker = ImprovementExperimentTracker(generator)
        self.current_experiment: Optional[ImprovementExperiment] = None
        
        # State
        self._baseline_state: Optional[Dict[str, Any]] = None
        self._current_best_score: float = 0.0
        self._iterations_without_improvement: int = 0
    
    async def run_improvement_loop(
        self,
        toolkit,  # MLToolkit instance
        baseline_score: float,
        baseline_model: str,
        baseline_trajectory_id: str,
        session_id: str,
        user_suggestions: Optional[List[str]] = None,
        max_iterations: Optional[int] = None,
        on_iteration_complete: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Run the complete self-improvement loop.
        
        Args:
            toolkit: MLToolkit with the ML pipeline
            baseline_score: Starting performance score
            baseline_model: Name of the baseline model
            baseline_trajectory_id: Trajectory ID of baseline run
            session_id: Current session identifier
            user_suggestions: Optional user-provided improvement suggestions
            max_iterations: Override default max iterations
            on_iteration_complete: Callback after each iteration
            
        Returns:
            Summary of the improvement experiment
        """
        max_iters = max_iterations or self.max_iterations
        
        # Start experiment
        self.current_experiment = self.experiment_tracker.start_experiment(
            session_id=session_id,
            baseline_score=baseline_score,
            baseline_model=baseline_model,
            baseline_trajectory_id=baseline_trajectory_id,
            max_iterations=max_iters
        )
        
        # Store baseline state
        self._baseline_state = self._capture_state(toolkit)
        self._current_best_score = baseline_score
        self._iterations_without_improvement = 0
        
        # Get current conditions for playbook lookup
        conditions = await self._get_conditions(toolkit)
        
        all_trajectories = []
        
        print(f"\n{'='*60}")
        print(f"SELF-IMPROVEMENT LOOP STARTED")
        print(f"{'='*60}")
        print(f"Baseline: {baseline_model} with score {baseline_score:.4f}")
        print(f"Max iterations: {max_iters}")
        print(f"{'='*60}\n")
        
        for iteration in range(max_iters):
            print(f"\n--- Iteration {iteration + 1}/{max_iters} ---")
            
            # 1. Generate improvement candidates
            candidates = await self._generate_candidates(
                toolkit, conditions, user_suggestions, iteration
            )
            
            if not candidates:
                print("No improvement candidates generated. Stopping.")
                break
            
            print(f"Generated {len(candidates)} improvement candidates")
            
            # 2. Test each candidate (ablation)
            iteration_results = []
            for i, candidate in enumerate(candidates[:self.max_changes_per_iteration]):
                print(f"  Testing candidate {i+1}/{min(len(candidates), self.max_changes_per_iteration)}: {candidate['description'][:50]}...")
                
                result = await self._test_candidate(toolkit, candidate, iteration)
                iteration_results.append(result)
                
                if result.is_beneficial:
                    print(f"    Result: Beneficial (+{result.improvement:.4f})")
                else:
                    print(f"    Result: Not beneficial ({result.improvement:+.4f})")
            
            # 3. Apply beneficial changes and get new score
            beneficial_changes = [r for r in iteration_results if r.is_beneficial]
            
            if beneficial_changes:
                # Apply all beneficial changes together
                new_score = await self._apply_beneficial_changes(toolkit, beneficial_changes)
                
                print(f"  Applied {len(beneficial_changes)} beneficial changes")
                print(f"  New score: {new_score:.4f} (was {self._current_best_score:.4f})")
                
                if new_score > self._current_best_score:
                    self._current_best_score = new_score
                    self._iterations_without_improvement = 0
                else:
                    self._iterations_without_improvement += 1
            else:
                self._iterations_without_improvement += 1
                print("  No beneficial changes this iteration")
            
            # 4. Record trajectory for this iteration
            traj = self._create_iteration_trajectory(toolkit, iteration + 1, iteration_results)
            all_trajectories.append(traj)
            self.experiment_tracker.complete_iteration(traj.trajectory_id, self._current_best_score)
            
            # 5. Check stopping conditions
            if self._should_stop():
                print(f"\nStopping early: {self._get_stop_reason()}")
                break
            
            # Callback if provided
            if on_iteration_complete:
                on_iteration_complete(iteration + 1, self._current_best_score, iteration_results)
        
        # Finalize experiment
        final_model = toolkit.state.get("best_model") or baseline_model
        experiment = self.experiment_tracker.end_experiment(
            final_score=self._current_best_score,
            final_model=final_model,
            stop_reason=self._get_stop_reason() if self._iterations_without_improvement >= 2 else None
        )
        
        # 6. Reflect and learn from the experiment
        print("\nReflecting on experiment...")
        lessons = await self.reflector.reflect_on_experiment(experiment, all_trajectories)
        
        # 7. Update playbook with lessons
        if lessons:
            curation_result = self.curator.curate_lessons(lessons)
            print(f"Learned {curation_result['items_created']} new strategies, updated {curation_result['items_merged']} existing")
        
        # Summary
        summary = experiment.get_improvement_summary()
        
        print(f"\n{'='*60}")
        print("IMPROVEMENT LOOP COMPLETE")
        print(f"{'='*60}")
        print(f"Final score: {self._current_best_score:.4f}")
        print(f"Total improvement: {experiment.total_improvement:+.4f}")
        print(f"Beneficial changes: {summary['beneficial_changes']}")
        print(f"Iterations: {experiment.current_iteration}")
        print(f"{'='*60}\n")
        
        return {
            "success": True,
            "experiment_id": experiment.experiment_id,
            "baseline_score": baseline_score,
            "final_score": self._current_best_score,
            "total_improvement": experiment.total_improvement,
            "iterations": experiment.current_iteration,
            "beneficial_changes": summary["beneficial_changes"],
            "successful_changes": experiment.successful_changes,
            "lessons_learned": len(lessons)
        }
    
    async def _generate_candidates(
        self,
        toolkit,
        conditions: Dict[str, Any],
        user_suggestions: Optional[List[str]],
        iteration: int
    ) -> List[Dict[str, Any]]:
        """Generate improvement candidates from playbook and LLM"""
        candidates = []
        
        # 1. User suggestions (highest priority)
        if user_suggestions and iteration == 0:
            for suggestion in user_suggestions:
                candidate = self._parse_user_suggestion(suggestion)
                if candidate:
                    candidate["source"] = "user"
                    candidate["priority"] = 10
                    candidates.append(candidate)
        
        # 2. Playbook strategies
        playbook_strategies = self.curator.get_strategies_for_improvement(
            conditions,
            focus_domains=["feature_interaction", "model_selection", "preprocessing"]
        )
        
        for strategy in playbook_strategies[:5]:
            candidate = {
                "type": self._domain_to_improvement_type(strategy["domain"]),
                "description": strategy["title"],
                "strategy": strategy["strategy"],
                "expected_improvement": strategy["expected_improvement"],
                "confidence": strategy["confidence"],
                "source": "playbook",
                "item_id": strategy["item_id"],
                "priority": strategy["confidence"] * strategy["expected_improvement"] * 100
            }
            candidates.append(candidate)
        
        # 3. LLM-generated candidates (if not enough from playbook)
        if len(candidates) < 3 and self.llm is not None:
            llm_candidates = await self._generate_llm_candidates(toolkit, conditions, iteration)
            candidates.extend(llm_candidates)
        
        # Sort by priority
        candidates.sort(key=lambda x: x.get("priority", 0), reverse=True)
        
        # Remove duplicates
        seen = set()
        unique = []
        for c in candidates:
            key = (c["type"].value if hasattr(c["type"], "value") else c["type"], c["description"][:50])
            if key not in seen:
                seen.add(key)
                unique.append(c)
        
        return unique
    
    async def _generate_llm_candidates(
        self,
        toolkit,
        conditions: Dict[str, Any],
        iteration: int
    ) -> List[Dict[str, Any]]:
        """Generate improvement candidates using LLM"""
        if self.llm is None:
            return []
        
        # Build context
        state = toolkit.state
        current_model = state.get("best_model") or "unknown"
        current_score = state.get("best_score") or 0
        
        feature_result = state.get("feature_result") or {}
        n_features = feature_result.get("n_features", 0) if feature_result else 0
        
        task_type = conditions.get('task_type', 'classification')
        
        # Task-specific model suggestions
        task_models = {
            "classification": "autogluon, xgboost, random_forest, lightgbm, catboost, logistic_regression",
            "regression": "xgboost, lightgbm, random_forest, ridge, linear_regression",
            "survival": "cox_ph, random_survival_forest, gradient_boosting_survival, coxnet"
        }
        available_models = task_models.get(task_type, task_models["classification"])
        
        # Task-specific metric info
        task_metrics = {
            "classification": "accuracy, F1, ROC-AUC (higher is better)",
            "regression": "R², MAE, RMSE (R² higher better, MAE/RMSE lower better)",
            "survival": "C-index, Integrated Brier Score (C-index higher better, IBS lower better)"
        }
        metric_info = task_metrics.get(task_type, task_metrics["classification"])
        
        prompt = f"""Suggest 2-3 specific improvements for this {task_type.upper()} ML pipeline.

CURRENT STATE:
- Task: {task_type}
- Cancer type: {conditions.get('cancer_type', 'general')}
- Best model: {current_model} (score: {current_score:.4f})
- Features: {n_features}
- Iteration: {iteration + 1}
- Primary metrics: {metric_info}

AVAILABLE MODELS FOR {task_type.upper()}:
{available_models}

PREVIOUS ATTEMPTS THIS EXPERIMENT:
{self._summarize_previous_attempts()}

Suggest DIFFERENT improvements. Be specific and actionable.
For {task_type} tasks, consider:
{"- Feature interactions relevant to clinical outcomes" if task_type == "classification" else ""}
{"- Handling censored observations properly" if task_type == "survival" else ""}
{"- Target variable transformations" if task_type == "regression" else ""}
- Model selection appropriate for the task type

Return JSON:
{{
    "improvements": [
        {{
            "type": "feature_interaction|feature_transform|model_change|hyperparameter|preprocessing",
            "description": "Short description",
            "action": "Specific action to take",
            "expected_impact": "low|medium|high"
        }}
    ]
}}
"""
        
        try:
            response = await self.llm.complete_json(prompt)
            
            candidates = []
            for imp in response.get("improvements", []):
                type_map = {
                    "feature_interaction": ImprovementType.FEATURE_INTERACTION,
                    "feature_transform": ImprovementType.FEATURE_TRANSFORM,
                    "model_change": ImprovementType.MODEL_CHANGE,
                    "hyperparameter": ImprovementType.HYPERPARAMETER,
                    "preprocessing": ImprovementType.PREPROCESSING
                }
                
                candidate = {
                    "type": type_map.get(imp.get("type"), ImprovementType.FEATURE_INTERACTION),
                    "description": imp.get("description", ""),
                    "strategy": imp.get("action", ""),
                    "expected_improvement": {"low": 0.01, "medium": 0.02, "high": 0.03}.get(imp.get("expected_impact", "medium"), 0.02),
                    "confidence": 0.5,
                    "source": "llm",
                    "priority": 5
                }
                candidates.append(candidate)
            
            return candidates
        except Exception as e:
            print(f"LLM candidate generation failed: {e}")
            return []
    
    async def _test_candidate(
        self,
        toolkit,
        candidate: Dict[str, Any],
        iteration: int
    ) -> AblationResult:
        """Test a single improvement candidate (ablation test)"""
        start_time = time.time()
        
        # Reset to baseline state
        await self._reset_to_baseline(toolkit)
        
        # Apply the candidate change
        try:
            score_with_change = await self._apply_change(toolkit, candidate)
        except Exception as e:
            print(f"    Error applying change: {e}")
            score_with_change = self._baseline_state.get("best_score", 0)
        
        execution_time = time.time() - start_time
        
        # Record result
        result = self.experiment_tracker.record_ablation(
            change_type=candidate["type"] if isinstance(candidate["type"], ImprovementType) else ImprovementType.FEATURE_INTERACTION,
            change_description=candidate["description"],
            change_details={
                "strategy": candidate.get("strategy", ""),
                "source": candidate.get("source", ""),
                "item_id": candidate.get("item_id")
            },
            score_with_change=score_with_change,
            execution_time=execution_time
        )
        
        # Update playbook with usage feedback
        if candidate.get("item_id"):
            self.curator.record_strategy_usage(
                item_id=candidate["item_id"],
                success=result.is_beneficial,
                improvement=result.improvement
            )
        
        return result
    
    async def _apply_change(self, toolkit, candidate: Dict[str, Any]) -> float:
        """Apply a single change and return new score"""
        change_type = candidate["type"]
        strategy = candidate.get("strategy", "")
        
        # Get task-aware default model
        default_model = self._get_default_model_for_task(toolkit)
        
        if change_type == ImprovementType.MODEL_CHANGE:
            # Try a different model
            new_model = self._extract_model_from_strategy(strategy)
            if new_model:
                result = await toolkit._train_model(new_model)
                if result.get("success"):
                    return result.get("cv_score", 0)
        
        elif change_type in [ImprovementType.FEATURE_INTERACTION, ImprovementType.FEATURE_TRANSFORM]:
            # Re-run feature engineering with modification
            # For now, trigger refine_features
            result = await toolkit._refine_features(
                performance_feedback=strategy,
                focus_areas=["feature_interactions" if change_type == ImprovementType.FEATURE_INTERACTION else "transformations"]
            )
            
            if result.get("success"):
                # Retrain best model (use 'or' pattern since value might be None)
                best_model = toolkit.state.get("best_model") or default_model
                train_result = await toolkit._train_model(best_model)
                if train_result.get("success"):
                    return train_result.get("cv_score", 0)
        
        elif change_type == ImprovementType.PREPROCESSING:
            # Try different preprocessing
            data_analysis = toolkit.state.get("data_analysis") or {}
            if data_analysis:
                result = await toolkit._engineer_features(
                    scaling_strategy="robust",  # Try robust scaling
                    encoding_strategy="onehot"
                )
                if result.get("success"):
                    best_model = toolkit.state.get("best_model") or default_model
                    train_result = await toolkit._train_model(best_model)
                    if train_result.get("success"):
                        return train_result.get("cv_score", 0)
        
        # Return baseline score if change failed
        return self._baseline_state.get("best_score", 0)
    
    def _get_default_model_for_task(self, toolkit) -> str:
        """Get a task-appropriate default model"""
        data_analysis = toolkit.state.get("data_analysis") or {}
        task_type = data_analysis.get("task_type", "classification")
        
        if task_type == "survival":
            return "random_survival_forest"
        elif task_type == "regression":
            return "random_forest_regressor"
        else:
            return "random_forest"
    
    async def _apply_beneficial_changes(
        self,
        toolkit,
        beneficial_results: List[AblationResult]
    ) -> float:
        """Apply all beneficial changes together"""
        # Reset to baseline
        await self._reset_to_baseline(toolkit)
        
        # Track best score from applied changes
        best_score_from_changes = self._baseline_state.get("best_score", 0)
        best_model_from_changes = self._baseline_state.get("best_model")
        
        # Apply each beneficial change
        for result in beneficial_results:
            try:
                candidate = {
                    "type": result.change_type,
                    "strategy": result.change_details.get("strategy", ""),
                    "description": result.change_description
                }
                score = await self._apply_change(toolkit, candidate)
                
                # Track if this change improved the score
                if score > best_score_from_changes:
                    best_score_from_changes = score
                    best_model_from_changes = toolkit.state.get("best_model")
                    
            except Exception as e:
                print(f"    Warning: Could not apply change '{result.change_description}': {e}")
        
        # Update toolkit state with best results from applied changes
        toolkit.state["best_score"] = best_score_from_changes
        if best_model_from_changes:
            toolkit.state["best_model"] = best_model_from_changes
        
        # Update baseline state to new state after changes
        self._baseline_state = self._capture_state(toolkit)
        
        return best_score_from_changes
    
    async def _reset_to_baseline(self, toolkit):
        """Reset toolkit state to baseline"""
        if self._baseline_state:
            # Restore key state elements
            toolkit.state["feature_result"] = self._baseline_state.get("feature_result")
            toolkit.state["trained_models"] = self._baseline_state.get("trained_models", {}).copy()
            toolkit.state["best_model"] = self._baseline_state.get("best_model")
            toolkit.state["best_score"] = self._baseline_state.get("best_score", 0)
    
    def _capture_state(self, toolkit) -> Dict[str, Any]:
        """Capture current toolkit state"""
        return {
            "feature_result": toolkit.state.get("feature_result"),
            "trained_models": toolkit.state.get("trained_models", {}).copy(),
            "best_model": toolkit.state.get("best_model"),
            "best_score": toolkit.state.get("best_score", 0),
            "evaluation_results": toolkit.state.get("evaluation_results", {}).copy()
        }
    
    def _create_iteration_trajectory(
        self,
        toolkit,
        iteration: int,
        results: List[AblationResult]
    ) -> Trajectory:
        """Create a trajectory for this iteration"""
        data_analysis = toolkit.state.get("data_analysis") or {}
        feature_result = toolkit.state.get("feature_result") or {}
        
        # Get task type correctly (survival, classification, or regression)
        task_type = feature_result.get("task_type") or data_analysis.get("task_type")
        if not task_type:
            # Infer from data if not explicitly set
            if "time_variable" in data_analysis or "survival_time" in str(data_analysis):
                task_type = "survival"
            else:
                task_type = "classification"  # Conservative default
        
        # Get dataset info with sample counts - try multiple sources
        dataset_info = data_analysis.get("dataset_info", {}).copy()
        
        # Try to get sample count from multiple possible sources
        if "n_samples" not in dataset_info or dataset_info["n_samples"] == 0:
            # Try feature_result first
            if feature_result and "n_samples_train" in feature_result:
                dataset_info["n_samples"] = feature_result.get("n_samples_train")
            # Try data_analysis
            elif "n_samples" in data_analysis:
                dataset_info["n_samples"] = data_analysis.get("n_samples")
            # Try direct from toolkit state
            elif toolkit and toolkit.state.get("X_train") is not None:
                import pandas as pd
                X = toolkit.state.get("X_train")
                if isinstance(X, pd.DataFrame):
                    dataset_info["n_samples"] = len(X)
                elif hasattr(X, "shape"):
                    dataset_info["n_samples"] = X.shape[0]
        
        traj = Trajectory(
            experiment_id=self.current_experiment.experiment_id if self.current_experiment else "",
            run_number=iteration,
            parent_trajectory_id=self.current_experiment.baseline_trajectory_id if self.current_experiment else None,
            dataset_info=dataset_info,
            task_type=task_type,
            baseline_score=self._baseline_state.get("best_score", 0) if self._baseline_state else 0
        )
        
        # Add changes
        for result in results:
            traj.changes_from_baseline.append({
                "type": result.change_type.value,
                "description": result.change_description,
                "improvement": result.improvement,
                "beneficial": result.is_beneficial
            })
        
        # Finalize
        traj.finalize(
            final_metrics={"score": self._current_best_score},
            best_model=toolkit.state.get("best_model", ""),
            best_score=self._current_best_score
        )
        
        return traj
    
    def _should_stop(self) -> bool:
        """Check if we should stop the improvement loop"""
        # No improvement for 2 iterations
        if self._iterations_without_improvement >= 2:
            return True
        
        # Max iterations reached (handled by loop)
        return False
    
    def _get_stop_reason(self) -> str:
        """Get reason for stopping"""
        if self._iterations_without_improvement >= 2:
            return "No improvement for 2 consecutive iterations"
        return "Max iterations reached"
    
    async def _get_conditions(self, toolkit) -> Dict[str, Any]:
        """Get current conditions for playbook lookup using LLM analysis"""
        state = toolkit.state
        data_analysis = state.get("data_analysis") or {}
        feature_result = state.get("feature_result") or {}
        
        n_samples = feature_result.get("n_samples_train", 0) if feature_result else 0
        n_features = feature_result.get("n_features", 0) if feature_result else 0
        task_type = data_analysis.get("task_type", "classification") if data_analysis else "classification"
        
        # Use LLM for cancer context and dataset categorization
        cancer_context = await self._detect_cancer_context(state)
        dataset_categories = await self._categorize_dataset(n_samples, n_features, task_type)
        
        return {
            "task_type": task_type,
            "cancer_type": cancer_context.get("cancer_type", "unknown"),
            "cancer_site": cancer_context.get("site", ""),
            "n_samples_range": dataset_categories.get("n_samples", "medium"),
            "n_features_range": dataset_categories.get("n_features", "medium"),
            "baseline_model": state.get("best_model")
        }
    
    async def _detect_cancer_context(self, state: Dict[str, Any]) -> Dict[str, str]:
        """Use LLM to detect cancer context from state"""
        if self.llm is None:
            return {"cancer_type": "unknown", "site": ""}
        
        objective = state.get("objective", "")
        dataset_path = state.get("dataset_path", "")
        feature_names = state.get("feature_names", [])[:20]
        
        context_parts = []
        if objective:
            context_parts.append(f"Objective: {objective}")
        if dataset_path:
            context_parts.append(f"Dataset: {dataset_path}")
        if feature_names:
            context_parts.append(f"Features: {', '.join(str(f) for f in feature_names)}")
        
        if not context_parts:
            return {"cancer_type": "unknown", "site": ""}
        
        context = "\n".join(context_parts)
        
        prompt = f"""Analyze this oncology ML task and extract cancer context.

TASK INFORMATION:
{context}

This is an ONCOLOGY dataset. Extract:
1. Cancer type (e.g., breast, lung, prostate, colorectal, melanoma, etc.)
2. Anatomical site if identifiable

Return JSON:
{{
    "cancer_type": "specific cancer type or 'unknown'",
    "site": "anatomical site if known"
}}
"""
        
        try:
            response = await self.llm.complete_json(prompt)
            return {
                "cancer_type": response.get("cancer_type", "unknown").lower(),
                "site": response.get("site", "")
            }
        except Exception as e:
            print(f"LLM cancer context detection failed: {e}")
            return {"cancer_type": "unknown", "site": ""}
    
    async def _categorize_dataset(self, n_samples: int, n_features: int, task_type: str) -> Dict[str, str]:
        """Use LLM to categorize dataset characteristics"""
        if self.llm is None or n_samples == 0:
            # Fallback to simple categorization
            return {
                "n_samples": "small" if n_samples < 500 else "medium" if n_samples < 5000 else "large",
                "n_features": "low" if n_features < 20 else "medium" if n_features < 100 else "high"
            }
        
        prompt = f"""Categorize this ONCOLOGY dataset for ML.

DATASET:
- Samples: {n_samples} patients
- Features: {n_features} variables
- Task: {task_type}

Categorize for clinical oncology context:
- Sample size: small/medium/large
- Feature count: low/medium/high

Return JSON:
{{
    "n_samples": "small|medium|large",
    "n_features": "low|medium|high"
}}
"""
        
        try:
            response = await self.llm.complete_json(prompt)
            return {
                "n_samples": response.get("n_samples", "medium"),
                "n_features": response.get("n_features", "medium")
            }
        except Exception as e:
            print(f"LLM categorization failed: {e}, using fallback")
            return {
                "n_samples": "small" if n_samples < 500 else "medium" if n_samples < 5000 else "large",
                "n_features": "low" if n_features < 20 else "medium" if n_features < 100 else "high"
            }
    
    def _parse_user_suggestion(self, suggestion: str) -> Optional[Dict[str, Any]]:
        """Parse a user suggestion into a candidate"""
        suggestion_lower = suggestion.lower()
        
        # Detect type
        if any(kw in suggestion_lower for kw in ["interaction", "combine", "multiply"]):
            imp_type = ImprovementType.FEATURE_INTERACTION
        elif any(kw in suggestion_lower for kw in ["transform", "log", "sqrt", "scale"]):
            imp_type = ImprovementType.FEATURE_TRANSFORM
        elif any(kw in suggestion_lower for kw in ["model", "train", "xgboost", "random forest", "logistic"]):
            imp_type = ImprovementType.MODEL_CHANGE
        elif any(kw in suggestion_lower for kw in ["hyperparameter", "tune", "learning rate"]):
            imp_type = ImprovementType.HYPERPARAMETER
        else:
            imp_type = ImprovementType.PREPROCESSING
        
        return {
            "type": imp_type,
            "description": suggestion,
            "strategy": suggestion,
            "expected_improvement": 0.02,
            "confidence": 0.6
        }
    
    def _domain_to_improvement_type(self, domain: str) -> ImprovementType:
        """Map domain to improvement type"""
        mapping = {
            "feature_interaction": ImprovementType.FEATURE_INTERACTION,
            "model_selection": ImprovementType.MODEL_CHANGE,
            "preprocessing": ImprovementType.PREPROCESSING,
            "hyperparameter": ImprovementType.HYPERPARAMETER,
            "clinical_pattern": ImprovementType.FEATURE_INTERACTION
        }
        return mapping.get(domain, ImprovementType.FEATURE_INTERACTION)
    
    def _extract_model_from_strategy(self, strategy: str) -> Optional[str]:
        """Extract model name from strategy string"""
        strategy_lower = strategy.lower()
        
        # All task types: classification, regression, survival
        models = {
            # Classification models
            "xgboost": ["xgboost", "xgb"],
            "random_forest": ["random forest", "rf"],
            "logistic_regression": ["logistic", "lr"],
            "catboost": ["catboost"],
            "lightgbm": ["lightgbm", "lgbm"],
            "gradient_boosting": ["gradient boosting", "gbm"],
            "autogluon": ["autogluon", "automl"],
            
            # Regression models
            "linear_regression": ["linear regression", "linear_regression"],
            "ridge": ["ridge"],
            "lasso": ["lasso"],
            "elastic_net": ["elastic net", "elasticnet"],
            
            # Survival models
            "cox_ph": ["cox", "cox_ph", "proportional hazard"],
            "coxnet": ["coxnet", "penalized cox"],
            "random_survival_forest": ["random survival forest", "rsf", "survival forest"],
            "gradient_boosting_survival": ["gradient boosting survival", "gbs", "survival boosting"],
            "deepsurv": ["deepsurv", "deep survival"]
        }
        
        for model_name, keywords in models.items():
            for kw in keywords:
                if kw in strategy_lower:
                    return model_name
        
        return None
    
    def _summarize_previous_attempts(self) -> str:
        """Summarize previous attempts for LLM context"""
        if not self.current_experiment or not self.current_experiment.ablation_results:
            return "None yet"
        
        lines = []
        for r in self.current_experiment.ablation_results[-5:]:
            status = "worked" if r.is_beneficial else "didn't work"
            lines.append(f"- {r.change_description}: {status} ({r.improvement:+.4f})")
        
        return "\n".join(lines) if lines else "None yet"

