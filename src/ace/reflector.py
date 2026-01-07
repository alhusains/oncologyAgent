"""
ACE Reflector Component

Analyzes execution trajectories to extract lessons.
The Reflector identifies what worked, what didn't, and why.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
import json

from .schemas import (
    Trajectory, TrajectoryStep, Lesson, LessonType,
    ActionType, ActionOutcome, ImprovementExperiment, AblationResult
)


class TrajectoryReflector:
    """
    Analyzes trajectories and improvement experiments to extract lessons.
    
    The Reflector performs structured analysis to identify:
    - What strategies worked and why
    - What strategies failed and why  
    - Patterns that generalize across experiments
    - Specific change attributions from ablation testing
    """
    
    def __init__(self, llm_client=None):
        """
        Args:
            llm_client: Optional LLM client for deep analysis
        """
        self.llm = llm_client
        self.reflection_history: List[Dict[str, Any]] = []
    
    async def reflect_on_trajectory(
        self,
        trajectory: Trajectory,
        playbook_context: Optional[str] = None
    ) -> List[Lesson]:
        """
        Perform reflection on a single trajectory.
        
        Args:
            trajectory: The trajectory to analyze
            playbook_context: Current playbook knowledge for context
            
        Returns:
            List of extracted lessons
        """
        lessons = []
        
        # 1. Analyze successful actions
        success_lessons = self._analyze_successes(trajectory)
        lessons.extend(success_lessons)
        
        # 2. Analyze failures
        failure_lessons = self._analyze_failures(trajectory)
        lessons.extend(failure_lessons)
        
        # 3. Identify action-type patterns
        pattern_lessons = self._identify_patterns(trajectory)
        lessons.extend(pattern_lessons)
        
        # 4. LLM deep analysis if available
        if self.llm is not None:
            try:
                llm_lessons = await self._llm_analysis(trajectory, playbook_context)
                lessons.extend(llm_lessons)
            except Exception as e:
                print(f"LLM reflection failed: {e}")
        
        # Record reflection
        self.reflection_history.append({
            "trajectory_id": trajectory.trajectory_id,
            "timestamp": datetime.now().isoformat(),
            "n_lessons": len(lessons),
            "run_number": trajectory.run_number
        })
        
        return lessons
    
    async def reflect_on_experiment(
        self,
        experiment: ImprovementExperiment,
        trajectories: List[Trajectory]
    ) -> List[Lesson]:
        """
        Perform reflection on a complete improvement experiment.
        
        This is the main reflection entry point for self-improvement loops.
        Analyzes ablation results to extract precise lessons.
        
        Args:
            experiment: The improvement experiment
            trajectories: All trajectories from the experiment
            
        Returns:
            List of lessons from the experiment
        """
        lessons = []
        
        # 1. Analyze ablation results - most valuable for learning
        ablation_lessons = self._analyze_ablations(experiment)
        lessons.extend(ablation_lessons)
        
        # 2. Compare trajectories across iterations
        comparison_lessons = self._compare_iterations(trajectories, experiment)
        lessons.extend(comparison_lessons)
        
        # 3. Overall experiment insights
        experiment_lessons = self._analyze_experiment_outcome(experiment)
        lessons.extend(experiment_lessons)
        
        # 4. LLM synthesis if available
        if self.llm is not None:
            try:
                synthesis = await self._llm_experiment_synthesis(experiment, trajectories)
                lessons.extend(synthesis)
            except Exception as e:
                print(f"LLM experiment synthesis failed: {e}")
        
        return lessons
    
    def _analyze_successes(self, trajectory: Trajectory) -> List[Lesson]:
        """Extract lessons from successful actions"""
        lessons = []
        
        for step in trajectory.get_successful_steps():
            # Determine domain from action type
            domain = self._action_type_to_domain(step.action_type)
            
            # Build evidence
            evidence = {
                "action_name": step.action_name,
                "inputs": step.action_inputs,
                "metric_delta": step.metric_delta,
                "metric_after": step.metric_after
            }
            
            lesson = Lesson(
                source_trajectories=[trajectory.trajectory_id],
                lesson_type=LessonType.SUCCESS_PATTERN,
                domain=domain,
                title=f"{step.action_name} improved performance",
                summary=f"{step.action_name} with inputs {self._summarize_inputs(step.action_inputs)} improved performance by {step.metric_delta:.4f}" if step.metric_delta else f"{step.action_name} was successful",
                detailed_analysis=f"In {trajectory.task_type} task for {trajectory.cancer_type or 'general'} cancer, {step.action_name} led to improvement.",
                applicable_conditions={
                    "task_type": trajectory.task_type,
                    "cancer_type": trajectory.cancer_type,
                    "n_samples_range": self._categorize_samples(trajectory.dataset_info.get("n_samples", 0))
                },
                evidence=evidence,
                attributed_changes=[{
                    "action": step.action_name,
                    "inputs": step.action_inputs,
                    "impact": step.metric_delta
                }],
                avg_improvement=step.metric_delta or 0.0,
                confidence=0.6 + min(0.3, abs(step.metric_delta or 0) * 5),
                recommendations=[f"Consider using {step.action_name} in similar contexts"]
            )
            lessons.append(lesson)
        
        return lessons
    
    def _analyze_failures(self, trajectory: Trajectory) -> List[Lesson]:
        """Extract lessons from failed actions"""
        lessons = []
        
        for step in trajectory.get_failed_steps():
            domain = self._action_type_to_domain(step.action_type)
            
            lesson = Lesson(
                source_trajectories=[trajectory.trajectory_id],
                lesson_type=LessonType.FAILURE_PATTERN,
                domain=domain,
                title=f"Avoid: {step.action_name} hurt performance",
                summary=f"{step.action_name} decreased performance" + (f" by {abs(step.metric_delta):.4f}" if step.metric_delta else ""),
                detailed_analysis=f"In {trajectory.task_type} task, {step.action_name} with inputs {self._summarize_inputs(step.action_inputs)} was counterproductive.",
                applicable_conditions={
                    "task_type": trajectory.task_type,
                    "cancer_type": trajectory.cancer_type
                },
                evidence={
                    "action_name": step.action_name,
                    "inputs": step.action_inputs,
                    "metric_delta": step.metric_delta
                },
                avg_improvement=step.metric_delta or 0.0,
                confidence=0.5,
                recommendations=[f"Avoid {step.action_name} in similar contexts", "Consider alternative approaches"]
            )
            lessons.append(lesson)
        
        return lessons
    
    def _identify_patterns(self, trajectory: Trajectory) -> List[Lesson]:
        """Identify patterns across the trajectory"""
        lessons = []
        
        # Pattern: Best model for this context
        if trajectory.best_model and trajectory.best_score > 0:
            lesson = Lesson(
                source_trajectories=[trajectory.trajectory_id],
                lesson_type=LessonType.IMPROVEMENT_INSIGHT,
                domain="model_selection",
                title=f"{trajectory.best_model} performed best",
                summary=f"{trajectory.best_model} achieved score {trajectory.best_score:.4f} for {trajectory.task_type}",
                detailed_analysis=f"For {trajectory.cancer_type or 'general'} {trajectory.task_type} with ~{trajectory.dataset_info.get('n_samples', 'unknown')} samples, {trajectory.best_model} was optimal.",
                applicable_conditions={
                    "task_type": trajectory.task_type,
                    "cancer_type": trajectory.cancer_type,
                    "n_samples_range": self._categorize_samples(trajectory.dataset_info.get("n_samples", 0)),
                    "n_features_range": self._categorize_features(trajectory.dataset_info.get("n_features", 0))
                },
                evidence={
                    "best_model": trajectory.best_model,
                    "best_score": trajectory.best_score,
                    "dataset_info": trajectory.dataset_info
                },
                avg_improvement=trajectory.best_score,
                confidence=0.7,
                recommendations=[f"Prioritize {trajectory.best_model} for similar tasks"]
            )
            lessons.append(lesson)
        
        # Pattern: Feature engineering impact
        fe_steps = trajectory.get_steps_by_type(ActionType.FEATURE_ENGINEERING)
        if fe_steps:
            total_impact = sum(s.metric_delta or 0 for s in fe_steps)
            if abs(total_impact) > 0.01:
                lesson = Lesson(
                    source_trajectories=[trajectory.trajectory_id],
                    lesson_type=LessonType.IMPROVEMENT_INSIGHT if total_impact > 0 else LessonType.WARNING,
                    domain="feature_interaction",
                    title=f"Feature engineering had {'positive' if total_impact > 0 else 'negative'} impact",
                    summary=f"Feature engineering steps had cumulative impact of {total_impact:+.4f}",
                    detailed_analysis=f"Across {len(fe_steps)} feature engineering operations, net impact was {total_impact:+.4f}",
                    applicable_conditions={
                        "task_type": trajectory.task_type,
                        "cancer_type": trajectory.cancer_type
                    },
                    avg_improvement=total_impact,
                    confidence=0.6
                )
                lessons.append(lesson)
        
        return lessons
    
    def _analyze_ablations(self, experiment: ImprovementExperiment) -> List[Lesson]:
        """
        Analyze ablation results to extract precise change attributions.
        
        This is the most valuable source of lessons - we know exactly
        what change caused what improvement.
        """
        lessons = []
        
        for result in experiment.ablation_results:
            if result.is_beneficial:
                # Successful change - high confidence lesson
                lesson = Lesson(
                    source_trajectories=experiment.iteration_trajectories,
                    lesson_type=LessonType.SUCCESS_PATTERN,
                    domain=self._improvement_type_to_domain(result.change_type),
                    title=f"Beneficial: {result.change_description}",
                    summary=f"{result.change_description} improved performance by {result.improvement:.4f} ({result.improvement_percentage:.1f}%)",
                    detailed_analysis=f"Ablation testing confirmed: {result.change_description} is beneficial. Baseline: {result.baseline_score:.4f}, After: {result.score_with_change:.4f}",
                    applicable_conditions={
                        "baseline_model": experiment.baseline_model
                    },
                    evidence={
                        "change_type": result.change_type.value,
                        "change_details": result.change_details,
                        "baseline_score": result.baseline_score,
                        "score_with_change": result.score_with_change,
                        "improvement": result.improvement,
                        "ablation_verified": True
                    },
                    attributed_changes=[{
                        "type": result.change_type.value,
                        "description": result.change_description,
                        "details": result.change_details,
                        "impact": result.improvement
                    }],
                    avg_improvement=result.improvement,
                    confidence=0.85,  # High confidence - ablation verified
                    recommendations=[f"Apply: {result.change_description}"]
                )
                lessons.append(lesson)
            
            elif result.improvement < -0.005:
                # Harmful change
                lesson = Lesson(
                    source_trajectories=experiment.iteration_trajectories,
                    lesson_type=LessonType.WARNING,
                    domain=self._improvement_type_to_domain(result.change_type),
                    title=f"Avoid: {result.change_description}",
                    summary=f"{result.change_description} hurt performance by {abs(result.improvement):.4f}",
                    detailed_analysis=f"Ablation testing confirmed: {result.change_description} is harmful in this context.",
                    evidence={
                        "change_type": result.change_type.value,
                        "change_details": result.change_details,
                        "improvement": result.improvement,
                        "ablation_verified": True
                    },
                    avg_improvement=result.improvement,
                    confidence=0.8,
                    recommendations=[f"Avoid: {result.change_description} in similar contexts"]
                )
                lessons.append(lesson)
        
        return lessons
    
    def _compare_iterations(
        self,
        trajectories: List[Trajectory],
        experiment: ImprovementExperiment
    ) -> List[Lesson]:
        """Compare trajectories across improvement iterations"""
        lessons = []
        
        if len(trajectories) < 2:
            return lessons
        
        # Sort by run number
        sorted_trajs = sorted(trajectories, key=lambda t: t.run_number)
        
        # Compare first and last
        first = sorted_trajs[0]
        last = sorted_trajs[-1]
        
        if last.best_score > first.best_score:
            total_improvement = last.best_score - first.best_score
            
            lesson = Lesson(
                source_trajectories=[t.trajectory_id for t in trajectories],
                lesson_type=LessonType.IMPROVEMENT_INSIGHT,
                domain="general",
                title=f"Improvement loop successful: +{total_improvement:.4f}",
                summary=f"Over {len(trajectories)} iterations, performance improved from {first.best_score:.4f} to {last.best_score:.4f}",
                detailed_analysis=f"Self-improvement loop achieved {total_improvement:.4f} improvement through {len(experiment.successful_changes)} beneficial changes.",
                evidence={
                    "n_iterations": len(trajectories),
                    "initial_score": first.best_score,
                    "final_score": last.best_score,
                    "total_improvement": total_improvement,
                    "successful_changes": experiment.successful_changes
                },
                avg_improvement=total_improvement,
                confidence=0.9,
                recommendations=["The improvement strategies used here can be applied to similar tasks"]
            )
            lessons.append(lesson)
        
        return lessons
    
    def _analyze_experiment_outcome(self, experiment: ImprovementExperiment) -> List[Lesson]:
        """Analyze overall experiment outcome"""
        lessons = []
        
        summary = experiment.get_improvement_summary()
        
        if summary["beneficial_changes"] > 0:
            # Record what types of changes worked
            change_types = {}
            for result in experiment.ablation_results:
                if result.is_beneficial:
                    ctype = result.change_type.value
                    if ctype not in change_types:
                        change_types[ctype] = []
                    change_types[ctype].append(result.improvement)
            
            for ctype, improvements in change_types.items():
                avg_imp = sum(improvements) / len(improvements)
                lesson = Lesson(
                    source_trajectories=experiment.iteration_trajectories,
                    lesson_type=LessonType.IMPROVEMENT_INSIGHT,
                    domain=self._improvement_type_to_domain_str(ctype),
                    title=f"{ctype} changes are effective",
                    summary=f"{ctype} changes averaged {avg_imp:.4f} improvement across {len(improvements)} trials",
                    evidence={
                        "change_type": ctype,
                        "n_trials": len(improvements),
                        "avg_improvement": avg_imp,
                        "all_improvements": improvements
                    },
                    avg_improvement=avg_imp,
                    confidence=0.7 + min(0.2, len(improvements) * 0.05)
                )
                lessons.append(lesson)
        
        return lessons
    
    async def _llm_analysis(
        self,
        trajectory: Trajectory,
        playbook_context: Optional[str]
    ) -> List[Lesson]:
        """Use LLM for deep trajectory analysis"""
        if self.llm is None:
            return []
        
        summary = self._build_trajectory_summary(trajectory)
        
        prompt = f"""Analyze this ML experiment trajectory and extract lessons.

TRAJECTORY:
{summary}

EXISTING KNOWLEDGE:
{playbook_context or "None"}

Extract 1-3 KEY lessons. Focus on:
1. What specific decisions helped/hurt?
2. What generalizable patterns emerge?
3. What domain-specific (oncology) insights apply?

Return JSON:
{{
    "lessons": [
        {{
            "type": "success_pattern|failure_pattern|improvement_insight|warning",
            "domain": "feature_interaction|model_selection|preprocessing|clinical_pattern",
            "title": "Short title",
            "summary": "One sentence summary",
            "conditions": {{"task_type": "...", "cancer_type": "..."}},
            "confidence": 0.5-0.9,
            "recommendation": "What to do"
        }}
    ]
}}
"""
        
        try:
            response = await self.llm.complete_json(prompt)
            
            lessons = []
            for data in response.get("lessons", [])[:3]:
                lesson = Lesson(
                    source_trajectories=[trajectory.trajectory_id],
                    lesson_type=LessonType(data.get("type", "improvement_insight")),
                    domain=data.get("domain", "general"),
                    title=data.get("title", ""),
                    summary=data.get("summary", ""),
                    applicable_conditions=data.get("conditions", {}),
                    confidence=data.get("confidence", 0.5),
                    recommendations=[data.get("recommendation", "")] if data.get("recommendation") else []
                )
                lessons.append(lesson)
            
            return lessons
        except Exception as e:
            print(f"LLM analysis error: {e}")
            return []
    
    async def _llm_experiment_synthesis(
        self,
        experiment: ImprovementExperiment,
        trajectories: List[Trajectory]
    ) -> List[Lesson]:
        """LLM synthesis of improvement experiment"""
        if self.llm is None:
            return []
        
        summary = experiment.get_improvement_summary()
        
        prompt = f"""Synthesize learnings from this self-improvement experiment.

EXPERIMENT SUMMARY:
- Iterations: {experiment.current_iteration}
- Baseline: {experiment.baseline_score:.4f} ({experiment.baseline_model})
- Final: {experiment.final_score:.4f}
- Total improvement: {experiment.total_improvement:.4f}

CHANGES TRIED:
{json.dumps([{
    'type': r.change_type.value,
    'description': r.change_description,
    'improvement': r.improvement,
    'beneficial': r.is_beneficial
} for r in experiment.ablation_results], indent=2)}

What are the 1-2 most important lessons from this experiment?

Return JSON:
{{
    "lessons": [
        {{
            "title": "Key insight",
            "summary": "What we learned",
            "domain": "feature_interaction|model_selection|preprocessing",
            "recommendation": "What to do in future",
            "confidence": 0.7-0.95
        }}
    ]
}}
"""
        
        try:
            response = await self.llm.complete_json(prompt)
            
            lessons = []
            for data in response.get("lessons", [])[:2]:
                lesson = Lesson(
                    source_trajectories=experiment.iteration_trajectories,
                    lesson_type=LessonType.IMPROVEMENT_INSIGHT,
                    domain=data.get("domain", "general"),
                    title=data.get("title", ""),
                    summary=data.get("summary", ""),
                    confidence=data.get("confidence", 0.7),
                    recommendations=[data.get("recommendation", "")] if data.get("recommendation") else []
                )
                lessons.append(lesson)
            
            return lessons
        except Exception as e:
            print(f"LLM synthesis error: {e}")
            return []
    
    def _build_trajectory_summary(self, trajectory: Trajectory) -> str:
        """Build human-readable trajectory summary"""
        lines = []
        lines.append(f"Task: {trajectory.task_type}, Cancer: {trajectory.cancer_type or 'general'}")
        lines.append(f"Dataset: {trajectory.dataset_info.get('n_samples', '?')} samples")
        lines.append(f"Result: {trajectory.best_model} with {trajectory.best_score:.4f}")
        lines.append(f"Run: {'Baseline' if trajectory.run_number == 0 else f'Iteration {trajectory.run_number}'}")
        
        if trajectory.improvement_delta is not None:
            lines.append(f"Improvement: {trajectory.improvement_delta:+.4f}")
        
        lines.append("\nSteps:")
        for step in trajectory.steps:
            symbol = {"success": "+", "failure": "-", "neutral": "o", "error": "!"}[step.outcome.value]
            delta = f" ({step.metric_delta:+.4f})" if step.metric_delta else ""
            lines.append(f"  {symbol} {step.action_name}{delta}")
        
        return "\n".join(lines)
    
    def _summarize_inputs(self, inputs: Dict[str, Any]) -> str:
        """Create brief summary of inputs"""
        if not inputs:
            return "{}"
        
        key_parts = []
        for k, v in list(inputs.items())[:3]:
            if isinstance(v, str) and len(v) < 30:
                key_parts.append(f"{k}={v}")
            elif isinstance(v, (int, float)):
                key_parts.append(f"{k}={v}")
        
        return "{" + ", ".join(key_parts) + "}" if key_parts else "{...}"
    
    def _action_type_to_domain(self, action_type: ActionType) -> str:
        """Map action type to domain"""
        mapping = {
            ActionType.DATA_ANALYSIS: "preprocessing",
            ActionType.FEATURE_ENGINEERING: "feature_interaction",
            ActionType.FEATURE_REFINEMENT: "feature_interaction",
            ActionType.MODEL_SELECTION: "model_selection",
            ActionType.MODEL_TRAINING: "model_selection",
            ActionType.HYPERPARAMETER_TUNING: "hyperparameter",
            ActionType.MODEL_EVALUATION: "model_selection",
            ActionType.ERROR_ANALYSIS: "error_pattern",
            ActionType.INTERPRETABILITY: "clinical_pattern"
        }
        return mapping.get(action_type, "general")
    
    def _improvement_type_to_domain(self, imp_type) -> str:
        """Map improvement type to domain"""
        from .schemas import ImprovementType
        mapping = {
            ImprovementType.FEATURE_INTERACTION: "feature_interaction",
            ImprovementType.FEATURE_TRANSFORM: "feature_interaction",
            ImprovementType.MODEL_CHANGE: "model_selection",
            ImprovementType.HYPERPARAMETER: "hyperparameter",
            ImprovementType.PREPROCESSING: "preprocessing",
            ImprovementType.ENSEMBLE: "model_selection"
        }
        return mapping.get(imp_type, "general")
    
    def _improvement_type_to_domain_str(self, type_str: str) -> str:
        """Map improvement type string to domain"""
        mapping = {
            "feature_interaction": "feature_interaction",
            "feature_transform": "feature_interaction",
            "model_change": "model_selection",
            "hyperparameter": "hyperparameter",
            "preprocessing": "preprocessing",
            "ensemble": "model_selection"
        }
        return mapping.get(type_str, "general")
    
    async def _categorize_dataset_llm(self, n_samples: int, n_features: int, task_type: str) -> Dict[str, str]:
        """Use LLM to categorize dataset characteristics contextually"""
        if self.llm is None:
            return self._categorize_dataset_fallback(n_samples, n_features)
        
        prompt = f"""Categorize this ONCOLOGY dataset for ML.

DATASET:
- Samples: {n_samples} patients
- Features: {n_features} variables  
- Task: {task_type}

For clinical oncology data, categorize:
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
            print(f"LLM categorization failed: {e}")
            return self._categorize_dataset_fallback(n_samples, n_features)
    
    def _categorize_dataset_fallback(self, n_samples: int, n_features: int) -> Dict[str, str]:
        """Fallback categorization without LLM"""
        return {
            "n_samples": "small" if n_samples < 500 else "medium" if n_samples < 5000 else "large",
            "n_features": "low" if n_features < 20 else "medium" if n_features < 100 else "high"
        }
    
    def _categorize_samples(self, n: int) -> str:
        """Simple categorization for sync contexts"""
        if n < 500:
            return "small"
        elif n < 5000:
            return "medium"
        return "large"
    
    def _categorize_features(self, n: int) -> str:
        """Simple categorization for sync contexts"""
        if n < 20:
            return "low"
        elif n < 100:
            return "medium"
        return "high"

