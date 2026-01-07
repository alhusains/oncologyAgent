"""
ACE Data Structures

Core schemas for the Agentic Context Engineering framework:
- Trajectories: Execution traces with outcomes
- Lessons: Extracted insights from trajectories
- Delta Items: Atomic knowledge units in the playbook
- Playbook: Evolving knowledge base
- Improvement experiments: Self-improvement tracking
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from enum import Enum
import uuid
import json


class ActionType(str, Enum):
    """Types of actions tracked in trajectories"""
    DATA_ANALYSIS = "data_analysis"
    FEATURE_ENGINEERING = "feature_engineering"
    MODEL_SELECTION = "model_selection"
    MODEL_TRAINING = "model_training"
    MODEL_EVALUATION = "model_evaluation"
    ERROR_ANALYSIS = "error_analysis"
    FEATURE_REFINEMENT = "feature_refinement"
    HYPERPARAMETER_TUNING = "hyperparameter_tuning"
    INTERPRETABILITY = "interpretability"


class ActionOutcome(str, Enum):
    """Outcome classification for actions"""
    SUCCESS = "success"          # Action improved results
    NEUTRAL = "neutral"          # No significant impact
    FAILURE = "failure"          # Action hurt results
    ERROR = "error"              # Action failed to execute


class LessonType(str, Enum):
    """Types of lessons extracted"""
    SUCCESS_PATTERN = "success_pattern"
    FAILURE_PATTERN = "failure_pattern"
    IMPROVEMENT_INSIGHT = "improvement_insight"
    WARNING = "warning"
    DOMAIN_KNOWLEDGE = "domain_knowledge"


class ImprovementType(str, Enum):
    """Types of improvements the agent can try"""
    FEATURE_INTERACTION = "feature_interaction"
    FEATURE_TRANSFORM = "feature_transform"
    MODEL_CHANGE = "model_change"
    HYPERPARAMETER = "hyperparameter"
    PREPROCESSING = "preprocessing"
    ENSEMBLE = "ensemble"


@dataclass
class TrajectoryStep:
    """Single step in an execution trajectory"""
    step_id: int
    action_type: ActionType
    action_name: str                    # e.g., "train_model"
    action_inputs: Dict[str, Any]       # Parameters used
    action_outputs: Dict[str, Any]      # Results
    outcome: ActionOutcome
    reasoning: str                      # LLM's reasoning for this action
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Performance tracking
    metric_before: Optional[float] = None
    metric_after: Optional[float] = None
    metric_delta: Optional[float] = None
    
    # Context at time of action
    context_snapshot: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_id": self.step_id,
            "action_type": self.action_type.value,
            "action_name": self.action_name,
            "action_inputs": self.action_inputs,
            "action_outputs": self.action_outputs,
            "outcome": self.outcome.value,
            "reasoning": self.reasoning,
            "timestamp": self.timestamp.isoformat(),
            "metric_before": self.metric_before,
            "metric_after": self.metric_after,
            "metric_delta": self.metric_delta,
            "context_snapshot": self.context_snapshot
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrajectoryStep":
        return cls(
            step_id=data["step_id"],
            action_type=ActionType(data["action_type"]),
            action_name=data["action_name"],
            action_inputs=data["action_inputs"],
            action_outputs=data["action_outputs"],
            outcome=ActionOutcome(data["outcome"]),
            reasoning=data.get("reasoning", ""),
            timestamp=datetime.fromisoformat(data["timestamp"]) if data.get("timestamp") else datetime.now(),
            metric_before=data.get("metric_before"),
            metric_after=data.get("metric_after"),
            metric_delta=data.get("metric_delta"),
            context_snapshot=data.get("context_snapshot", {})
        )


@dataclass
class Trajectory:
    """
    Complete execution trajectory for one experiment run.
    
    Captures the full reasoning chain and outcomes for reflection.
    """
    trajectory_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    experiment_id: str = ""
    run_number: int = 0  # 0 = baseline, 1+ = improvement iterations
    parent_trajectory_id: Optional[str] = None  # Links to previous run
    created_at: datetime = field(default_factory=datetime.now)
    
    # Dataset context
    dataset_info: Dict[str, Any] = field(default_factory=dict)
    cancer_type: Optional[str] = None
    task_type: str = "classification"
    
    # Execution steps
    steps: List[TrajectoryStep] = field(default_factory=list)
    
    # Changes from baseline (for improvement runs)
    changes_from_baseline: List[Dict[str, Any]] = field(default_factory=list)
    
    # Final outcomes
    final_metrics: Dict[str, float] = field(default_factory=dict)
    best_model: Optional[str] = None
    best_score: float = 0.0
    
    # Comparison with baseline
    baseline_score: Optional[float] = None
    improvement_delta: Optional[float] = None
    
    # Metadata
    total_duration_seconds: float = 0.0
    completed: bool = False
    
    def add_step(self, step: TrajectoryStep):
        """Add a step to the trajectory"""
        step.step_id = len(self.steps)
        self.steps.append(step)
    
    def get_steps_by_type(self, action_type: ActionType) -> List[TrajectoryStep]:
        """Get all steps of a specific type"""
        return [s for s in self.steps if s.action_type == action_type]
    
    def get_successful_steps(self) -> List[TrajectoryStep]:
        """Get all steps that improved performance"""
        return [s for s in self.steps if s.outcome == ActionOutcome.SUCCESS]
    
    def get_failed_steps(self) -> List[TrajectoryStep]:
        """Get all steps that hurt performance"""
        return [s for s in self.steps if s.outcome == ActionOutcome.FAILURE]
    
    def compute_net_improvement(self) -> float:
        """Compute net improvement from all steps"""
        return sum(s.metric_delta or 0 for s in self.steps)
    
    def finalize(self, final_metrics: Dict[str, float], best_model: str, best_score: float):
        """Finalize the trajectory with final results"""
        self.final_metrics = final_metrics
        self.best_model = best_model
        self.best_score = best_score
        self.total_duration_seconds = (datetime.now() - self.created_at).total_seconds()
        self.completed = True
        
        if self.baseline_score is not None:
            self.improvement_delta = best_score - self.baseline_score
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "trajectory_id": self.trajectory_id,
            "experiment_id": self.experiment_id,
            "run_number": self.run_number,
            "parent_trajectory_id": self.parent_trajectory_id,
            "created_at": self.created_at.isoformat(),
            "dataset_info": self.dataset_info,
            "cancer_type": self.cancer_type,
            "task_type": self.task_type,
            "steps": [s.to_dict() for s in self.steps],
            "changes_from_baseline": self.changes_from_baseline,
            "final_metrics": self.final_metrics,
            "best_model": self.best_model,
            "best_score": self.best_score,
            "baseline_score": self.baseline_score,
            "improvement_delta": self.improvement_delta,
            "total_duration_seconds": self.total_duration_seconds,
            "completed": self.completed
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Trajectory":
        traj = cls(
            trajectory_id=data.get("trajectory_id", str(uuid.uuid4())),
            experiment_id=data.get("experiment_id", ""),
            run_number=data.get("run_number", 0),
            parent_trajectory_id=data.get("parent_trajectory_id"),
            dataset_info=data.get("dataset_info", {}),
            cancer_type=data.get("cancer_type"),
            task_type=data.get("task_type", "classification"),
            changes_from_baseline=data.get("changes_from_baseline", []),
            final_metrics=data.get("final_metrics", {}),
            best_model=data.get("best_model"),
            best_score=data.get("best_score", 0.0),
            baseline_score=data.get("baseline_score"),
            improvement_delta=data.get("improvement_delta"),
            total_duration_seconds=data.get("total_duration_seconds", 0.0),
            completed=data.get("completed", False)
        )
        traj.steps = [TrajectoryStep.from_dict(s) for s in data.get("steps", [])]
        return traj


@dataclass
class Lesson:
    """
    A lesson extracted from trajectory analysis.
    Gets converted to DeltaItem(s) by the Curator.
    """
    lesson_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source_trajectories: List[str] = field(default_factory=list)  # trajectory_ids
    lesson_type: LessonType = LessonType.IMPROVEMENT_INSIGHT
    
    # What was learned
    title: str = ""
    summary: str = ""
    detailed_analysis: str = ""
    
    # Domain classification
    domain: str = "general"  # feature_interaction, model_selection, etc.
    
    # Context for applicability
    applicable_conditions: Dict[str, Any] = field(default_factory=dict)
    
    # Evidence
    evidence: Dict[str, Any] = field(default_factory=dict)
    attributed_changes: List[Dict[str, Any]] = field(default_factory=list)
    
    # Impact
    avg_improvement: float = 0.0
    confidence: float = 0.5
    
    # Recommendations
    recommendations: List[str] = field(default_factory=list)
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "lesson_id": self.lesson_id,
            "source_trajectories": self.source_trajectories,
            "lesson_type": self.lesson_type.value,
            "title": self.title,
            "summary": self.summary,
            "detailed_analysis": self.detailed_analysis,
            "domain": self.domain,
            "applicable_conditions": self.applicable_conditions,
            "evidence": self.evidence,
            "attributed_changes": self.attributed_changes,
            "avg_improvement": self.avg_improvement,
            "confidence": self.confidence,
            "recommendations": self.recommendations,
            "created_at": self.created_at.isoformat()
        }


@dataclass
class DeltaItem:
    """
    Atomic unit of knowledge in the playbook.
    
    Delta items are structured, incremental updates that:
    - Have unique identifiers for tracking
    - Include metadata for relevance matching
    - Can be merged, updated, or deprecated
    """
    item_id: str = field(default_factory=lambda: f"delta_{uuid.uuid4().hex[:12]}")
    domain: str = "general"
    
    # The actual knowledge
    title: str = ""
    content: str = ""
    strategy: str = ""  # Actionable strategy
    conditions: Dict[str, Any] = field(default_factory=dict)
    
    # Evidence and confidence
    evidence_sources: List[str] = field(default_factory=list)  # lesson_ids
    evidence_count: int = 0
    confidence: float = 0.5
    success_rate: float = 0.5
    avg_improvement: float = 0.0
    
    # Usage tracking
    usage_count: int = 0
    last_used: Optional[datetime] = None
    successful_uses: int = 0
    failed_uses: int = 0
    
    # Lifecycle
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    deprecated: bool = False
    superseded_by: Optional[str] = None
    
    def update_with_evidence(self, success: bool, improvement: float = 0.0):
        """Update item based on new usage evidence"""
        self.usage_count += 1
        self.last_used = datetime.now()
        self.updated_at = datetime.now()
        
        if success:
            self.successful_uses += 1
        else:
            self.failed_uses += 1
        
        # Update success rate
        if self.usage_count > 0:
            self.success_rate = self.successful_uses / self.usage_count
        
        # Update avg improvement (exponential moving average)
        alpha = 0.3
        self.avg_improvement = alpha * improvement + (1 - alpha) * self.avg_improvement
        
        # Update confidence based on evidence
        self.confidence = min(0.95, 0.4 + (self.evidence_count * 0.05) + (self.success_rate * 0.3))
    
    def to_prompt_format(self) -> str:
        """Format for injection into LLM prompts"""
        conf_pct = f"{self.confidence:.0%}"
        success_pct = f"{self.success_rate:.0%}" if self.usage_count > 0 else "N/A"
        return f"- [{self.domain.upper()}] {self.title}: {self.strategy} (conf: {conf_pct}, success: {success_pct})"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "item_id": self.item_id,
            "domain": self.domain,
            "title": self.title,
            "content": self.content,
            "strategy": self.strategy,
            "conditions": self.conditions,
            "evidence_sources": self.evidence_sources,
            "evidence_count": self.evidence_count,
            "confidence": self.confidence,
            "success_rate": self.success_rate,
            "avg_improvement": self.avg_improvement,
            "usage_count": self.usage_count,
            "last_used": self.last_used.isoformat() if self.last_used else None,
            "successful_uses": self.successful_uses,
            "failed_uses": self.failed_uses,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "deprecated": self.deprecated,
            "superseded_by": self.superseded_by
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DeltaItem":
        item = cls(
            item_id=data.get("item_id", f"delta_{uuid.uuid4().hex[:12]}"),
            domain=data.get("domain", "general"),
            title=data.get("title", ""),
            content=data.get("content", ""),
            strategy=data.get("strategy", ""),
            conditions=data.get("conditions", {}),
            evidence_sources=data.get("evidence_sources", []),
            evidence_count=data.get("evidence_count", 0),
            confidence=data.get("confidence", 0.5),
            success_rate=data.get("success_rate", 0.5),
            avg_improvement=data.get("avg_improvement", 0.0),
            usage_count=data.get("usage_count", 0),
            successful_uses=data.get("successful_uses", 0),
            failed_uses=data.get("failed_uses", 0),
            deprecated=data.get("deprecated", False),
            superseded_by=data.get("superseded_by")
        )
        if data.get("last_used"):
            item.last_used = datetime.fromisoformat(data["last_used"])
        if data.get("created_at"):
            item.created_at = datetime.fromisoformat(data["created_at"])
        if data.get("updated_at"):
            item.updated_at = datetime.fromisoformat(data["updated_at"])
        return item


@dataclass
class PlaybookDomain:
    """A domain-specific section of the playbook"""
    domain_name: str
    description: str = ""
    items: Dict[str, DeltaItem] = field(default_factory=dict)
    
    def get_applicable_items(
        self,
        conditions: Dict[str, Any],
        min_confidence: float = 0.3,
        max_items: int = 10
    ) -> List[DeltaItem]:
        """Get items applicable to current conditions, sorted by relevance"""
        applicable = []
        
        for item in self.items.values():
            if item.deprecated or item.confidence < min_confidence:
                continue
            
            # Score based on condition matching and confidence
            match_score = self._compute_match_score(item.conditions, conditions)
            if match_score > 0.2:  # Minimum threshold
                relevance = match_score * item.confidence * (0.5 + item.success_rate * 0.5)
                applicable.append((relevance, item))
        
        # Sort by relevance and return top items
        applicable.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in applicable[:max_items]]
    
    def _compute_match_score(self, item_conditions: Dict, current: Dict) -> float:
        """Compute how well item conditions match current context"""
        if not item_conditions:
            return 0.5  # Generic items get moderate score
        
        matches = 0
        total = len(item_conditions)
        
        for key, value in item_conditions.items():
            if key not in current:
                continue
            
            current_val = current[key]
            
            if isinstance(value, list):
                if current_val in value:
                    matches += 1
            elif value == current_val:
                matches += 1
            elif isinstance(value, str) and isinstance(current_val, str):
                if value.lower() in current_val.lower() or current_val.lower() in value.lower():
                    matches += 0.5
        
        return matches / total if total > 0 else 0.5
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "domain_name": self.domain_name,
            "description": self.description,
            "items": {k: v.to_dict() for k, v in self.items.items()}
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PlaybookDomain":
        domain = cls(
            domain_name=data.get("domain_name", ""),
            description=data.get("description", "")
        )
        domain.items = {
            k: DeltaItem.from_dict(v) 
            for k, v in data.get("items", {}).items()
        }
        return domain


@dataclass
class Playbook:
    """
    The evolving knowledge base that accumulates domain knowledge.
    
    This is the heart of ACE - a structured, versioned collection
    of learned strategies that grows and improves over time.
    """
    playbook_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    version: int = 1
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    # Domain-specific knowledge sections
    domains: Dict[str, PlaybookDomain] = field(default_factory=dict)
    
    # Statistics
    total_trajectories_processed: int = 0
    total_lessons_extracted: int = 0
    total_experiments: int = 0
    
    # Changelog
    changelog: List[Dict[str, Any]] = field(default_factory=list)
    
    def __post_init__(self):
        """Initialize default domains if not present"""
        default_domains = {
            "feature_interaction": "Learned feature interactions and transformations",
            "model_selection": "Model selection strategies for different contexts",
            "preprocessing": "Data preprocessing and handling strategies",
            "clinical_pattern": "Oncology-specific clinical patterns",
            "error_pattern": "Error patterns and how to avoid them",
            "hyperparameter": "Hyperparameter tuning insights"
        }
        for name, desc in default_domains.items():
            if name not in self.domains:
                self.domains[name] = PlaybookDomain(domain_name=name, description=desc)
    
    @property
    def total_items(self) -> int:
        return sum(len(d.items) for d in self.domains.values())
    
    def get_context_for_prompt(
        self,
        conditions: Dict[str, Any],
        max_items_per_domain: int = 3,
        max_total_items: int = 12
    ) -> str:
        """Get playbook context formatted for LLM prompts"""
        all_items = []
        
        for domain in self.domains.values():
            items = domain.get_applicable_items(
                conditions,
                min_confidence=0.4,
                max_items=max_items_per_domain
            )
            all_items.extend(items)
        
        # Sort by confidence * success_rate and take top items
        all_items.sort(key=lambda x: x.confidence * x.success_rate, reverse=True)
        top_items = all_items[:max_total_items]
        
        if not top_items:
            return ""
        
        lines = ["\n=== LEARNED STRATEGIES (from previous experiments) ==="]
        for item in top_items:
            lines.append(item.to_prompt_format())
        lines.append("=" * 50)
        
        return "\n".join(lines)
    
    def add_log_entry(self, action: str, details: Dict[str, Any]):
        """Add entry to changelog"""
        self.changelog.append({
            "action": action,
            "timestamp": datetime.now().isoformat(),
            **details
        })
        # Keep last 200 entries
        if len(self.changelog) > 200:
            self.changelog = self.changelog[-200:]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "playbook_id": self.playbook_id,
            "version": self.version,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "domains": {k: v.to_dict() for k, v in self.domains.items()},
            "total_trajectories_processed": self.total_trajectories_processed,
            "total_lessons_extracted": self.total_lessons_extracted,
            "total_experiments": self.total_experiments,
            "changelog": self.changelog[-50:]  # Save last 50
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Playbook":
        pb = cls(
            playbook_id=data.get("playbook_id", str(uuid.uuid4())),
            version=data.get("version", 1),
            total_trajectories_processed=data.get("total_trajectories_processed", 0),
            total_lessons_extracted=data.get("total_lessons_extracted", 0),
            total_experiments=data.get("total_experiments", 0),
            changelog=data.get("changelog", [])
        )
        if data.get("created_at"):
            pb.created_at = datetime.fromisoformat(data["created_at"])
        if data.get("updated_at"):
            pb.updated_at = datetime.fromisoformat(data["updated_at"])
        
        pb.domains = {}
        for k, v in data.get("domains", {}).items():
            pb.domains[k] = PlaybookDomain.from_dict(v)
        
        # Ensure all default domains exist
        pb.__post_init__()
        
        return pb


@dataclass
class AblationResult:
    """Result of a single ablation test (one change)"""
    change_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    change_type: ImprovementType = ImprovementType.FEATURE_INTERACTION
    change_description: str = ""
    change_details: Dict[str, Any] = field(default_factory=dict)
    
    # Results
    baseline_score: float = 0.0
    score_with_change: float = 0.0
    improvement: float = 0.0
    improvement_percentage: float = 0.0
    
    # Assessment
    is_beneficial: bool = False
    confidence: float = 0.5
    
    # Timing
    execution_time_seconds: float = 0.0
    
    def compute_improvement(self):
        """Compute improvement metrics"""
        self.improvement = self.score_with_change - self.baseline_score
        if self.baseline_score > 0:
            self.improvement_percentage = (self.improvement / self.baseline_score) * 100
        self.is_beneficial = self.improvement > 0.001  # 0.1% threshold


@dataclass
class ImprovementExperiment:
    """
    Tracks a complete self-improvement experiment with multiple iterations.
    
    This is the main structure for the improvement loop.
    """
    experiment_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    
    # Configuration
    max_iterations: int = 3
    current_iteration: int = 0
    
    # Baseline
    baseline_trajectory_id: Optional[str] = None
    baseline_score: float = 0.0
    baseline_model: Optional[str] = None
    
    # Trajectories for each iteration
    iteration_trajectories: List[str] = field(default_factory=list)  # trajectory_ids
    
    # Ablation results
    ablation_results: List[AblationResult] = field(default_factory=list)
    
    # Final results
    final_score: float = 0.0
    final_model: Optional[str] = None
    total_improvement: float = 0.0
    successful_changes: List[Dict[str, Any]] = field(default_factory=list)
    
    # Status
    completed: bool = False
    stopped_early: bool = False
    stop_reason: Optional[str] = None
    
    def add_ablation_result(self, result: AblationResult):
        """Add an ablation result"""
        self.ablation_results.append(result)
        if result.is_beneficial:
            self.successful_changes.append({
                "change_id": result.change_id,
                "type": result.change_type.value,
                "description": result.change_description,
                "improvement": result.improvement
            })
    
    def get_improvement_summary(self) -> Dict[str, Any]:
        """Get summary of improvements"""
        beneficial = [r for r in self.ablation_results if r.is_beneficial]
        harmful = [r for r in self.ablation_results if r.improvement < -0.001]
        
        return {
            "total_changes_tried": len(self.ablation_results),
            "beneficial_changes": len(beneficial),
            "harmful_changes": len(harmful),
            "neutral_changes": len(self.ablation_results) - len(beneficial) - len(harmful),
            "total_improvement": self.total_improvement,
            "improvement_percentage": (self.total_improvement / self.baseline_score * 100) if self.baseline_score > 0 else 0,
            "best_change": max(self.ablation_results, key=lambda x: x.improvement).change_description if self.ablation_results else None
        }
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "session_id": self.session_id,
            "created_at": self.created_at.isoformat(),
            "max_iterations": self.max_iterations,
            "current_iteration": self.current_iteration,
            "baseline_trajectory_id": self.baseline_trajectory_id,
            "baseline_score": self.baseline_score,
            "baseline_model": self.baseline_model,
            "iteration_trajectories": self.iteration_trajectories,
            "ablation_results": [
                {
                    "change_id": r.change_id,
                    "change_type": r.change_type.value,
                    "change_description": r.change_description,
                    "change_details": r.change_details,
                    "baseline_score": r.baseline_score,
                    "score_with_change": r.score_with_change,
                    "improvement": r.improvement,
                    "is_beneficial": r.is_beneficial
                }
                for r in self.ablation_results
            ],
            "final_score": self.final_score,
            "final_model": self.final_model,
            "total_improvement": self.total_improvement,
            "successful_changes": self.successful_changes,
            "completed": self.completed,
            "stopped_early": self.stopped_early,
            "stop_reason": self.stop_reason
        }

