"""
ACE - Agentic Context Engineering Framework

A self-improving framework for ML agents that learns from experience
through structured trajectory analysis, reflection, and playbook evolution.

Based on the ACE paper (arxiv.org/abs/2510.04618), adapted for clinical ML.

Components:
- Generator: Tracks execution trajectories
- Reflector: Extracts lessons from trajectories  
- Curator: Manages the evolving playbook
- Controller: Orchestrates self-improvement loops
"""

from .schemas import (
    ActionType,
    ActionOutcome,
    TrajectoryStep,
    Trajectory,
    Lesson,
    DeltaItem,
    PlaybookDomain,
    Playbook,
    ImprovementExperiment,
    AblationResult
)

from .generator import TrajectoryGenerator
from .reflector import TrajectoryReflector
from .curator import PlaybookCurator
from .controller import ImprovementController

__all__ = [
    # Schemas
    "ActionType",
    "ActionOutcome", 
    "TrajectoryStep",
    "Trajectory",
    "Lesson",
    "DeltaItem",
    "PlaybookDomain",
    "Playbook",
    "ImprovementExperiment",
    "AblationResult",
    # Components
    "TrajectoryGenerator",
    "TrajectoryReflector",
    "PlaybookCurator",
    "ImprovementController",
]

