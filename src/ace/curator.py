"""
ACE Curator Component

Manages the playbook - converts lessons to delta items and maintains
the evolving knowledge base.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path
import json
import uuid

from .schemas import Playbook, PlaybookDomain, DeltaItem, Lesson, LessonType


class PlaybookCurator:
    """
    Manages the evolving playbook of learned knowledge.
    
    The Curator is responsible for:
    - Converting lessons to structured delta items
    - Merging new items with existing knowledge
    - Deduplicating and pruning the playbook
    - Persistence and versioning
    """
    
    def __init__(
        self,
        playbook_path: str = "knowledge/playbook.json",
        auto_save: bool = True
    ):
        self.playbook_path = Path(playbook_path)
        self.playbook_path.parent.mkdir(parents=True, exist_ok=True)
        self.auto_save = auto_save
        
        # Similarity threshold for merging (0.8 = 80% similar)
        self.merge_threshold = 0.75
        
        # Load or create playbook
        self.playbook = self._load_or_create()
        
        # Seed with domain knowledge if new
        if self.playbook.total_items == 0:
            self._seed_oncology_knowledge()
    
    def curate_lessons(self, lessons: List[Lesson]) -> Dict[str, Any]:
        """
        Process lessons from the Reflector and update the playbook.
        
        Args:
            lessons: List of lessons to process
            
        Returns:
            Summary of curation results
        """
        results = {
            "items_created": 0,
            "items_updated": 0,
            "items_merged": 0,
            "processed_lessons": len(lessons)
        }
        
        for lesson in lessons:
            # Convert lesson to delta item
            delta_item = self._lesson_to_delta(lesson)
            
            # Find similar existing items
            similar = self._find_similar_items(delta_item)
            
            if similar:
                # Merge with most similar
                best_match = similar[0]
                self._merge_items(best_match, delta_item, lesson)
                results["items_merged"] += 1
            else:
                # Add as new
                self._add_item(delta_item)
                results["items_created"] += 1
        
        # Update statistics
        self.playbook.total_lessons_extracted += len(lessons)
        self.playbook.updated_at = datetime.now()
        self.playbook.version += 1
        
        # Periodic maintenance
        if self.playbook.total_lessons_extracted % 20 == 0:
            self._maintenance()
        
        # Save if auto-save enabled
        if self.auto_save:
            self.save()
        
        return results
    
    def get_context_for_prompt(
        self,
        conditions: Dict[str, Any],
        max_items: int = 10
    ) -> str:
        """
        Get playbook context formatted for LLM prompts.
        
        This is how the playbook influences agent behavior.
        """
        return self.playbook.get_context_for_prompt(conditions, max_items_per_domain=3, max_total_items=max_items)
    
    def get_strategies_for_improvement(
        self,
        conditions: Dict[str, Any],
        focus_domains: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Get actionable strategies for self-improvement.
        
        Returns strategies sorted by expected impact.
        """
        strategies = []
        
        domains_to_check = focus_domains or list(self.playbook.domains.keys())
        
        for domain_name in domains_to_check:
            domain = self.playbook.domains.get(domain_name)
            if not domain:
                continue
            
            items = domain.get_applicable_items(conditions, min_confidence=0.5, max_items=5)
            
            for item in items:
                if item.avg_improvement > 0:
                    strategies.append({
                        "item_id": item.item_id,
                        "domain": item.domain,
                        "title": item.title,
                        "strategy": item.strategy or item.content,
                        "expected_improvement": item.avg_improvement,
                        "confidence": item.confidence,
                        "success_rate": item.success_rate,
                        "usage_count": item.usage_count,
                        "conditions": item.conditions
                    })
        
        # Sort by expected impact (improvement * confidence)
        strategies.sort(key=lambda x: x["expected_improvement"] * x["confidence"], reverse=True)
        
        return strategies
    
    def record_strategy_usage(
        self,
        item_id: str,
        success: bool,
        improvement: float = 0.0
    ):
        """
        Record that a strategy was used and its outcome.
        
        This feedback updates the playbook's confidence scores.
        """
        for domain in self.playbook.domains.values():
            if item_id in domain.items:
                domain.items[item_id].update_with_evidence(success, improvement)
                
                self.playbook.add_log_entry("strategy_used", {
                    "item_id": item_id,
                    "success": success,
                    "improvement": improvement
                })
                
                if self.auto_save:
                    self.save()
                return
    
    def _lesson_to_delta(self, lesson: Lesson) -> DeltaItem:
        """Convert a lesson to a delta item"""
        # Build strategy from recommendations
        strategy = ""
        if lesson.recommendations:
            strategy = lesson.recommendations[0]
        elif lesson.summary:
            strategy = lesson.summary
        
        # Build content
        content = lesson.detailed_analysis or lesson.summary
        
        delta = DeltaItem(
            item_id=f"delta_{lesson.lesson_id[:12]}",
            domain=lesson.domain,
            title=lesson.title[:100] if lesson.title else lesson.summary[:100],
            content=content[:500] if content else "",
            strategy=strategy[:200] if strategy else "",
            conditions=lesson.applicable_conditions,
            evidence_sources=[lesson.lesson_id],
            evidence_count=1,
            confidence=lesson.confidence,
            success_rate=0.8 if lesson.lesson_type == LessonType.SUCCESS_PATTERN else 0.3 if lesson.lesson_type == LessonType.FAILURE_PATTERN else 0.5,
            avg_improvement=lesson.avg_improvement
        )
        
        return delta
    
    def _find_similar_items(self, item: DeltaItem) -> List[DeltaItem]:
        """Find similar items in the same domain"""
        domain = self.playbook.domains.get(item.domain)
        if not domain:
            return []
        
        similar = []
        for existing in domain.items.values():
            if existing.deprecated:
                continue
            
            similarity = self._compute_similarity(item, existing)
            if similarity >= self.merge_threshold:
                similar.append((similarity, existing))
        
        similar.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in similar]
    
    def _compute_similarity(self, item1: DeltaItem, item2: DeltaItem) -> float:
        """Compute similarity between two items"""
        # Title similarity (word overlap)
        words1 = set(item1.title.lower().split())
        words2 = set(item2.title.lower().split())
        
        if not words1 or not words2:
            title_sim = 0.0
        else:
            title_sim = len(words1 & words2) / len(words1 | words2)
        
        # Content similarity
        content_words1 = set(item1.content.lower().split())
        content_words2 = set(item2.content.lower().split())
        
        if not content_words1 or not content_words2:
            content_sim = 0.0
        else:
            content_sim = len(content_words1 & content_words2) / len(content_words1 | content_words2)
        
        # Condition overlap
        if item1.conditions and item2.conditions:
            common_keys = set(item1.conditions.keys()) & set(item2.conditions.keys())
            matching = sum(1 for k in common_keys if item1.conditions[k] == item2.conditions[k])
            condition_sim = matching / len(common_keys) if common_keys else 0.5
        else:
            condition_sim = 0.5
        
        # Weighted combination
        return 0.4 * title_sim + 0.4 * content_sim + 0.2 * condition_sim
    
    def _merge_items(self, existing: DeltaItem, new: DeltaItem, lesson: Lesson):
        """Merge new item into existing"""
        # Add evidence
        if lesson.lesson_id not in existing.evidence_sources:
            existing.evidence_sources.append(lesson.lesson_id)
        existing.evidence_count = len(existing.evidence_sources)
        
        # Update confidence with more evidence
        existing.confidence = min(0.95, existing.confidence + 0.05)
        
        # Update average improvement (exponential moving average)
        if new.avg_improvement != 0:
            alpha = 1 / (existing.evidence_count + 1)
            existing.avg_improvement = (1 - alpha) * existing.avg_improvement + alpha * new.avg_improvement
        
        # Update success rate based on lesson type
        if lesson.lesson_type == LessonType.SUCCESS_PATTERN:
            existing.successful_uses += 1
        elif lesson.lesson_type == LessonType.FAILURE_PATTERN:
            existing.failed_uses += 1
        
        total = existing.successful_uses + existing.failed_uses
        if total > 0:
            existing.success_rate = existing.successful_uses / total
        
        existing.updated_at = datetime.now()
        
        self.playbook.add_log_entry("item_merged", {
            "item_id": existing.item_id,
            "merged_lesson": lesson.lesson_id
        })
    
    def _add_item(self, item: DeltaItem):
        """Add a new item to the playbook"""
        domain = self.playbook.domains.get(item.domain)
        if not domain:
            domain = PlaybookDomain(domain_name=item.domain, description=f"Auto-created: {item.domain}")
            self.playbook.domains[item.domain] = domain
        
        domain.items[item.item_id] = item
        
        self.playbook.add_log_entry("item_added", {
            "item_id": item.item_id,
            "domain": item.domain,
            "title": item.title
        })
    
    def _maintenance(self):
        """Periodic maintenance: deduplicate, prune low-quality items"""
        items_removed = 0
        
        for domain in self.playbook.domains.values():
            to_remove = []
            
            for item_id, item in domain.items.items():
                # Remove deprecated
                if item.deprecated:
                    to_remove.append(item_id)
                    continue
                
                # Remove very low confidence with no usage
                if item.confidence < 0.25 and item.usage_count == 0 and item.evidence_count < 2:
                    to_remove.append(item_id)
                    continue
                
                # Deprecate consistently failing items
                if item.usage_count >= 3 and item.success_rate < 0.15:
                    item.deprecated = True
                    to_remove.append(item_id)
            
            for item_id in to_remove:
                del domain.items[item_id]
                items_removed += 1
        
        if items_removed > 0:
            self.playbook.add_log_entry("maintenance", {"items_removed": items_removed})
    
    def _seed_oncology_knowledge(self):
        """
        Seed playbook with minimal initial examples (1 per domain).
        
        These are just examples to bootstrap the learning process.
        The agent will build its own knowledge progressively through experience.
        """
        seed_items = [
            # Feature interaction - example
            DeltaItem(
                item_id="seed_fi_example",
                domain="feature_interaction",
                title="Age-tumor interaction for prognosis",
                content="Clinical features often interact meaningfully. Example: age * tumor_size captures age-adjusted burden.",
                strategy="Consider creating interactions between clinical variables that may have multiplicative effects",
                conditions={},
                confidence=0.5,
                success_rate=0.5,
                avg_improvement=0.0
            ),
            
            # Preprocessing - example
            DeltaItem(
                item_id="seed_pp_example",
                domain="preprocessing",
                title="Biomarkers often need log transform",
                content="Many biomarkers (PSA, CEA, CA-125) follow log-normal distributions in oncology data.",
                strategy="Try log1p transform on biomarker features if distributions are skewed",
                conditions={},
                confidence=0.5,
                success_rate=0.5,
                avg_improvement=0.0
            ),
            
            # Model selection - example
            DeltaItem(
                item_id="seed_ms_example",
                domain="model_selection",
                title="Tree-based models work well for clinical data",
                content="Random Forest and XGBoost handle mixed feature types and missing data well in clinical settings.",
                strategy="Start with tree-based models for tabular clinical data",
                conditions={},
                confidence=0.5,
                success_rate=0.5,
                avg_improvement=0.0
            ),
            
            # Clinical pattern - example  
            DeltaItem(
                item_id="seed_cp_example",
                domain="clinical_pattern",
                title="Cancer stage is usually highly predictive",
                content="TNM stage or overall stage is typically the strongest predictor in oncology outcomes.",
                strategy="Ensure stage information is properly encoded and used in modeling",
                conditions={},
                confidence=0.5,
                success_rate=0.5,
                avg_improvement=0.0
            ),
            
            # Error pattern - example
            DeltaItem(
                item_id="seed_ep_example",
                domain="error_pattern",
                title="Watch for overfitting on small datasets",
                content="Clinical datasets are often small; large train-test performance gaps indicate overfitting.",
                strategy="Use regularization and simpler models when overfitting is detected",
                conditions={},
                confidence=0.5,
                success_rate=0.5,
                avg_improvement=0.0
            ),
            
            # Hyperparameter - example
            DeltaItem(
                item_id="seed_hp_example",
                domain="hyperparameter",
                title="Conservative regularization for small samples",
                content="Small oncology datasets benefit from stronger regularization to prevent overfitting.",
                strategy="Use higher regularization strength (lower max_depth, higher min_samples_leaf)",
                conditions={},
                confidence=0.5,
                success_rate=0.5,
                avg_improvement=0.0
            ),
        ]
        
        for item in seed_items:
            self._add_item(item)
        
        self.playbook.add_log_entry("seeded", {"n_items": len(seed_items), "note": "minimal bootstrap examples"})
    
    def save(self):
        """Save playbook to disk"""
        with open(self.playbook_path, 'w') as f:
            json.dump(self.playbook.to_dict(), f, indent=2, default=str)
    
    def _load_or_create(self) -> Playbook:
        """Load existing playbook or create new one"""
        if self.playbook_path.exists():
            try:
                with open(self.playbook_path, 'r') as f:
                    data = json.load(f)
                return Playbook.from_dict(data)
            except Exception as e:
                print(f"Warning: Could not load playbook: {e}. Creating new one.")
        
        return Playbook()
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of playbook state"""
        return {
            "version": self.playbook.version,
            "total_items": self.playbook.total_items,
            "trajectories_processed": self.playbook.total_trajectories_processed,
            "lessons_extracted": self.playbook.total_lessons_extracted,
            "domains": {
                name: {
                    "n_items": len(domain.items),
                    "top_items": [
                        {"title": item.title, "confidence": item.confidence}
                        for item in sorted(domain.items.values(), key=lambda x: x.confidence, reverse=True)[:3]
                    ]
                }
                for name, domain in self.playbook.domains.items()
            }
        }
    
    def print_summary(self):
        """Print formatted playbook summary"""
        summary = self.get_summary()
        
        print("\n" + "=" * 60)
        print("PLAYBOOK SUMMARY")
        print("=" * 60)
        print(f"Version: {summary['version']}")
        print(f"Total Items: {summary['total_items']}")
        print(f"Trajectories Processed: {summary['trajectories_processed']}")
        print(f"Lessons Extracted: {summary['lessons_extracted']}")
        print("\nDomains:")
        for name, info in summary['domains'].items():
            print(f"  {name}: {info['n_items']} items")
            for item in info['top_items']:
                print(f"    - {item['title'][:50]}... (conf: {item['confidence']:.2f})")
        print("=" * 60)

