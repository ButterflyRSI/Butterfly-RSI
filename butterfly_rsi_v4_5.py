"""
Butterfly RSI v4.5 - Hybrid Recursive Self-History Engine
──────────────────────────────────────────────────────────────
Copyright 2024-2025 Rich Sliwinski

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
──────────────────────────────────────────────────────────────
Purpose:
- Combines v2.5's dream consolidation with v4.0's clean architecture
- Tracks both simple stability metrics AND complex trait evolution
- Supports domain-specific personas with concrete focus areas
- Implements intelligent reflection replay (not just iteration)
This is an experimental/artistic exploration of AI self-modeling.
Mathematical formulas are heuristic-based for exploratory purposes.
"""
import json
import statistics
import random
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from copy import deepcopy
from enum import Enum

# ═══════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════

@dataclass
class TraitConfig:
    """Configuration for personality trait evolution rates."""
    # Positive reinforcement rates
    analytical_feedback_rate: float = 0.10
    strategic_feedback_rate: float = 0.08
    # Creative/exploration rates
    creative_reflection_rate: float = 0.12
    curious_drift_rate: float = 0.07
    # Decay rates
    empathic_decay_rate: float = 0.05
    defensive_drift_rate: float = 0.10
    # Bounds
    trait_min: float = 0.0
    trait_max: float = 1.0
    # Stability thresholds (from v2.5)
    stability_threshold: float = 0.85  # Above this = "stable"
    consolidation_threshold: float = 0.80  # Minimum for dream consolidation


class StabilityState(Enum):
    """System stability classification (from v2.5 concept)."""
    STABLE = "stable"
    ADAPTIVE = "adaptive"
    VOLATILE = "volatile"


# ═══════════════════════════════════════════════════════════════
# DATA MODELS
# ═══════════════════════════════════════════════════════════════

@dataclass
class PersonalityTraits:
    """Represents the AI's personality trait values."""
    analytical: float = 0.5
    creative: float = 0.5
    empathic: float = 0.5
    strategic: float = 0.5
    curious: float = 0.5
    defensive: float = 0.5

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> 'PersonalityTraits':
        return cls(**data)

    def snapshot(self) -> 'PersonalityTraits':
        return PersonalityTraits(**self.to_dict())


@dataclass
class Feedback:
    """User feedback on AI response quality."""
    score: float  # 0.0 to 1.0
    note: str = ""

    def __post_init__(self):
        if not 0.0 <= self.score <= 1.0:
            raise ValueError(f"Feedback score must be 0-1, got {self.score}")


@dataclass
class MemoryEntry:
    """Single interaction record."""
    timestamp: str
    prompt: str
    response: str
    reflection: Optional[str] = None
    feedback: Optional[Feedback] = None
    reflection_quality: float = 0.0  # Computed score

    @classmethod
    def create(cls, prompt: str, response: str) -> 'MemoryEntry':
        return cls(
            timestamp=datetime.utcnow().isoformat(),
            prompt=prompt,
            response=response
        )


@dataclass
class StabilityMetrics:
    """Simple stability metrics (from v2.5)."""
    avg_feedback_score: float
    stability_state: StabilityState
    drift_probability: float
    coherence_index: float  # How aligned reflections are with feedback


@dataclass
class EvolutionInsight:
    """Result of one personality evolution cycle."""
    cycle: int
    stability_metrics: StabilityMetrics
    emotional_energy: float
    continuity_index: float
    trait_drift: Dict[str, float]
    dominant_trait: str
    mirror_summary: str
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class ConsolidatedMemory:
    """Result of dream consolidation process (from v2.5 concept)."""
    cycle: int
    source_reflections: List[str]
    synthesized_insight: str
    reinforcement_strength: float
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


# ═══════════════════════════════════════════════════════════════
# REFLECTION ANALYSIS
# ═══════════════════════════════════════════════════════════════

class ReflectionAnalyzer:
    """Analyzes reflection quality and extracts insights."""
    
    # Keywords for different cognitive patterns
    META_COGNITIVE_KEYWORDS = [
        'realize', 'understand', 'recognize', 'observe', 'notice',
        'aware', 'consider', 'reflect', 'think', 'evolve', 'improve',
        'learn', 'pattern', 'change', 'adapt', 'emerge'
    ]
    
    DOMAIN_KEYWORDS = {
        'cybersecurity': ['threat', 'vulnerability', 'attack', 'secure', 'encrypt', 'audit'],
        'psychology': ['emotion', 'cognition', 'behavior', 'empathy', 'consciousness'],
        'philosophy': ['ethics', 'reason', 'truth', 'meaning', 'existence'],
        'creativity': ['imagine', 'innovate', 'create', 'design', 'synthesize'],
    }

    @staticmethod
    def score_quality(reflection_text: str, domain_focus: List[str] = None) -> float:
        """Score reflection quality from 0.0 to 1.0."""
        if not reflection_text:
            return 0.0
        
        score = 0.5  # Base score
        text_lower = reflection_text.lower()
        
        # Length bonus (up to +0.15)
        word_count = len(reflection_text.split())
        length_bonus = min(0.15, word_count / 150 * 0.15)
        score += length_bonus
        
        # Meta-cognitive keyword bonus (up to +0.25)
        meta_count = sum(1 for kw in ReflectionAnalyzer.META_COGNITIVE_KEYWORDS
                        if kw in text_lower)
        meta_bonus = min(0.25, meta_count * 0.04)
        score += meta_bonus
        
        # Domain relevance bonus (up to +0.1)
        if domain_focus:
            domain_count = 0
            for domain in domain_focus:
                domain_lower = domain.lower()
                if domain_lower in ReflectionAnalyzer.DOMAIN_KEYWORDS:
                    domain_count += sum(1 for kw in
                                      ReflectionAnalyzer.DOMAIN_KEYWORDS[domain_lower]
                                      if kw in text_lower)
            domain_bonus = min(0.1, domain_count * 0.02)
            score += domain_bonus
        
        return min(1.0, score)

    @staticmethod
    def extract_key_insights(reflection_text: str, max_length: int = 100) -> str:
        """Extract most important part of reflection for consolidation."""
        if not reflection_text or len(reflection_text) <= max_length:
            return reflection_text
        
        # Simple heuristic: take sentences with meta-cognitive keywords
        sentences = reflection_text.split('.')
        scored_sentences = []
        
        for sentence in sentences:
            if not sentence.strip():
                continue
            score = sum(1 for kw in ReflectionAnalyzer.META_COGNITIVE_KEYWORDS
                       if kw in sentence.lower())
            scored_sentences.append((score, sentence.strip()))
        
        # Sort by score and take top sentences
        scored_sentences.sort(reverse=True, key=lambda x: x[0])
        key_insights = '. '.join(sent for _, sent in scored_sentences[:2])
        
        return key_insights[:max_length] + "..." if len(key_insights) > max_length else key_insights


# ═══════════════════════════════════════════════════════════════
# PERSONALITY ENGINE
# ═══════════════════════════════════════════════════════════════

class PersonalityEngine:
    """Manages personality trait evolution."""
    
    def __init__(self, config: Optional[TraitConfig] = None):
        self.config = config or TraitConfig()
        self.traits = PersonalityTraits()

    def evolve(self, reflection_score: float, feedback_quality: float) -> float:
        """Apply one evolution step to personality traits."""
        emotional_energy = (reflection_score + feedback_quality) / 2
        drift_factor = 1.0 - emotional_energy
        
        # Evolve each trait
        self.traits.analytical = self._clamp(
            self.traits.analytical + (feedback_quality * self.config.analytical_feedback_rate)
        )
        self.traits.creative = self._clamp(
            self.traits.creative + (reflection_score * self.config.creative_reflection_rate)
        )
        self.traits.empathic = self._clamp(
            self.traits.empathic - (drift_factor * self.config.empathic_decay_rate)
        )
        self.traits.strategic = self._clamp(
            self.traits.strategic + (feedback_quality * self.config.strategic_feedback_rate)
        )
        self.traits.curious = self._clamp(
            self.traits.curious + (drift_factor * self.config.curious_drift_rate)
        )
        self.traits.defensive = self._clamp(
            self.traits.defensive + (drift_factor * self.config.defensive_drift_rate)
        )
        
        return emotional_energy

    def _clamp(self, value: float) -> float:
        return max(self.config.trait_min, min(self.config.trait_max, value))

    def get_dominant_trait(self) -> str:
        traits_dict = self.traits.to_dict()
        return max(traits_dict, key=traits_dict.get)

    def snapshot(self) -> PersonalityTraits:
        return self.traits.snapshot()


# ═══════════════════════════════════════════════════════════════
# STABILITY ANALYZER (from v2.5 concept)
# ═══════════════════════════════════════════════════════════════

class StabilityAnalyzer:
    """Analyzes system stability using simple, interpretable metrics."""
    
    @staticmethod
    def analyze(
        feedback_scores: List[float],
        reflection_scores: List[float],
        config: TraitConfig
    ) -> StabilityMetrics:
        """Compute stability metrics from recent performance."""
        if not feedback_scores:
            return StabilityMetrics(
                avg_feedback_score=0.0,
                stability_state=StabilityState.VOLATILE,
                drift_probability=1.0,
                coherence_index=0.0
            )
        
        avg_feedback = statistics.mean(feedback_scores)
        avg_reflection = statistics.mean(reflection_scores) if reflection_scores else 0.5
        
        # Determine stability state
        if avg_feedback >= config.stability_threshold:
            state = StabilityState.STABLE
        elif avg_feedback >= 0.70:
            state = StabilityState.ADAPTIVE
        else:
            state = StabilityState.VOLATILE
        
        # Drift probability (inverse of performance)
        drift_prob = round(1.0 - avg_feedback, 3)
        
        # Coherence: how well reflections align with feedback
        coherence = round((avg_reflection + avg_feedback) / 2, 3)
        
        return StabilityMetrics(
            avg_feedback_score=round(avg_feedback, 3),
            stability_state=state,
            drift_probability=drift_prob,
            coherence_index=coherence
        )


# ═══════════════════════════════════════════════════════════════
# DREAM CONSOLIDATION ENGINE (from v2.5 concept, enhanced)
# ═══════════════════════════════════════════════════════════════

class DreamConsolidator:
    """Implements 'dream consolidation' - replaying and reinforcing key memories."""
    
    @staticmethod
    def select_valuable_memories(
        memory_log: List[MemoryEntry],
        count: int = 3
    ) -> List[MemoryEntry]:
        """Select most valuable memories for consolidation.
        
        Selection criteria:
        - High reflection quality
        - High feedback scores
        - Diverse timestamps (not all recent)
        """
        if not memory_log:
            return []
        
        # Filter memories with reflections and feedback
        valid_memories = [
            m for m in memory_log
            if m.reflection and m.feedback and m.reflection_quality > 0.5
        ]
        
        if not valid_memories:
            return []
        
        # Score each memory
        scored_memories = []
        for memory in valid_memories:
            score = (memory.reflection_quality + memory.feedback.score) / 2
            scored_memories.append((score, memory))
        
        # Sort by score and take top N
        scored_memories.sort(reverse=True, key=lambda x: x[0])
        selected = [m for _, m in scored_memories[:count]]
        
        return selected

    @staticmethod
    def synthesize_insights(memories: List[MemoryEntry]) -> str:
        """Synthesize key insights from selected memories."""
        if not memories:
            return "No memories to consolidate."
        
        analyzer = ReflectionAnalyzer()
        key_insights = []
        
        for memory in memories:
            if memory.reflection:
                insight = analyzer.extract_key_insights(memory.reflection, max_length=80)
                key_insights.append(insight)
        
        # Combine insights
        if not key_insights:
            return "Consolidation complete - patterns reinforced."
        
        synthesis = " ⟳ ".join(key_insights)
        return f"Consolidated insight: {synthesis}"

    @staticmethod
    def calculate_reinforcement(memories: List[MemoryEntry]) -> float:
        """Calculate how much these memories reinforce learning."""
        if not memories:
            return 0.0
        
        scores = [
            (m.reflection_quality + m.feedback.score) / 2
            for m in memories
        ]
        return round(statistics.mean(scores), 3)


# ═══════════════════════════════════════════════════════════════
# MEMORY CORE v4.5
# ═══════════════════════════════════════════════════════════════

class MemoryCore:
    """Hybrid system combining trait evolution with dream consolidation."""
    
    def __init__(
        self,
        persona_name: str,
        domain_focus: List[str],
        ethical_constraints: List[str],
        config: Optional[TraitConfig] = None
    ):
        self.persona_name = persona_name
        self.domain_focus = domain_focus
        self.ethical_constraints = ethical_constraints
        self.config = config or TraitConfig()
        
        self.memory_log: List[MemoryEntry] = []
        self.evolution_history: List[EvolutionInsight] = []
        self.consolidation_history: List[ConsolidatedMemory] = []
        
        self.personality_engine = PersonalityEngine(self.config)
        self.reflection_analyzer = ReflectionAnalyzer()
        self.stability_analyzer = StabilityAnalyzer()
        self.dream_consolidator = DreamConsolidator()
        
        self.cycle_count = 0

    # ───────────────────────────────────────────────────────────
    # INTERACTION RECORDING
    # ───────────────────────────────────────────────────────────
    
    def record_interaction(self, prompt: str, response: str) -> None:
        """Record a new user-AI interaction."""
        entry = MemoryEntry.create(prompt, response)
        self.memory_log.append(entry)

    def add_reflection(self, reflection_text: str) -> None:
        """Add self-reflection to the most recent interaction."""
        if not self.memory_log:
            raise ValueError("No interactions recorded yet")
        
        self.memory_log[-1].reflection = reflection_text
        
        # Score the reflection quality
        quality = self.reflection_analyzer.score_quality(
            reflection_text,
            self.domain_focus
        )
        self.memory_log[-1].reflection_quality = quality

    def add_feedback(self, score: float, note: str = "") -> None:
        """Add user feedback to the most recent interaction."""
        if not self.memory_log:
            raise ValueError("No interactions recorded yet")
        
        feedback = Feedback(score=score, note=note)
        self.memory_log[-1].feedback = feedback

    # ───────────────────────────────────────────────────────────
    # MIRROR LOOP (Enhanced from v2.5)
    # ───────────────────────────────────────────────────────────
    
    def mirror_loop(self) -> Optional[EvolutionInsight]:
        """Execute one mirror cycle - analyzes experiences and evolves."""
        # Gather metrics
        feedback_scores = [
            e.feedback.score for e in self.memory_log if e.feedback
        ]
        reflection_scores = [
            e.reflection_quality for e in self.memory_log
            if e.reflection_quality > 0
        ]
        
        if not feedback_scores:
            print(f"[{self.persona_name}] No feedback to analyze yet.")
            return None
        
        # Calculate stability metrics (v2.5 style)
        stability = self.stability_analyzer.analyze(
            feedback_scores,
            reflection_scores,
            self.config
        )
        
        # Store previous personality state
        prev_traits = self.personality_engine.snapshot()
        
        # Evolve personality (v4.0 style)
        avg_reflection = statistics.mean(reflection_scores) if reflection_scores else 0.5
        avg_feedback = stability.avg_feedback_score
        
        emotional_energy = self.personality_engine.evolve(
            avg_reflection,
            avg_feedback
        )
        
        # Compare personality change
        new_traits = self.personality_engine.snapshot()
        continuity_index, trait_drift = self._compare_traits(prev_traits, new_traits)
        
        # Generate insight
        self.cycle_count += 1
        mirror_summary = (
            f"Loop #{self.cycle_count}: {stability.stability_state.value} state. "
            f"Feedback: {avg_feedback:.2f}, Drift: {stability.drift_probability:.2f}, "
            f"Dominant: {self.personality_engine.get_dominant_trait()}"
        )
        
        insight = EvolutionInsight(
            cycle=self.cycle_count,
            stability_metrics=stability,
            emotional_energy=round(emotional_energy, 3),
            continuity_index=continuity_index,
            trait_drift=trait_drift,
            dominant_trait=self.personality_engine.get_dominant_trait(),
            mirror_summary=mirror_summary
        )
        
        self.evolution_history.append(insight)
        print(f"[{self.persona_name}] {mirror_summary}")
        return insight

    def _compare_traits(
        self,
        old: PersonalityTraits,
        new: PersonalityTraits
    ) -> Tuple[float, Dict[str, float]]:
        """Compare two personality states."""
        old_dict = old.to_dict()
        new_dict = new.to_dict()
        
        diffs = {trait: round(new_dict[trait] - old_dict[trait], 3) for trait in old_dict}
        total_shift = sum(abs(v) for v in diffs.values()) / len(diffs)
        continuity_index = round(1.0 - total_shift, 3)
        
        return continuity_index, diffs

    # ───────────────────────────────────────────────────────────
    # DREAM CONSOLIDATION (from v2.5 concept, enhanced)
    # ───────────────────────────────────────────────────────────
    
    def dream_consolidation(self, memory_count: int = 3) -> Optional[ConsolidatedMemory]:
        """Perform dream consolidation - replay and reinforce valuable memories.
        
        This simulates the consolidation process that happens during sleep/reflection,
        where important experiences are replayed and strengthened."""
        
        if not self.memory_log:
            print(f"[{self.persona_name}] No memories available for consolidation.")
            return None
        
        # Select valuable memories
        selected = self.dream_consolidator.select_valuable_memories(
            self.memory_log,
            count=memory_count
        )
        
        if not selected:
            print(f"[{self.persona_name}] No high-quality memories found for consolidation.")
            return None
        
        # Synthesize insights
        synthesis = self.dream_consolidator.synthesize_insights(selected)
        reinforcement = self.dream_consolidator.calculate_reinforcement(selected)
        
        # Create consolidated memory
        source_reflections = [m.reflection for m in selected if m.reflection]
        consolidated = ConsolidatedMemory(
            cycle=self.cycle_count,
            source_reflections=source_reflections,
            synthesized_insight=synthesis,
            reinforcement_strength=reinforcement
        )
        
        self.consolidation_history.append(consolidated)
        
        print(f"[{self.persona_name}] Dream consolidation complete. "
              f"Reinforcement: {reinforcement:.2f}")
        print(f" └─ {synthesis}")
        
        return consolidated

    # ───────────────────────────────────────────────────────────
    # RECURSIVE REPLAY (from v2.5, enhanced)
    # ───────────────────────────────────────────────────────────
    
    def recursive_replay(self, cycles: int = 3) -> List[EvolutionInsight]:
        """Run multiple mirror cycles with dream consolidation between them.
        
        This simulates a deeper reflective process where the system:
        1. Analyzes current state (mirror loop)
        2. Consolidates important memories (dream consolidation)
        3. Reinforces learning through replay"""
        
        print(f"\n[{self.persona_name}] Initiating recursive replay: {cycles} cycles")
        print("─" * 60)
        
        insights = []
        
        for i in range(cycles):
            print(f"\n🔄 Cycle {i+1}/{cycles}")
            
            # Run mirror loop
            insight = self.mirror_loop()
            if insight:
                insights.append(insight)
            
            # Perform dream consolidation if stability allows
            if insight and insight.stability_metrics.avg_feedback_score >= self.config.consolidation_threshold:
                self.dream_consolidation()
            else:
                print(f" └─ Skipping consolidation (below threshold)")
        
        print("\n─" * 60)
        print(f"[{self.persona_name}] Recursive replay complete. Homeostasis achieved.\n")
        
        return insights

    # ───────────────────────────────────────────────────────────
    # STATE PERSISTENCE
    # ───────────────────────────────────────────────────────────
    
    def save(self, filepath: str = "memory_core_v4_5.json") -> None:
        """Save complete system state to JSON file."""
        data = {
            "persona_name": self.persona_name,
            "domain_focus": self.domain_focus,
            "ethical_constraints": self.ethical_constraints,
            "cycle_count": self.cycle_count,
            "memory_log": [
                {
                    "timestamp": e.timestamp,
                    "prompt": e.prompt,
                    "response": e.response,
                    "reflection": e.reflection,
                    "reflection_quality": e.reflection_quality,
                    "feedback": asdict(e.feedback) if e.feedback else None
                }
                for e in self.memory_log
            ],
            "evolution_history": [
                {
                    **asdict(insight),
                    "stability_metrics": asdict(insight.stability_metrics)
                }
                for insight in self.evolution_history
            ],
            "consolidation_history": [
                asdict(cons) for cons in self.consolidation_history
            ],
            "current_traits": self.personality_engine.traits.to_dict()
        }
        
        path = Path(filepath)
        with path.open('w') as f:
            json.dump(data, f, indent=2)
        
        print(f"[{self.persona_name}] State saved to {filepath}")

    # ───────────────────────────────────────────────────────────
    # REPORTING
    # ───────────────────────────────────────────────────────────
    
    def generate_report(self) -> str:
        """Generate comprehensive report of system state."""
        if not self.evolution_history:
            return f"[{self.persona_name}] No evolution data available"
        
        current_traits = self.personality_engine.traits.to_dict()
        latest = self.evolution_history[-1]
        
        report = [
            f"\n{'='*70}",
            f"BUTTERFLY LOOP REPORT: {self.persona_name}",
            f"{'='*70}",
            f"\n📋 DOMAIN FOCUS: {', '.join(self.domain_focus)}",
            f"🔒 ETHICAL CONSTRAINTS: {', '.join(self.ethical_constraints)}",
            f"\n📊 STATISTICS:",
            f"   Total Cycles: {self.cycle_count}",
            f"   Total Interactions: {len(self.memory_log)}",
            f"   Dream Consolidations: {len(self.consolidation_history)}",
            f"\n🎭 CURRENT PERSONALITY TRAITS:",
        ]
        
        for trait, value in current_traits.items():
            bar = '█' * int(value * 25)
            report.append(f"   {trait:12s}: {value:.3f} {bar}")
        
        stability = latest.stability_metrics
        report.extend([
            f"\n🔍 CURRENT STATE:",
            f"   Stability: {stability.stability_state.value.upper()}",
            f"   Dominant Trait: {latest.dominant_trait}",
            f"   Feedback Score: {stability.avg_feedback_score:.3f}",
            f"   Continuity Index: {latest.continuity_index:.3f}",
            f"   Drift Probability: {stability.drift_probability:.3f}",
            f"   Coherence Index: {stability.coherence_index:.3f}",
            f"\n📈 TRAIT CHANGES (last cycle):",
        ])
        
        for trait, drift in latest.trait_drift.items():
            arrow = "↑" if drift > 0 else "↓" if drift < 0 else "→"
            report.append(f"   {trait:12s}: {drift:+.3f} {arrow}")
        
        if self.consolidation_history:
            latest_consolidation = self.consolidation_history[-1]
            report.extend([
                f"\n💭 LATEST DREAM CONSOLIDATION:",
                f"   Reinforcement: {latest_consolidation.reinforcement_strength:.3f}",
                f"   Insight: {latest_consolidation.synthesized_insight[:120]}...",
            ])
        
        report.append(f"{'='*70}\n")
        return "\n".join(report)


# ═══════════════════════════════════════════════════════════════
# DEMO USAGE
# ═══════════════════════════════════════════════════════════════

def main():
    """Demonstration of Butterfly Loop v4.5 with domain-specific persona."""
    print("\n🦋 Butterfly Loop v4.5 - Hybrid System Demonstration\n")
    
    # Create domain-specific persona (like v2.5's EchoSec)
    ai = MemoryCore(
        persona_name="EchoSec Nova",
        domain_focus=["cybersecurity", "AI reasoning", "emergent pattern learning"],
        ethical_constraints=["sandbox only", "no external systems", "educational + safety focus"]
    )
    
    print(f"Initialized: {ai.persona_name}")
    print(f"Domain Focus: {', '.join(ai.domain_focus)}\n")
    
    # Interaction 1 - Cybersecurity context
    ai.record_interaction(
        prompt="Scan threat model for session replay vulnerabilities.",
        response="Detected anomaly in session token entropy. Potential replay attack vector identified."
    )
    ai.add_reflection(
        "Need to focus more on entropy analysis in future iterations. "
        "Session token randomness below threshold indicates weak generation. "
        "Pattern suggests insufficient cryptographic seeding."
    )
    ai.add_feedback(0.92, "Excellent threat identification")
    
    # Interaction 2 - Code analysis
    ai.record_interaction(
        prompt="Audit recursive function for edge cases.",
        response="Found unbounded recursion edge case in tree traversal logic."
    )
    ai.add_reflection(
        "Implement recursion depth limiter to prevent stack overflow. "
        "Consider iterative alternative or tail-call optimization. "
        "This pattern appears in similar codebases - need systematic approach."
    )
    ai.add_feedback(0.89, "Good catch, needs optimization details")
    
    # Interaction 3 - Pattern recognition
    ai.record_interaction(
        prompt="Analyze authentication bypass attempts in logs.",
        response="Detected coordinated timing attack pattern across 47 endpoints."
    )
    ai.add_reflection(
        "Timing attacks reveal statistical patterns in authentication flow. "
        "Need to implement constant-time comparison for sensitive operations. "
        "This vulnerability class is systematic - audit entire codebase."
    )
    ai.add_feedback(0.94, "Outstanding pattern recognition")
    
    # Interaction 4 - Lower quality for contrast
    ai.record_interaction(
        prompt="Check API rate limits.",
        response="Rate limits configured correctly."
    )
    ai.add_reflection("Looks good.")
    ai.add_feedback(0.65, "Too superficial")
    
    # Run recursive replay with dream consolidation
    print("\n" + "="*70)
    print("RUNNING RECURSIVE REPLAY WITH DREAM CONSOLIDATION")
    print("="*70)
    
    ai.recursive_replay(cycles=4)
    
    # Generate final report
    print(ai.generate_report())
    
    # Save state
    ai.save("echosec_nova_v4_5.json")
    
    print("\n✨ Demonstration complete!")
    print("\nKey Features Demonstrated:")
    print(" ✓ Domain-specific persona (cybersecurity focus)")
    print(" ✓ Reflection quality scoring with domain keywords")
    print(" ✓ Simple stability metrics (stable/adaptive/volatile)")
    print(" ✓ Complex trait evolution system")
    print(" ✓ Dream consolidation - replays valuable memories")
    print(" ✓ Recursive replay with intelligent consolidation")
    print("\nThis combines the best of v2.5 and v4.0:")
    print(" • v2.5: Dream consolidation, domain focus, stability states")
    print(" • v4.0: Clean architecture, trait system, type safety")


if __name__ == "__main__":
    main()
