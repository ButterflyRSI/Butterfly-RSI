# ButterflyRSI
### Recursive Self-Correcting Intelligence Framework

**AI memory system with drift detection and dream consolidation**

[![License: Custom](https://img.shields.io/badge/License-Butterfly_AI_Sovereign-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![No Dependencies](https://img.shields.io/badge/dependencies-none-green.svg)]()

## The Problem

Current AI memory systems accumulate context but can't distinguish signal from noise. Over time, they drift, degrade, and lose coherence.

- No quality filtering (garbage in = garbage retained)
- No drift detection (system doesn't know when it's diverging)
- No consolidation strategy (can't store everything forever)
- Result: Memory systems that get worse over time

## The Solution

ButterflyRSI solves this through three core innovations:

### 1. Drift Detection & Self-Correction
The system monitors its own stability in real-time and corrects when diverging. Your AI knows when it's losing coherence and self-corrects.

### 2. Dream Consolidation
Inspired by neuroscience (sleep memory replay): Selects only high-quality memories, synthesizes insights from valuable experiences, prevents noise accumulation.

### 3. Personality Evolution
6-dimensional trait system that adapts without losing identity: Analytical, Creative, Empathic, Strategic, Curious, Defensive. Continuity Index tracks personality stability. Adaptive but coherent.

## Quick Start
```python
from butterfly_rsi_v4_5 import MemoryCore

ai = MemoryCore(
    persona_name="ResearchAssistant",
    domain_focus=["analysis", "reasoning"],
    ethical_constraints=["educational use", "cite sources"]
)

ai.record_interaction(
    prompt="Analyze this data pattern",
    response="Detected trend X with confidence Y"
)

ai.add_reflection("Pattern recognition improving. Need to validate assumptions.")
ai.add_feedback(0.92, "Excellent analysis")
ai.mirror_loop()
ai.dream_consolidation()
print(ai.generate_report())
```

No external dependencies. Pure Python stdlib.

## Why This Matters

### Built From Necessity
This framework wasn't built in a lab. It was developed to solve real cognitive scaffolding needs - managing complex medical protocols, tracking stability under constraints, and maintaining coherence when traditional support systems failed.

### Inspired by Neuroscience
The "dream consolidation" mechanism mimics how biological brains process memories during sleep. Not all experiences are equal. Valuable patterns get replayed and strengthened. Low-quality noise is filtered out. Intelligent learning, not just accumulation.

### Recursive Self-Correction
Unlike other AI memory systems, ButterflyRSI knows when it's drifting (stability monitoring), corrects automatically (homeostatic regulation), maintains coherence (continuity tracking), and is self-aware of its own state.

## The Evolution: v2.5 to v4.5

ButterflyRSI wasn't built overnight. Here's how it evolved over a year of development.

### v2.5 (Early 2024): Proof of Concept
Core innovations established: Dream consolidation metaphor, stability/drift tracking, mirror loop architecture, domain-specific personas.

Limitations: 150 lines single monolithic class, dictionary-based data structures, random memory selection, no personality system.

Key insight: The biological inspiration was right - dream consolidation works.

### v4.5 (Current): Production Framework
Architectural maturation: 5 specialized components (analyzers, engines, consolidators), dataclasses with full type hints, quality-weighted selection (not random), 6-dimensional personality evolution, conditional consolidation (only above threshold), 500 lines of production-ready code.

What changed: From random to intelligent memory selection, from static to adaptive personality system, from monolithic to modular architecture, from prototype to production quality.

The journey demonstrates sustained development over months, real-world testing (built while managing complex medical protocols), iterative refinement based on actual use. Not vaporware - executed and evolved.

## Architecture
```
MemoryCore (Orchestration)
    ├── ReflectionAnalyzer (Quality scoring)
    ├── PersonalityEngine (Trait evolution)
    ├── StabilityAnalyzer (Drift detection & correction)
    └── DreamConsolidator (Memory selection)
```

Each component is independently testable, single responsibility, clearly documented, and type-safe.

## Installation

No installation needed. Pure Python stdlib.
```bash
git clone https://github.com/ButterflyRSI/butterfly-rsi.git
cd butterfly-rsi
python butterfly_rsi_v4_5.py
```

Requirements: Python 3.8+
Dependencies: None. Zero. Nada.

## License

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
See LICENSE file for full details.

## Origin Story

This framework emerged from necessity, not theory. Built while managing post-transplant medical protocols, complex pharmacological optimization, housing instability, and real cognitive scaffolding needs.

The result: a system that actually works because it had to.

Key insight: Your AI should know when it's losing coherence - just like humans need to recognize when they're drifting.

## Theoretical Foundation

ButterflyRSI emerged from fundamental questions about feedback, computation, and consciousness.

My original whitepaper "The Deeper You Go, The More It Thinks Like You" explored how feedback loops manifest across computational layers, from Python to quantum circuits, mirroring biological cognition.

Key insight: Self-regulation through feedback is universal. From cells maintaining homeostasis to neurons processing information, feedback enables adaptation and learning.

ButterflyRSI implements this principle: Mirror loop = feedback mechanism, Drift detection = homeostatic regulation, Dream consolidation = synaptic pruning, Personality evolution = adaptive plasticity, Self-correction = biological intelligence.

Theory to Prototype to Production.

## Contact & Licensing

For inquiries:
- Email: rtsliwinski@gmail.com
- GitHub: @ButterflyRSI

For collaboration or questions: Open an issue or reach out directly.

## Citation

If you use ButterflyRSI in research or projects:
```
@software{butterflyrsi_2024,
  author = {Rich Sliwinski (RJ)},
  title = {ButterflyRSI: Recursive Self-Correcting Intelligence Framework},
  year = {2024},
  url = {https://github.com/ButterflyRSI/butterfly-rsi}
}
```

## What Makes This Different?

vs. Traditional AI Memory:

| Feature | Traditional | ButterflyRSI |
|---------|------------|---------------|
| Memory Selection | Append-only | Quality-weighted |
| Drift Detection | None | Real-time monitoring |
| Self-Correction | None | Automatic homeostasis |
| Consolidation | None | Intelligent replay |
| Personality | Static | Evolving (6D traits) |
| Self-Awareness | None | Stability tracking |

ButterflyRSI is self-aware, self-correcting memory - it knows when it's working and when it's not.

Ready to try it? Clone the repo and run the demo. No dependencies, no bullshit.

Built by someone who refused die and decided to learn quantum physics instead.
