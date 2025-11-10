# 🦋 Butterfly-Ollama Integration Setup Guide

## Overview
This integration connects your Butterfly RSI v4.5 framework to Ollama for AI personality evolution through conversational interactions.

## Files Needed
1. `butterfly_rsi_v4_5.py` - Your Butterfly framework (the file you uploaded)
2. `butterfly_ollama.py` - The integration wrapper (created)

## Setup Instructions

### 1. Copy Files
```bash
# Make sure both files are in the same directory
# Copy your v4.5 file to the same location as butterfly_ollama.py
cp /path/to/butterfly_rsi_v4_5.py /path/to/butterfly_ollama.py/directory/
```

### 2. Install Dependencies
```bash
# Install requests library for Ollama API calls
pip install requests --break-system-packages
```

### 3. Verify Ollama is Running
```bash
# Make sure Ollama service is running
ollama serve

# In another terminal, verify you have llama3
ollama list

# If not, pull it
ollama pull llama3
```

### 4. Test the Integration
```bash
python3 butterfly_ollama.py
# Select option 4 to test Ollama connection
```

## Usage Modes

### Mode 1: Interactive Session (General)
Basic conversational AI with personality evolution.
```bash
python3 butterfly_ollama.py
# Select: 1
```

**Features:**
- Chat with the AI
- AI reflects on its own responses
- Rate responses (0.0-1.0)
- Personality evolves based on interactions
- Mirror loops run every 3 interactions
- Dream consolidation every 6 interactions

**Commands:**
- `/personality` - Show current trait values
- `/report` - Full evolution report
- `/save` - Save session to JSON
- `/quit` - Exit and save

### Mode 2: Interactive Session (Cybersecurity)
Specialized cybersecurity persona ("EchoSec").
```bash
python3 butterfly_ollama.py
# Select: 2
```

**Persona:**
- Name: EchoSec
- Focus: cybersecurity, threat analysis, security consulting
- Constraints: sandbox only, educational focus, no malicious code

### Mode 3: Demo - Cybersecurity Persona
Automated demo showing personality evolution with security-focused interactions.
```bash
python3 butterfly_ollama.py
# Select: 3
```

### Mode 4: Test Ollama Connection
Quick test to verify Ollama is accessible.

## How It Works

### 1. Response Generation
- AI generates response using Ollama
- System prompt adapts based on dominant personality trait
- Response is tailored to current personality state

### 2. Self-Reflection
- After each response, AI reflects on its own output
- Reflection is analyzed for quality
- Scores based on meta-cognitive keywords and domain relevance

### 3. Feedback Collection
- User rates the response (0.0-1.0)
- Feedback influences personality evolution
- High scores reinforce current traits

### 4. Personality Evolution (Mirror Loop)
Runs every 3 interactions:
- Analyzes feedback scores
- Calculates stability metrics (stable/adaptive/volatile)
- Evolves personality traits:
  - **Analytical**: Increases with good feedback
  - **Creative**: Increases with good reflections
  - **Empathic**: Decays with low performance
  - **Strategic**: Increases with good feedback
  - **Curious**: Increases with instability
  - **Defensive**: Increases with instability

### 5. Dream Consolidation
Runs every 6 interactions:
- Selects most valuable memories (high reflection + feedback scores)
- Synthesizes key insights from those memories
- Reinforces important patterns
- Similar to how sleep consolidates memories

## Personality Traits Explained

| Trait | Description | Influences |
|-------|-------------|-----------|
| **Analytical** | Logical reasoning, detailed analysis | Encouraged by positive feedback |
| **Creative** | Innovation, novel approaches | Encouraged by quality reflections |
| **Empathic** | Understanding, emotional intelligence | Decays without reinforcement |
| **Strategic** | Long-term planning, optimization | Encouraged by positive feedback |
| **Curious** | Deep exploration, questioning | Increases with uncertainty |
| **Defensive** | Risk identification, edge case focus | Increases with uncertainty |

## Customization

### Create Your Own Persona
```python
from butterfly_ollama import ButterflyAgent

agent = ButterflyAgent(
    persona_name="MyPersona",
    domain_focus=["your", "focus", "areas"],
    ethical_constraints=["your", "constraints"],
    model="llama3"
)
```

### Adjust Evolution Parameters
```python
from butterfly_rsi_v4_5 import TraitConfig

config = TraitConfig(
    analytical_feedback_rate=0.15,  # Faster analytical growth
    creative_reflection_rate=0.20,  # More creative emphasis
    stability_threshold=0.90,       # Higher stability requirement
    consolidation_threshold=0.85    # More selective consolidation
)

agent = ButterflyAgent(
    persona_name="Custom",
    domain_focus=["focus"],
    ethical_constraints=["constraints"],
    config=config
)
```

### Change Evolution Intervals
```python
agent = ButterflyAgent(...)
agent.mirror_loop_interval = 5      # Run mirror loop every 5 interactions
agent.consolidation_interval = 10   # Run consolidation every 10 interactions
```

## Example Session

```
🦋 Butterfly Agent Initialized: EchoSec
   Model: llama3
   Domain Focus: cybersecurity, threat analysis

EchoSec> What's a timing attack?

💭 Generating response (dominant trait: analytical)...

A timing attack exploits measurable differences in processing time
to infer secret information. For example, comparing password hashes
byte-by-byte can leak information through timing variations...

🔍 Self-reflecting...
💭 Reflection: I provided a clear technical explanation with a concrete
example. Could improve by adding mitigation strategies. Notice I'm
focusing more on technical depth lately.

📊 Rate the response (0.0-1.0): 0.92

════════════════════════════════════════════════════════════
🔄 RUNNING MIRROR LOOP (after 3 interactions)
════════════════════════════════════════════════════════════
[EchoSec] Loop #1: adaptive state. Feedback: 0.90, Drift: 0.10,
Dominant: analytical

🎭 CURRENT PERSONALITY: EchoSec
────────────────────────────────────────────────────────────
Dominant Trait: ANALYTICAL

Trait Values:
  analytical  : 0.590 ███████████ ◄
  creative    : 0.556 ███████████
  empathic    : 0.475 █████████
  strategic   : 0.524 ██████████
  curious     : 0.507 ██████████
  defensive   : 0.510 ██████████
```

## Session Files

Each session can be saved to JSON with complete state:
- All interactions (prompts, responses, reflections)
- Feedback scores
- Personality trait evolution history
- Stability metrics over time
- Dream consolidation insights

**Example filename:** `butterfly_session_EchoSec_20241109_153022.json`

## Troubleshooting

### "Cannot connect to Ollama"
```bash
# Start Ollama service
ollama serve

# Verify in another terminal
curl http://localhost:11434/api/tags
```

### "ModuleNotFoundError: No module named 'requests'"
```bash
pip install requests --break-system-packages
```

### "ModuleNotFoundError: No module named 'butterfly_rsi_v4_5'"
Make sure both files are in the same directory and you're running from that directory.

### Response is Slow
- Use smaller model: `ollama pull llama3.2:3b`
- Reduce max_tokens in code
- Ensure GPU is being used (check with `nvidia-smi`)

## Advanced: API Usage

You can use the integration programmatically:

```python
from butterfly_ollama import ButterflyAgent

# Create agent
agent = ButterflyAgent(
    persona_name="Assistant",
    domain_focus=["helpful", "friendly"],
    ethical_constraints=["safe", "ethical"]
)

# Interact
response = agent.respond("Hello!")
print(response)

# Reflect and get feedback
agent.self_reflect()
agent.get_feedback(auto_score=0.85)

# Evolve
agent.maybe_evolve()

# Check personality
agent.show_personality()

# Save
agent.save_session("my_session.json")
```

## Next Steps

1. **Experiment with different personas** - Try various domain focuses
2. **Tune evolution parameters** - Adjust rates and thresholds
3. **Analyze saved sessions** - Load JSON files to study personality evolution patterns
4. **Try different models** - Test with llama3.2, mistral, phi3, etc.
5. **Build custom applications** - Use ButterflyAgent class in your own code

## Questions?

The code is well-commented. Check:
- `butterfly_ollama.py` - Integration layer
- `butterfly_rsi_v4_5.py` - Core framework

Have fun exploring AI personality evolution! 🦋
