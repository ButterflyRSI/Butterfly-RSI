# 🦋 GETTING STARTED - Quick Reference

## What You Need

✅ **butterfly_ollama.py** - Integration wrapper connecting Ollama to your Butterfly framework

✅ **butterfly_rsi_v4_5.py** - Core Engine for Butterfly framework

✅ **README.md** - Complete documentation

✅ **quickstart.sh** - Automated setup script

✅ **examples.py** - Example usage scripts (WiFi security, creative writing, batch analysis)


## Setup Steps (5 minutes)

### Step 1: Copy Your v4.5 File
```bash
# Copy your butterfly_rsi_v4_5.py file to this directory
# It needs to be in the same folder as butterfly_ollama.py
cp /path/to/your/butterfly_rsi_v4_5.py .
```

### Step 2: Install Dependencies
```bash
pip install requests --break-system-packages
```

### Step 3: Verify Ollama
```bash
# Make sure Ollama is running (should already be from your install)
ollama list

# You should see llama3 in the list
# If not: ollama pull llama3
```

### Step 4: Run It!
```bash
# Option A: Use the quickstart script
./quickstart.sh

# Option B: Run directly
python3 butterfly_ollama.py
```

## Quick Test

```bash
python3 butterfly_ollama.py
# Select: 4 (Test Ollama Connection)
```

Should output:
```
✅ Ollama is running!
Response: Hello, Butterfly! [response from llama3]
```

## Your First Session

```bash
python3 butterfly_ollama.py
# Select: 2 (Interactive Session - Cybersecurity)
```

Then try:
```
EchoSec> What's the best way to secure a WiFi network?
[AI responds]
[AI reflects on its response]
Rate the response (0.0-1.0): 0.9

EchoSec> How does WPA3 improve on WPA2?
[continues...]

/personality   # See current traits
/report        # Full evolution report
/quit          # Save and exit
```

## Try the Examples

```bash
python3 examples.py
# Select: 1 (WiFi Security Assistant)
```

This runs a pre-scripted demo showing:
- 5 WiFi security questions
- Auto-reflection after each
- Personality evolution over time
- Final report showing trait changes

## What Happens During a Session

1. **You ask a question** → AI generates response using Ollama
2. **AI reflects** → Analyzes its own response quality
3. **You rate it** → Feedback score (0.0-1.0)
4. **Every 3 interactions** → Mirror loop runs (personality evolves)
5. **Every 6 interactions** → Dream consolidation (reinforces key memories)
6. **Session ends** → Full report + saved JSON file

## Understanding Personality Evolution

### Traits
- **Analytical** ↑ with good feedback (logical reasoning)
- **Creative** ↑ with good reflections (innovation)
- **Strategic** ↑ with good feedback (planning)
- **Curious** ↑ when uncertain (exploration)
- **Defensive** ↑ when uncertain (risk awareness)
- **Empathic** ↓ without reinforcement (understanding)

### States
- **Stable** - High performance (feedback > 0.85)
- **Adaptive** - Good performance (feedback > 0.70)
- **Volatile** - Low performance (feedback < 0.70)

## Customization Examples

### Create Your Own Persona
```python
from butterfly_ollama import ButterflyAgent

agent = ButterflyAgent(
    persona_name="NetSec",
    domain_focus=["network security", "firewalls", "IDS/IPS"],
    ethical_constraints=["lab only", "defensive security"],
    model="llama3"
)
```

### Adjust Evolution Speed
```python
agent.mirror_loop_interval = 5      # Every 5 interactions
agent.consolidation_interval = 10   # Every 10 interactions
```

### Custom Trait Config
```python
from butterfly_rsi_v4_5 import TraitConfig

config = TraitConfig(
    analytical_feedback_rate=0.15,    # Faster analytical growth
    creative_reflection_rate=0.20,    # More creative emphasis
)

agent = ButterflyAgent(..., config=config)
```

## Files Generated

After each session:
```
butterfly_session_EchoSec_20241109_203045.json
```

Contains:
- All interactions (prompts, responses, reflections)
- Personality trait evolution over time
- Stability metrics
- Dream consolidation insights

## Troubleshooting

**"Cannot import butterfly_rsi_v4_5"**
→ Copy your v4.5 file to this directory

**"Cannot connect to Ollama"**
→ Run `ollama serve` in another terminal

**Slow responses**
→ Normal for 8B model, or try `ollama pull llama3.2:3b` for faster (less capable) model

## Next Steps

1. ✅ Copy your butterfly_rsi_v4_5.py file here
2. ✅ Run `./quickstart.sh` to verify setup
3. ✅ Try interactive mode (#2 - Cybersecurity)
4. ✅ Run examples.py to see different personas
5. ✅ Build your own custom personas
6. ✅ Analyze saved session JSON files


Have fun! 🦋🔒
