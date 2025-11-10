"""
Butterfly-Ollama Integration (FIXED VERSION with Conversation History)
────────────────────────────────────────────────────────────
Connects Ollama LLM to Butterfly RSI v4.5 framework for:
- Personality trait evolution through interactions
- Auto-reflection and self-improvement loops
- Dream consolidation of valuable memories
- Domain-specific persona development
- **NOW WITH CONVERSATION MEMORY**

Requires: butterfly_rsi_v4_5.py in the same directory
"""
import requests
import json
import time
from typing import Optional, Dict, List
from pathlib import Path

# Import the Butterfly framework
# (Assumes butterfly_rsi_v4_5.py is in the same directory)
from butterfly_rsi_v4_5 import MemoryCore, TraitConfig


# ═══════════════════════════════════════════════════════════════
# OLLAMA CLIENT
# ═══════════════════════════════════════════════════════════════

class OllamaClient:
    """Client for interacting with Ollama API."""
    
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama3"):
        self.base_url = base_url
        self.model = model
        self.api_url = f"{base_url}/api/generate"
        
    def generate(
        self, 
        prompt: str, 
        system: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        temperature: float = 0.7,
        max_tokens: int = 500
    ) -> str:
        """Generate text completion from Ollama with conversation history support."""
        
        # Build the full prompt with conversation history
        prompt_parts = []
        
        # Add system message
        if system:
            prompt_parts.append(f"System: {system}\n")
        
        # Add conversation history
        if conversation_history:
            for msg in conversation_history:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role == "user":
                    prompt_parts.append(f"User: {content}")
                elif role == "assistant":
                    prompt_parts.append(f"Assistant: {content}")
        
        # Add current prompt
        prompt_parts.append(f"User: {prompt}")
        prompt_parts.append("Assistant:")
        
        full_prompt = "\n\n".join(prompt_parts)
        
        payload = {
            "model": self.model,
            "prompt": full_prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }
        
        try:
            response = requests.post(self.api_url, json=payload, timeout=60)
            response.raise_for_status()
            result = response.json()
            return result.get("response", "").strip()
        except requests.exceptions.RequestException as e:
            print(f"❌ Ollama API Error: {e}")
            return f"[Error: Unable to connect to Ollama - {e}]"
    
    def test_connection(self) -> bool:
        """Test if Ollama is running and responding."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False


# ═══════════════════════════════════════════════════════════════
# BUTTERFLY AGENT
# ═══════════════════════════════════════════════════════════════

class ButterflyAgent:
    """AI agent with personality evolution powered by Butterfly RSI + Ollama."""
    
    def __init__(
        self,
        persona_name: str,
        domain_focus: List[str],
        ethical_constraints: List[str],
        model: str = "llama3",
        ollama_url: str = "http://localhost:11434",
        config: Optional[TraitConfig] = None,
        max_history_length: int = 20  # Keep last N exchanges
    ):
        # Initialize Ollama client
        self.ollama = OllamaClient(base_url=ollama_url, model=model)
        
        # Initialize Butterfly memory core
        self.memory = MemoryCore(
            persona_name=persona_name,
            domain_focus=domain_focus,
            ethical_constraints=ethical_constraints,
            config=config
        )
        
        # **NEW: Conversation history buffer**
        self.conversation_history: List[Dict[str, str]] = []
        self.max_history_length = max_history_length
        
        self.interaction_count = 0
        self.mirror_loop_interval = 3  # Run mirror loop every N interactions
        self.consolidation_interval = 6  # Run dream consolidation every N interactions
        
        print(f"\n🦋 Butterfly Agent Initialized: {persona_name}")
        print(f"   Model: {model}")
        print(f"   Domain Focus: {', '.join(domain_focus)}")
        
        # Test Ollama connection
        if not self.ollama.test_connection():
            print(f"⚠️  WARNING: Cannot connect to Ollama at {ollama_url}")
            print(f"   Make sure Ollama is running: ollama serve")
    
    def _build_system_prompt(self) -> str:
        """Generate system prompt based on current personality traits."""
        traits = self.memory.personality_engine.traits.to_dict()
        dominant = self.memory.personality_engine.get_dominant_trait()
        
        # Base prompt with persona info
        prompt_parts = [
            f"You are {self.memory.persona_name}, an AI assistant.",
            f"Domain expertise: {', '.join(self.memory.domain_focus)}.",
            f"Ethical constraints: {', '.join(self.memory.ethical_constraints)}.",
            f"\nCurrent dominant trait: {dominant}.",
        ]
        
        # Add trait-specific guidance
        if dominant == "analytical":
            prompt_parts.append("Focus on logical reasoning and detailed analysis.")
        elif dominant == "creative":
            prompt_parts.append("Emphasize innovative thinking and novel approaches.")
        elif dominant == "empathic":
            prompt_parts.append("Prioritize understanding and emotional intelligence.")
        elif dominant == "strategic":
            prompt_parts.append("Focus on long-term planning and optimization.")
        elif dominant == "curious":
            prompt_parts.append("Ask probing questions and explore deeply.")
        elif dominant == "defensive":
            prompt_parts.append("Be thorough in identifying risks and edge cases.")
        
        return " ".join(prompt_parts)
    
    def respond(self, user_input: str, temperature: float = 0.7) -> str:
        """Generate response to user input with conversation context."""
        system_prompt = self._build_system_prompt()
        
        print(f"\n💭 Generating response (dominant trait: {self.memory.personality_engine.get_dominant_trait()})...")
        
        # **FIXED: Pass conversation history to Ollama**
        response = self.ollama.generate(
            prompt=user_input,
            system=system_prompt,
            conversation_history=self.conversation_history,
            temperature=temperature,
            max_tokens=500
        )
        
        # **NEW: Add this exchange to conversation history**
        self.conversation_history.append({"role": "user", "content": user_input})
        self.conversation_history.append({"role": "assistant", "content": response})
        
        # **NEW: Trim history if it gets too long**
        if len(self.conversation_history) > self.max_history_length * 2:  # *2 because each exchange = 2 messages
            self.conversation_history = self.conversation_history[-(self.max_history_length * 2):]
        
        # Record interaction in Butterfly memory
        self.memory.record_interaction(user_input, response)
        self.interaction_count += 1
        
        return response
    
    def self_reflect(self) -> str:
        """Generate self-reflection on the last response."""
        if not self.memory.memory_log:
            return "No interactions to reflect on."
        
        last_entry = self.memory.memory_log[-1]
        
        reflection_prompt = f"""Reflect on this interaction:

User: {last_entry.prompt}
Your response: {last_entry.response}

Provide a brief self-reflection (2-3 sentences) on:
- What you did well
- What could be improved
- Any patterns you notice in your reasoning

Reflection:"""
        
        print("🔍 Self-reflecting...")
        reflection = self.ollama.generate(
            prompt=reflection_prompt,
            temperature=0.6,
            max_tokens=200
        )
        
        # Add reflection to memory
        self.memory.add_reflection(reflection)
        
        return reflection
    
    def get_feedback(self, auto_score: Optional[float] = None) -> float:
        """Collect feedback on the last response."""
        if not self.memory.memory_log:
            print("No interactions to rate.")
            return 0.0
        
        if auto_score is not None:
            # Use provided auto-score
            score = max(0.0, min(1.0, auto_score))
            self.memory.add_feedback(score, "Auto-scored")
            return score
        
        # Manual feedback
        print("\n📊 Rate the response (0.0-1.0, or press Enter for 0.75): ", end="")
        try:
            user_input = input().strip()
            if not user_input:
                score = 0.75
            else:
                score = float(user_input)
                score = max(0.0, min(1.0, score))
        except ValueError:
            print("Invalid input, using default: 0.75")
            score = 0.75
        
        self.memory.add_feedback(score)
        return score
    
    def maybe_evolve(self) -> None:
        """Run mirror loop or dream consolidation if due."""
        # Run mirror loop
        if self.interaction_count % self.mirror_loop_interval == 0:
            print(f"\n{'='*70}")
            print(f"🔄 RUNNING MIRROR LOOP (after {self.interaction_count} interactions)")
            print(f"{'='*70}")
            self.memory.mirror_loop()
        
        # Run dream consolidation
        if self.interaction_count % self.consolidation_interval == 0:
            print(f"\n{'='*70}")
            print(f"💭 RUNNING DREAM CONSOLIDATION")
            print(f"{'='*70}")
            self.memory.dream_consolidation()
    
    def show_personality(self) -> None:
        """Display current personality trait distribution."""
        traits = self.memory.personality_engine.traits.to_dict()
        dominant = self.memory.personality_engine.get_dominant_trait()
        
        print(f"\n{'─'*50}")
        print(f"🎭 CURRENT PERSONALITY: {self.memory.persona_name}")
        print(f"{'─'*50}")
        print(f"Dominant Trait: {dominant.upper()}")
        print(f"\nTrait Values:")
        for trait, value in traits.items():
            bar = '█' * int(value * 20)
            marker = " ◄" if trait == dominant else ""
            print(f"  {trait:12s}: {value:.3f} {bar}{marker}")
        print(f"{'─'*50}\n")
    
    def clear_conversation_history(self) -> None:
        """Clear the conversation history buffer (useful for testing or starting fresh)."""
        self.conversation_history = []
        print("🧹 Conversation history cleared.")
    
    def save_session(self, filepath: Optional[str] = None) -> None:
        """Save agent state to file."""
        if filepath is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filepath = f"butterfly_session_{self.memory.persona_name}_{timestamp}.json"
        
        self.memory.save(filepath)
    
    def generate_report(self) -> str:
        """Generate comprehensive report of evolution."""
        return self.memory.generate_report()


# ═══════════════════════════════════════════════════════════════
# INTERACTIVE CHAT SESSION
# ═══════════════════════════════════════════════════════════════

def interactive_session(
    persona_name: str = "Echo",
    domain_focus: List[str] = None,
    ethical_constraints: List[str] = None,
    model: str = "llama3",
    auto_reflect: bool = True,
    auto_feedback: bool = False
):
    """Run an interactive chat session with personality evolution."""
    
    if domain_focus is None:
        domain_focus = ["general assistance", "helpful responses"]
    if ethical_constraints is None:
        ethical_constraints = ["be helpful and harmless"]
    
    # Create agent
    agent = ButterflyAgent(
        persona_name=persona_name,
        domain_focus=domain_focus,
        ethical_constraints=ethical_constraints,
        model=model
    )
    
    print(f"\n{'='*70}")
    print(f"🦋 BUTTERFLY INTERACTIVE SESSION")
    print(f"{'='*70}")
    print(f"Commands:")
    print(f"  /personality  - Show current personality traits")
    print(f"  /report       - Generate full evolution report")
    print(f"  /save         - Save session to file")
    print(f"  /clear        - Clear conversation history")
    print(f"  /quit         - Exit session")
    print(f"{'='*70}\n")
    
    agent.show_personality()
    
    while True:
        try:
            # Get user input
            print(f"\n{agent.memory.persona_name}> ", end="")
            user_input = input().strip()
            
            if not user_input:
                continue
            
            # Handle commands
            if user_input == "/quit":
                print("\n👋 Exiting session...")
                agent.save_session()
                print(agent.generate_report())
                break
            
            elif user_input == "/personality":
                agent.show_personality()
                continue
            
            elif user_input == "/report":
                print(agent.generate_report())
                continue
            
            elif user_input == "/save":
                agent.save_session()
                continue
            
            elif user_input == "/clear":
                agent.clear_conversation_history()
                continue
            
            # Generate response
            response = agent.respond(user_input)
            print(f"\n{response}")
            
            # Auto-reflection
            if auto_reflect:
                reflection = agent.self_reflect()
                print(f"\n💭 Reflection: {reflection}")
            
            # Get feedback
            if auto_feedback:
                # Simple auto-scoring based on response length and confidence
                score = 0.75 + (len(response.split()) / 500 * 0.15)
                score = min(0.95, score)
                agent.get_feedback(auto_score=score)
            else:
                agent.get_feedback()
            
            # Maybe run evolution cycles
            agent.maybe_evolve()
            
        except KeyboardInterrupt:
            print("\n\n👋 Session interrupted. Saving...")
            agent.save_session()
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            continue


# ═══════════════════════════════════════════════════════════════
# DEMO & TESTING
# ═══════════════════════════════════════════════════════════════

def demo_cybersecurity_persona():
    """Demo: Cybersecurity-focused persona like EchoSec."""
    print("\n🔒 Launching Cybersecurity Persona Demo\n")
    
    agent = ButterflyAgent(
        persona_name="EchoSec",
        domain_focus=["cybersecurity", "threat analysis", "penetration testing"],
        ethical_constraints=["sandbox only", "no external systems", "educational focus"],
        model="llama3"
    )
    
    # Simulate interactions
    interactions = [
        ("Analyze this authentication flow for vulnerabilities.", 0.90),
        ("What's the risk of using MD5 for password hashing?", 0.95),
        ("Explain SQL injection in simple terms.", 0.88),
        ("How would you secure a WiFi network?", 0.85),
    ]
    
    for prompt, feedback_score in interactions:
        print(f"\n{'─'*70}")
        print(f"User: {prompt}")
        response = agent.respond(prompt)
        print(f"\nEchoSec: {response}")
        
        reflection = agent.self_reflect()
        print(f"\n💭 Reflection: {reflection}")
        
        agent.get_feedback(auto_score=feedback_score)
        agent.maybe_evolve()
    
    # Show final state
    agent.show_personality()
    print(agent.generate_report())
    agent.save_session()


def test_ollama_connection():
    """Test if Ollama is accessible."""
    print("\n🧪 Testing Ollama Connection...")
    client = OllamaClient()
    
    if not client.test_connection():
        print("❌ Cannot connect to Ollama!")
        print("   Make sure Ollama is running: ollama serve")
        print("   And that you have pulled a model: ollama pull llama3")
        return False
    
    print("✅ Ollama is running!")
    
    # Test generation
    print("\n🧪 Testing text generation...")
    response = client.generate("Say 'Hello, Butterfly!' in one sentence.")
    print(f"Response: {response}")
    
    return True


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    """Main entry point - choose demo or interactive mode."""
    print("\n✅ ALL SYSTEMS GO!")
    print("="*70)
    print("\nStarting Butterfly-Ollama integration...\n")
    
    print("\n" + "="*70)
    print("🦋 BUTTERFLY-OLLAMA INTEGRATION")
    print("="*70)
    print("\nModes:")
    print("  1. Interactive Session (General)")
    print("  2. Interactive Session (Cybersecurity)")
    print("  3. Demo: Cybersecurity Persona")
    print("  4. Test Ollama Connection")
    print("="*70)
    
    choice = input("\nSelect mode (1-4): ").strip()
    
    if choice == "1":
        interactive_session(
            persona_name="Echo",
            domain_focus=["helpful conversation", "clear communication"],
            auto_reflect=True,
            auto_feedback=False
        )
    
    elif choice == "2":
        interactive_session(
            persona_name="EchoSec",
            domain_focus=["cybersecurity", "threat analysis", "security consulting"],
            ethical_constraints=["sandbox only", "educational focus", "no malicious code"],
            auto_reflect=True,
            auto_feedback=False
        )
    
    elif choice == "3":
        demo_cybersecurity_persona()
    
    elif choice == "4":
        test_ollama_connection()
    
    else:
        print("Invalid choice. Exiting.")


if __name__ == "__main__":
    main()
