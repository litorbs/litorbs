from typing import List, Dict, Any, Optional
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
import numpy as np
from dataclasses import dataclass
from .memory import EpisodicMemory
from .cognitive import CognitiveArchitecture
from .environment import Environment
  
@dataclass
class AgentConfig:
    cognitive_architecture: str = "transformer"
    learning_rate: float = 0.001
    memory_size: int = 10000
    batch_size: int = 64
    hidden_size: int = 768
    num_attention_heads: int = 12
    num_hidden_layers: int = 12
    model_name: str = "gpt2"
 
class Agent:
    """
    NeuroPilot's core agent class that implements advanced cognitive capabilities
    and adaptive learning mechanisms.
    """
    
    def __init__(self, config: Optional[AgentConfig] = None):
        self.config = config or AgentConfig()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize components
        self.cognitive_architecture = CognitiveArchitecture(
            architecture_type=self.config.cognitive_architecture,
            hidden_size=self.config.hidden_size,
            num_attention_heads=self.config.num_attention_heads,
            num_hidden_layers=self.config.num_hidden_layers
        )
        
        self.memory = EpisodicMemory(
            capacity=self.config.memory_size,
            embedding_size=self.config.hidden_size
        )
        
        # Initialize language model and tokenizer
        self.model = AutoModel.from_pretrained(self.config.model_name).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(
            self.cognitive_architecture.parameters(),
            lr=self.config.learning_rate
        )

    def perceive(self, observation: Dict[str, Any]) -> torch.Tensor:
        """
        Process and encode environmental observations into internal representations.
        """
        if isinstance(observation, str):
            inputs = self.tokenizer(
                observation,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512 
            ).to(self.device)
            
            with torch.no_grad():
                encoded = self.model(**inputs).last_hidden_state.mean(dim=1)
        else:
            # Handle other types of observations (images, structured data, etc.)
            encoded = self.cognitive_architecture.encode(observation)
            
        return encoded

    def think(self, encoded_input: torch.Tensor) -> torch.Tensor:
        """
        Process encoded inputs through the cognitive architecture to generate
        thoughts and decisions.
        """
        # Retrieve relevant memories
        memories = self.memory.retrieve_relevant(encoded_input)
        
        # Combine current input with memories
        combined = torch.cat([encoded_input, memories], dim=-1)
        
        # Process through cognitive architecture
        thoughts = self.cognitive_architecture(combined)
        
        return thoughts

    def decide(self, thoughts: torch.Tensor) -> Dict[str, Any]:
        """
        Convert processed thoughts into actionable decisions.
        """
        # Generate action probabilities
        action_probs = self.cognitive_architecture.decision_layer(thoughts)
        
        # Sample action from probabilities
        action = torch.multinomial(action_probs, 1)
        
        return {
            "action": action.item(),
            "confidence": action_probs.max().item(),
            "thought_vector": thoughts.detach()
        }

    def learn(self, experience: Dict[str, Any]):
        """
        Update agent's knowledge based on experience.
        """
        # Store experience in memory
        self.memory.store(
            state=experience["state"],
            action=experience["action"],
            reward=experience["reward"],
            next_state=experience["next_state"]
        )
        
        # Perform learning update
        self.optimizer.zero_grad()
        loss = self.cognitive_architecture.compute_loss(experience)
        loss.backward()
        self.optimizer.step()
        
        return {"loss": loss.item()}

    def train(self, environment: Environment, episodes: int, goals: List[str]):
        """
        Train the agent in an environment.
        """
        for episode in range(episodes):
            state = environment.reset()
            done = False
            total_reward = 0
            
            while not done:
                # Perception
                encoded_state = self.perceive(state)
                
                # Thinking
                thoughts = self.think(encoded_state)
                
                # Decision making
                decision = self.decide(thoughts)
                
                # Take action
                next_state, reward, done, info = environment.step(decision["action"])
                
                # Learning
                experience = {
                    "state": state,
                    "action": decision["action"],
                    "reward": reward,
                    "next_state": next_state,
                    "done": done
                }
                
                learn_info = self.learn(experience)
                total_reward += reward
                state = next_state
            
            print(f"Episode {episode + 1}/{episodes}, Total Reward: {total_reward}")

    def save(self, path: str):
        """
        Save agent's state and parameters.
        """
        torch.save({
            'cognitive_architecture_state': self.cognitive_architecture.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'config': self.config,
            'memory': self.memory.state_dict()
        }, path)

    def load(self, path: str):
        """
        Load agent's state and parameters.
        """
        checkpoint = torch.load(path)
        self.cognitive_architecture.load_state_dict(checkpoint['cognitive_architecture_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.config = checkpoint['config']
        self.memory.load_state_dict(checkpoint['memory'])
