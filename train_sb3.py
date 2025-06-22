#!/usr/bin/env python3
"""
Stable-Baselines3 Training for Pursuit/Evasion Environment

Clean, focused training script for multi-agent RL with SB3.
Uses proper Gymnasium inheritance for SB3 compatibility.
Supports CPU, MPS (Apple Silicon), and CUDA devices.
"""

import numpy as np
import time
import torch
import gymnasium as gym
from pettingzoo_env import env

def detect_device():
    """Detect the best available device for training"""
    if torch.cuda.is_available():
        device = "cuda"
        print(f"🚀 Using CUDA device: {torch.cuda.get_device_name()}")
    elif torch.backends.mps.is_available():
        device = "mps"
        print("🍎 Using Apple Silicon MPS device")
    else:
        device = "cpu"
        print("💻 Using CPU device")
    
    return device

def test_environment():
    """Quick test to make sure environment works"""
    print("🧪 Testing environment...")
    test_env = env()
    observations, infos = test_env.reset()
    
    # Test one step
    actions = {agent: test_env.action_space(agent).sample() for agent in test_env.agents}
    observations, rewards, terminations, truncations, infos = test_env.step(actions)
    
    print(f"✅ Environment working!")
    print(f"   Agents: {test_env.agents}")
    print(f"   Observation shape: {observations['visitor'].shape}")
    print(f"   Action space: {test_env.action_space('visitor')}")
    
    test_env.close()

class SingleAgentWrapper(gym.Env):
    """Proper Gymnasium environment wrapper for single agent training"""
    
    def __init__(self, agent_id, render_mode=None):
        super().__init__()
        self.agent_id = agent_id
        self.other_agent_id = "escort" if agent_id == "visitor" else "visitor"
        self.base_env = env(render_mode=render_mode)
        self.other_agent_model = None
        self.last_obs = None
        self.step_count = 0
        self.render_frequency = 10 if render_mode is None else 5
        
        # Set action and observation spaces
        self.action_space = self.base_env.action_space(self.agent_id)
        self.observation_space = self.base_env.observation_space(self.agent_id)
        
        # Store render mode as internal attribute
        self._render_mode = render_mode
    
    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        obs, infos = self.base_env.reset()
        self.last_obs = obs
        self.step_count = 0
        return obs[self.agent_id], infos.get(self.agent_id, {})
    
    def step(self, action):
        # Create actions for both agents
        actions = {self.agent_id: action}
        
        # Get action from other agent
        if self.other_agent_model is not None and self.last_obs is not None:
            other_obs = self.last_obs[self.other_agent_id]
            other_action, _ = self.other_agent_model.predict(other_obs, deterministic=False)
            actions[self.other_agent_id] = other_action
        else:
            # Random action if no model or no previous observation
            actions[self.other_agent_id] = self.base_env.action_space(self.other_agent_id).sample()
        
        obs, rewards, terms, truncs, infos = self.base_env.step(actions)
        self.last_obs = obs
        
        # Render occasionally (check if we have render mode set during init)
        self.step_count += 1
        if hasattr(self, '_render_mode') and self._render_mode == "human" and self.step_count % self.render_frequency == 0:
            self.base_env.render()
            if self.render_frequency == 5:  # Visualization mode
                time.sleep(0.01)
        
        reward = rewards.get(self.agent_id, 0.0)
        terminated = terms.get(self.agent_id, False)
        truncated = truncs.get(self.agent_id, False)
        info = infos.get(self.agent_id, {})
        observation = obs.get(self.agent_id, np.zeros(108, dtype=np.float32))
        
        return observation, reward, terminated, truncated, info
    
    def set_other_agent_model(self, model):
        """Set the model for the other agent"""
        self.other_agent_model = model
    
    def render(self):
        return self.base_env.render()
    
    def close(self):
        self.base_env.close()

def train_both_agents_together():
    """Train both agents simultaneously with proper Gymnasium environments"""
    try:
        from stable_baselines3 import PPO
        
        # Detect device for training
        device = detect_device()
        
        print("\n🤖 Training Both Agents Together (True Multi-Agent)...")
        print("Both agents will learn to counter each other's strategies!")
        
        # Create environments for both agents
        visitor_env = SingleAgentWrapper("visitor")
        escort_env = SingleAgentWrapper("escort")
        
        # Initialize models with device support
        visitor_model = PPO(
            "MlpPolicy", 
            visitor_env, 
            verbose=1,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            device=device,
        )
        
        escort_model = PPO(
            "MlpPolicy", 
            escort_env, 
            verbose=1,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            device=device,
        )
        
        # Alternating training - each agent learns against the other's current policy
        total_rounds = 5
        steps_per_round = 20000
        
        print(f"Starting alternating training: {total_rounds} rounds of {steps_per_round} steps each")
        print(f"Device: {device}")
        
        for round_num in range(total_rounds):
            print(f"\n🔄 Training Round {round_num + 1}/{total_rounds}")
            
            # Set each agent to play against the other's current policy
            visitor_env.set_other_agent_model(escort_model)
            escort_env.set_other_agent_model(visitor_model)
            
            # Train visitor for this round
            print(f"   🔴 Training Visitor Agent...")
            visitor_model.learn(total_timesteps=steps_per_round, reset_num_timesteps=False)
            
            # Train escort for this round  
            print(f"   🟠 Training Escort Agent...")
            escort_model.learn(total_timesteps=steps_per_round, reset_num_timesteps=False)
            
            # Save intermediate models
            visitor_model.save(f"visitor_model_round_{round_num}")
            escort_model.save(f"escort_model_round_{round_num}")
            
            print(f"   ✅ Round {round_num + 1} complete - models saved")
        
        # Save final models
        visitor_model.save("visitor_model_final")
        escort_model.save("escort_model_final")
        
        visitor_env.close()
        escort_env.close()
        
        print("\n🎉 Multi-agent training complete!")
        print("Both agents have learned to counter each other's strategies")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   Make sure stable-baselines3 is installed")

def train_both_agents_with_visualization():
    """Train both agents simultaneously with visualization"""
    try:
        from stable_baselines3 import PPO
        
        # Detect device for training
        device = detect_device()
        
        print("\n🎮 Training Both Agents Together (WITH VISUALIZATION)...")
        print("You'll see the training in real-time - this will be slower but more interesting!")
        print("Both agents will learn to counter each other's strategies!")
        
        # Create environments for both agents WITH VISUALIZATION
        visitor_env = SingleAgentWrapper("visitor", render_mode="human")
        escort_env = SingleAgentWrapper("escort", render_mode="human")
        
        # Initialize models with smaller batches for faster visualization and device support
        visitor_model = PPO(
            "MlpPolicy", 
            visitor_env, 
            verbose=1,
            learning_rate=3e-4,
            n_steps=512,  # Smaller for faster updates
            batch_size=32,
            n_epochs=5,
            device=device,
        )
        
        escort_model = PPO(
            "MlpPolicy", 
            escort_env, 
            verbose=1,
            learning_rate=3e-4,
            n_steps=512,  # Smaller for faster updates
            batch_size=32,
            n_epochs=5,
            device=device,
        )
        
        # Shorter training for visualization
        total_rounds = 3
        steps_per_round = 5000
        
        print(f"Starting visual training: {total_rounds} rounds of {steps_per_round} steps each")
        print(f"Device: {device}")
        print("Watch as the agents learn different strategies!")
        
        for round_num in range(total_rounds):
            print(f"\n🔄 Training Round {round_num + 1}/{total_rounds}")
            
            # Set each agent to play against the other's current policy
            visitor_env.set_other_agent_model(escort_model)
            escort_env.set_other_agent_model(visitor_model)
            
            # Train visitor for this round
            print(f"   🔴 Training Visitor Agent (watch the red agent learn to escape)...")
            visitor_model.learn(total_timesteps=steps_per_round, reset_num_timesteps=False)
            
            # Train escort for this round  
            print(f"   🟠 Training Escort Agent (watch the orange agent learn to pursue)...")
            escort_model.learn(total_timesteps=steps_per_round, reset_num_timesteps=False)
            
            # Save intermediate models
            visitor_model.save(f"visitor_model_visual_round_{round_num}")
            escort_model.save(f"escort_model_visual_round_{round_num}")
            
            print(f"   ✅ Round {round_num + 1} complete - models saved")
        
        # Save final models
        visitor_model.save("visitor_model_visual_final")
        escort_model.save("escort_model_visual_final")
        
        visitor_env.close()
        escort_env.close()
        
        print("\n🎉 Visual multi-agent training complete!")
        print("You should have seen the agents' strategies evolve over time!")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   Make sure stable-baselines3 is installed")

def test_trained_models():
    """Test the trained models against each other"""
    try:
        from stable_baselines3 import PPO
        
        print("\n🎮 Testing Trained Models...")
        
        # Try to load final models first, then fallback to regular models
        try:
            visitor_model = PPO.load("visitor_model_final")
            escort_model = PPO.load("escort_model_final")
            print("Loaded final models from multi-agent training")
        except FileNotFoundError:
            try:
                visitor_model = PPO.load("visitor_model_visual_final")
                escort_model = PPO.load("escort_model_visual_final")
                print("Loaded visual training models")
            except FileNotFoundError:
                print("❌ No trained models found. Run training first!")
                return
        
        # Test environment
        test_env = env(render_mode="human")
        
        for episode in range(5):
            print(f"Episode {episode + 1}/5")
            observations, _ = test_env.reset()
            episode_rewards = {"visitor": 0, "escort": 0}
            steps = 0
            
            while test_env.agents:
                actions = {}
                
                # Get actions from trained models
                if "visitor" in test_env.agents:
                    visitor_obs = observations["visitor"]
                    visitor_action, _ = visitor_model.predict(visitor_obs, deterministic=True)
                    actions["visitor"] = visitor_action
                
                if "escort" in test_env.agents:
                    escort_obs = observations["escort"]
                    escort_action, _ = escort_model.predict(escort_obs, deterministic=True)
                    actions["escort"] = escort_action
                
                observations, rewards, terminations, truncations, infos = test_env.step(actions)
                
                # Accumulate rewards
                for agent in rewards:
                    episode_rewards[agent] += rewards[agent]
                
                steps += 1
                test_env.render()
                time.sleep(0.05)  # Slow down for viewing
                
                if steps > 1000:  # Prevent infinite episodes
                    break
            
            print(f"   Steps: {steps}, Visitor: {episode_rewards['visitor']:.2f}, Escort: {episode_rewards['escort']:.2f}")
        
        test_env.close()
        
    except ImportError:
        print("❌ Stable-Baselines3 not installed.")

def main():
    """Main training menu"""
    print("🏁 Stable-Baselines3 Multi-Agent Pursuit/Evasion Training")
    print("=" * 60)
    
    while True:
        print("\nSelect option:")
        print("1. Test Environment")
        print("2. Train Both Agents Together (Fast)")
        print("3. Train Both Agents Together (WITH VISUALIZATION)")
        print("4. Test Trained Models")
        print("5. Exit")
        
        choice = input("\nEnter choice (1-5): ").strip()
        
        if choice == "1":
            test_environment()
        elif choice == "2":
            train_both_agents_together()
        elif choice == "3":
            train_both_agents_with_visualization()
        elif choice == "4":
            test_trained_models()
        elif choice == "5":
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice. Please try again.")

if __name__ == "__main__":
    main()