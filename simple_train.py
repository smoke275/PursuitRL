#!/usr/bin/env python3
"""
Simple Continuous Training - Both agents train together with progress tracking
"""

import numpy as np
import time
import torch
import json
import os
from datetime import datetime
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

class SingleAgentWrapper(gym.Env):
    """Simple Gymnasium wrapper for single agent training"""
    
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
            # Random action if no model
            actions[self.other_agent_id] = self.base_env.action_space(self.other_agent_id).sample()
        
        obs, rewards, terms, truncs, infos = self.base_env.step(actions)
        self.last_obs = obs
        
        # Render occasionally
        self.step_count += 1
        if self._render_mode == "human" and self.step_count % self.render_frequency == 0:
            self.base_env.render()
            if self.render_frequency == 5:
                time.sleep(0.01)
        
        reward = rewards.get(self.agent_id, 0.0)
        terminated = terms.get(self.agent_id, False)
        truncated = truncs.get(self.agent_id, False)
        info = infos.get(self.agent_id, {})
        observation = obs.get(self.agent_id, np.zeros(108, dtype=np.float32))
        
        return observation, reward, terminated, truncated, info
    
    def set_other_agent_model(self, model):
        self.other_agent_model = model
    
    def render(self):
        return self.base_env.render()
    
    def close(self):
        self.base_env.close()

def load_training_progress():
    """Load training progress from file"""
    progress_file = "training_progress.json"
    if os.path.exists(progress_file):
        with open(progress_file, 'r') as f:
            return json.load(f)
    else:
        return {
            "total_steps": 0,
            "training_sessions": 0,
            "last_trained": None,
            "visitor_total_steps": 0,
            "escort_total_steps": 0,
            "history": []
        }

def save_training_progress(progress):
    """Save training progress to file"""
    with open("training_progress.json", 'w') as f:
        json.dump(progress, f, indent=2)

def load_or_create_models(device):
    """Load existing models or create new ones"""
    try:
        from stable_baselines3 import PPO
        
        # Try to load existing models
        if os.path.exists("visitor_model.zip") and os.path.exists("escort_model.zip"):
            print("📂 Loading existing models...")
            visitor_model = PPO.load("visitor_model", device=device)
            escort_model = PPO.load("escort_model", device=device)
            print("✅ Existing models loaded successfully!")
            return visitor_model, escort_model, True
        else:
            print("🆕 Creating new models...")
            # Create new environments for training
            visitor_env = SingleAgentWrapper("visitor")
            escort_env = SingleAgentWrapper("escort")
            
            # Create new models
            visitor_model = PPO(
                "MlpPolicy", 
                visitor_env, 
                verbose=1,
                learning_rate=3e-4,
                n_steps=1024,
                batch_size=64,
                n_epochs=10,
                device=device,
            )
            
            escort_model = PPO(
                "MlpPolicy", 
                escort_env, 
                verbose=1,
                learning_rate=3e-4,
                n_steps=1024,
                batch_size=64,
                n_epochs=10,
                device=device,
            )
            
            visitor_env.close()
            escort_env.close()
            print("✅ New models created!")
            return visitor_model, escort_model, False
            
    except Exception as e:
        print(f"❌ Error loading/creating models: {e}")
        return None, None, False

def train_session(steps=5000, with_visualization=False):
    """Run a training session"""
    try:
        from stable_baselines3 import PPO
        
        device = detect_device()
        progress = load_training_progress()
        
        print(f"\n🎯 Training Session #{progress['training_sessions'] + 1}")
        print(f"📊 Previous total steps: {progress['total_steps']}")
        print(f"🎮 This session: {steps} steps")
        print("=" * 50)
        
        # Load or create models
        visitor_model, escort_model, models_existed = load_or_create_models(device)
        if visitor_model is None:
            return
        
        # Create training environments
        render_mode = "human" if with_visualization else None
        visitor_env = SingleAgentWrapper("visitor", render_mode=render_mode)
        escort_env = SingleAgentWrapper("escort", render_mode=render_mode)
        
        # Update the models' environments
        visitor_model.set_env(visitor_env)
        escort_model.set_env(escort_env)
        
        # Set up adversarial training - each agent trains against the other
        visitor_env.set_other_agent_model(escort_model)
        escort_env.set_other_agent_model(visitor_model)
        
        print(f"{'🎮 ' if with_visualization else '⚡ '}Training both agents simultaneously...")
        if with_visualization:
            print("Watch the agents compete in real-time!")
        
        start_time = time.time()
        
        # Train both agents simultaneously (they're playing against each other)
        # We alternate who gets trained each mini-batch to keep it fair
        steps_per_agent = steps // 2
        
        print(f"🔴 Training Visitor Agent ({steps_per_agent} steps)...")
        visitor_model.learn(total_timesteps=steps_per_agent, reset_num_timesteps=False)
        
        print(f"🟠 Training Escort Agent ({steps_per_agent} steps)...")
        escort_model.learn(total_timesteps=steps_per_agent, reset_num_timesteps=False)
        
        training_time = time.time() - start_time
        
        # Save models
        print("💾 Saving models...")
        visitor_model.save("visitor_model")
        escort_model.save("escort_model")
        
        # Update progress
        progress["total_steps"] += steps
        progress["visitor_total_steps"] += steps_per_agent
        progress["escort_total_steps"] += steps_per_agent
        progress["training_sessions"] += 1
        progress["last_trained"] = datetime.now().isoformat()
        progress["history"].append({
            "session": progress["training_sessions"],
            "steps": steps,
            "training_time": training_time,
            "date": datetime.now().isoformat(),
            "with_visualization": with_visualization
        })
        
        save_training_progress(progress)
        
        visitor_env.close()
        escort_env.close()
        
        print("\n🎉 Training Session Complete!")
        print(f"⏱️  Training time: {training_time:.1f} seconds")
        print(f"📈 Total steps trained: {progress['total_steps']}")
        print(f"🔢 Total sessions: {progress['training_sessions']}")
        print("💾 Models and progress saved!")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")

def show_progress():
    """Show training progress"""
    progress = load_training_progress()
    
    print("\n📊 Training Progress")
    print("=" * 40)
    print(f"Total Steps Trained: {progress['total_steps']:,}")
    print(f"Training Sessions: {progress['training_sessions']}")
    print(f"Visitor Steps: {progress['visitor_total_steps']:,}")
    print(f"Escort Steps: {progress['escort_total_steps']:,}")
    
    if progress['last_trained']:
        last_date = datetime.fromisoformat(progress['last_trained'])
        print(f"Last Trained: {last_date.strftime('%Y-%m-%d %H:%M:%S')}")
    
    if progress['history']:
        print(f"\nRecent Sessions:")
        for session in progress['history'][-5:]:  # Show last 5 sessions
            date = datetime.fromisoformat(session['date'])
            viz = "🎮" if session.get('with_visualization', False) else "⚡"
            print(f"  {viz} Session {session['session']}: {session['steps']} steps ({session['training_time']:.1f}s) - {date.strftime('%m/%d %H:%M')}")

def test_models():
    """Test the current models"""
    try:
        from stable_baselines3 import PPO
        
        if not (os.path.exists("visitor_model.zip") and os.path.exists("escort_model.zip")):
            print("❌ No trained models found. Train first!")
            return
        
        print("🎮 Testing Current Models...")
        
        visitor_model = PPO.load("visitor_model")
        escort_model = PPO.load("escort_model")
        
        test_env = env(render_mode="human")
        
        for episode in range(3):
            print(f"Episode {episode + 1}/3")
            observations, _ = test_env.reset()
            episode_rewards = {"visitor": 0, "escort": 0}
            steps = 0
            
            while test_env.agents and steps < 1000:
                actions = {}
                
                if "visitor" in test_env.agents:
                    visitor_obs = observations["visitor"]
                    visitor_action, _ = visitor_model.predict(visitor_obs, deterministic=True)
                    actions["visitor"] = visitor_action
                
                if "escort" in test_env.agents:
                    escort_obs = observations["escort"]
                    escort_action, _ = escort_model.predict(escort_obs, deterministic=True)
                    actions["escort"] = escort_action
                
                observations, rewards, terminations, truncations, infos = test_env.step(actions)
                
                for agent in rewards:
                    episode_rewards[agent] += rewards[agent]
                
                steps += 1
                test_env.render()
                time.sleep(0.05)
            
            print(f"   Steps: {steps}, Visitor: {episode_rewards['visitor']:.2f}, Escort: {episode_rewards['escort']:.2f}")
        
        test_env.close()
        
    except Exception as e:
        print(f"❌ Testing failed: {e}")

def main():
    """Main menu"""
    print("🏁 Simple Continuous Multi-Agent Training")
    print("=" * 50)
    
    while True:
        print("\nSelect option:")
        print("1. Train 5000 steps (Fast)")
        print("2. Train 5000 steps (With Visualization)")
        print("3. Train custom steps")
        print("4. Show Training Progress")
        print("5. Test Current Models")
        print("6. Exit")
        
        choice = input("\nEnter choice (1-6): ").strip()
        
        if choice == "1":
            train_session(5000, with_visualization=False)
        elif choice == "2":
            train_session(5000, with_visualization=True)
        elif choice == "3":
            try:
                steps = int(input("Enter number of steps: "))
                viz = input("With visualization? (y/n): ").lower().startswith('y')
                train_session(steps, with_visualization=viz)
            except ValueError:
                print("❌ Invalid number of steps")
        elif choice == "4":
            show_progress()
        elif choice == "5":
            test_models()
        elif choice == "6":
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice. Please try again.")

if __name__ == "__main__":
    main()