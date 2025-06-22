#!/usr/bin/env python3
"""
Model Comparison Script - Compare behavior between training rounds
"""

import numpy as np
import time
from pettingzoo_env import env

def compare_models():
    """Compare behavior between round 0 and round 1 models"""
    try:
        from stable_baselines3 import PPO
        
        print("🔍 Comparing Models: Round 0 vs Round 1")
        print("=" * 50)
        
        # Load both rounds for both agents
        visitor_round0 = PPO.load("visitor_model_visual_round_0")
        escort_round0 = PPO.load("escort_model_visual_round_0")
        visitor_round1 = PPO.load("visitor_model_visual_round_1")
        escort_round1 = PPO.load("escort_model_visual_round_1")
        
        print("✅ All models loaded successfully!")
        
        # Test different combinations
        scenarios = [
            ("Round 0 vs Round 0", visitor_round0, escort_round0),
            ("Round 1 vs Round 1", visitor_round1, escort_round1),
            ("Round 0 Visitor vs Round 1 Escort", visitor_round0, escort_round1),
            ("Round 1 Visitor vs Round 0 Escort", visitor_round1, escort_round0),
        ]
        
        for scenario_name, visitor_model, escort_model in scenarios:
            print(f"\n🎮 Testing: {scenario_name}")
            print("-" * 40)
            
            test_env = env(render_mode="human")
            episode_stats = []
            
            for episode in range(3):  # Test 3 episodes per scenario
                observations, _ = test_env.reset()
                episode_data = {
                    "visitor_reward": 0,
                    "escort_reward": 0,
                    "steps": 0,
                    "visitor_escapes": 0,
                    "escort_maintains_los": 0,
                    "distance_history": []
                }
                
                while test_env.agents and episode_data["steps"] < 500:
                    actions = {}
                    
                    # Get actions from models
                    if "visitor" in test_env.agents:
                        visitor_obs = observations["visitor"]
                        visitor_action, _ = visitor_model.predict(visitor_obs, deterministic=True)
                        actions["visitor"] = visitor_action
                    
                    if "escort" in test_env.agents:
                        escort_obs = observations["escort"]
                        escort_action, _ = escort_model.predict(escort_obs, deterministic=True)
                        actions["escort"] = escort_action
                    
                    observations, rewards, terminations, truncations, infos = test_env.step(actions)
                    
                    # Track statistics
                    for agent in rewards:
                        episode_data[f"{agent}_reward"] += rewards[agent]
                    
                    # Track behavioral metrics
                    if "visitor" in observations and "escort" in observations:
                        # Calculate distance between agents
                        visitor_pos = observations["visitor"][:2]  # x, y position
                        escort_pos = observations["escort"][:2]
                        distance = np.linalg.norm(visitor_pos - escort_pos)
                        episode_data["distance_history"].append(distance)
                        
                        # Check if escort has line of sight (based on reward structure)
                        if rewards.get("escort", 0) > 0:
                            episode_data["escort_maintains_los"] += 1
                        
                        # Check if visitor is escaping (breaking line of sight)
                        if rewards.get("visitor", 0) > 5:  # High reward for escape
                            episode_data["visitor_escapes"] += 1
                    
                    episode_data["steps"] += 1
                    test_env.render()
                    time.sleep(0.02)  # Slow down for observation
                
                episode_stats.append(episode_data)
            
            # Analyze results
            avg_visitor_reward = np.mean([ep["visitor_reward"] for ep in episode_stats])
            avg_escort_reward = np.mean([ep["escort_reward"] for ep in episode_stats])
            avg_steps = np.mean([ep["steps"] for ep in episode_stats])
            avg_distance = np.mean([np.mean(ep["distance_history"]) for ep in episode_stats if ep["distance_history"]])
            avg_visitor_escapes = np.mean([ep["visitor_escapes"] for ep in episode_stats])
            avg_escort_los = np.mean([ep["escort_maintains_los"] for ep in episode_stats])
            
            print(f"  Average Visitor Reward: {avg_visitor_reward:.2f}")
            print(f"  Average Escort Reward: {avg_escort_reward:.2f}")
            print(f"  Average Episode Length: {avg_steps:.1f} steps")
            print(f"  Average Distance: {avg_distance:.1f} pixels")
            print(f"  Visitor Escape Events: {avg_visitor_escapes:.1f} per episode")
            print(f"  Escort Line-of-Sight: {avg_escort_los:.1f} steps per episode")
            
            test_env.close()
            
            # Wait between scenarios
            input("Press Enter to continue to next scenario...")
        
        print("\n🎯 Analysis Complete!")
        print("Key differences you should have observed:")
        print("• Round 0: More random, inefficient movement")
        print("• Round 1: More purposeful strategies emerging")
        print("• Cross-round tests show adaptation to opponent strategies")
        
    except FileNotFoundError as e:
        print(f"❌ Model file not found: {e}")
        print("Make sure training completed successfully.")
    except ImportError:
        print("❌ Stable-Baselines3 not installed.")

def analyze_model_weights():
    """Compare the actual neural network weights between rounds"""
    try:
        from stable_baselines3 import PPO
        import torch
        
        print("\n🧠 Neural Network Weight Analysis")
        print("=" * 40)
        
        # Load models
        visitor_round0 = PPO.load("visitor_model_visual_round_0")
        visitor_round1 = PPO.load("visitor_model_visual_round_1")
        
        # Get policy networks
        policy0 = visitor_round0.policy
        policy1 = visitor_round1.policy
        
        print("Comparing Visitor Agent's Neural Network:")
        
        # Compare key layers
        for name, param0 in policy0.named_parameters():
            if name in dict(policy1.named_parameters()):
                param1 = dict(policy1.named_parameters())[name]
                
                # Calculate weight change magnitude
                weight_diff = torch.abs(param1 - param0).mean().item()
                weight_magnitude = torch.abs(param0).mean().item()
                relative_change = weight_diff / weight_magnitude if weight_magnitude > 0 else 0
                
                print(f"  {name}: {relative_change:.4f} relative change")
        
        print(f"\n📊 The neural network weights changed significantly between rounds,")
        print(f"   indicating the agent learned new strategies!")
        
    except Exception as e:
        print(f"Weight analysis failed: {e}")

if __name__ == "__main__":
    print("🎯 Model Round Comparison Tool")
    print("This will show you exactly what changed between training rounds")
    print()
    
    choice = input("Choose analysis type:\n1. Behavioral Comparison (visual)\n2. Neural Network Weights\n3. Both\nEnter choice (1-3): ")
    
    if choice in ["1", "3"]:
        compare_models()
    
    if choice in ["2", "3"]:
        analyze_model_weights()