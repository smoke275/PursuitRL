#!/usr/bin/env python3
"""
Clean PettingZoo Environment for Stable-Baselines3 Training

Simple parallel environment focused on Stable-Baselines3 compatibility.
No AEC complexity - just pure parallel multi-agent RL.
"""

import numpy as np
import pygame
import functools
from typing import Dict, Any, Optional
from gymnasium import spaces
from pettingzoo import ParallelEnv

# Import the base environment
from simple_two_agent_simulation import TwoAgentEnvironment

class PursuitEvasionEnv(ParallelEnv):
    """
    Clean PettingZoo Parallel Environment for Stable-Baselines3
    """
    
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "name": "pursuit_evasion_v1",
        "is_parallelizable": True,
    }
    
    def __init__(self, width: int = 1280, height: int = 720, render_mode: Optional[str] = None):
        super().__init__()
        
        # Store render mode
        self.render_mode = render_mode
        
        # Create the base environment
        self.base_env = TwoAgentEnvironment(
            width=width, 
            height=height, 
            training_mode=True,
            render_mode=render_mode
        )
        
        # Agent setup
        self.agents = ["visitor", "escort"]
        self.possible_agents = self.agents[:]
        
        # Action and observation spaces
        self._action_spaces = {
            agent: spaces.Box(
                low=np.array([-50.0, -1.0], dtype=np.float32), 
                high=np.array([50.0, 1.0], dtype=np.float32), 
                dtype=np.float32
            ) for agent in self.agents
        }
        
        self._observation_spaces = {
            agent: spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(108,),  # 4 (own state) + 4 (other state) + 100 (vision)
                dtype=np.float32
            ) for agent in self.agents
        }
    
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return self._observation_spaces[agent]
    
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return self._action_spaces[agent]
    
    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        
        observations = self.base_env.reset()
        self.agents = self.possible_agents[:]
        infos = {agent: {} for agent in self.agents}
        return observations, infos
    
    def step(self, actions):
        observations, rewards, terminations, truncations, infos = self.base_env.step(actions)
        
        # Remove terminated/truncated agents
        self.agents = [
            agent for agent in self.agents 
            if not (terminations.get(agent, False) or truncations.get(agent, False))
        ]
        
        return observations, rewards, terminations, truncations, infos
    
    def render(self):
        if self.render_mode == "human":
            if self.base_env.screen is not None:
                self.base_env.screen.fill((0, 0, 0))
                self.base_env.render(self.base_env.screen, self.base_env.font, show_vision=True)
                pygame.display.flip()
        elif self.render_mode == "rgb_array":
            if self.base_env.screen is not None:
                self.base_env.screen.fill((0, 0, 0))
                self.base_env.render(self.base_env.screen, self.base_env.font, show_vision=True)
                return np.array(pygame.surfarray.array3d(self.base_env.screen))
        return None
    
    def close(self):
        if self.base_env.screen is not None:
            pygame.quit()

# Simple factory function
def env(**kwargs):
    """Create the environment"""
    return PursuitEvasionEnv(**kwargs)