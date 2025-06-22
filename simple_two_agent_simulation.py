#!/usr/bin/env python3
"""
Simple Two-Agent Simulation

A stripped-down version of the multi-agent simulation with just:
- Environment (walls, doors, windows)
- Visitor Agent (controlled with arrow keys)
- Escort Agent (controlled with WASD keys)

Restructured to prepare for PettingZoo conversion with proper environment class.

Controls:
- Arrow Keys: Control Visitor Agent (red)
- WASD Keys: Control Escort Agent (orange)
- ESC: Quit
"""

import pygame
import sys
import os
import numpy as np
import random
from math import sin, cos, pi
from typing import Dict, Tuple, List, Optional, Any

# Make sure the project directory is in the path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Import basic components
from multitrack.models.simulation_environment import SimulationEnvironment
from multitrack.utils.config import *

class SimpleAgent:
    """Basic unicycle model agent with 360-degree vision"""
    
    def __init__(self, x, y, color, agent_name="Agent"):
        # State: [x, y, theta, v]
        self.state = np.array([x, y, 0.0, 0.0])
        self.color = color
        self.agent_name = agent_name
        
        # Control inputs: [linear_vel, angular_vel]
        self.controls = np.array([0.0, 0.0])
        
        # For collision handling
        self.prev_state = self.state.copy()
        
        # Agent radius
        self.radius = 10
        
        # Vision system
        self.vision_range = DEFAULT_VISION_RANGE  # From config
        self.vision_beams = 100  # Number of vision beams around 360 degrees
        self.vision_distances = [0.0] * self.vision_beams  # Distance to obstacle for each beam
        self.vision_endpoints = [(0, 0)] * self.vision_beams  # Endpoints of each beam
    
    def get_observation(self, other_agent_state: Optional[np.ndarray] = None) -> np.ndarray:
        """Get agent observation for RL training"""
        # Own state: [x, y, theta, v]
        own_state = self.state.copy()
        
        # Other agent's state: [x, y, theta, v] or zeros if not provided
        if other_agent_state is not None:
            other_state = other_agent_state.copy()
        else:
            other_state = np.zeros(4)  # [0, 0, 0, 0] if other agent not available
        
        # Vision distances (100 rays around 360 degrees)
        vision_data = np.array(self.vision_distances)
        
        # Combine all observation components
        obs = np.concatenate([
            own_state,      # [x, y, theta, v] - 4 elements
            other_state,    # [x, y, theta, v] - 4 elements  
            vision_data     # 100 vision distances
        ])
        
        return obs.astype(np.float32)  # Total: 108 elements
    
    def get_state(self) -> np.ndarray:
        """Get agent state"""
        return self.state.copy()
    
    def cast_vision_rays(self, walls, doors=None):
        """Cast 100 rays in 360 degrees around the agent to detect obstacles"""
        x, y = self.state[0], self.state[1]
        
        if doors is None:
            doors = []
        
        for i in range(self.vision_beams):
            # Calculate angle for this beam (360 degrees / 100 beams = 3.6 degrees per beam)
            angle = (2 * pi * i) / self.vision_beams
            
            # Ray direction
            dx = cos(angle)
            dy = sin(angle)
            
            # Cast ray and find first intersection
            max_distance = self.vision_range
            hit_distance = max_distance
            
            # Check intersection with walls
            for wall in walls:
                distance = self._ray_rect_intersection(x, y, dx, dy, wall)
                if distance is not None and distance < hit_distance:
                    # Check if this intersection point is within a door (if so, ignore it)
                    intersection_x = x + distance * dx
                    intersection_y = y + distance * dy
                    intersection_point = pygame.Rect(int(intersection_x) - 2, int(intersection_y) - 2, 4, 4)
                    
                    # Check if intersection is in a door area
                    in_door = False
                    for door in doors:
                        if intersection_point.colliderect(door):
                            in_door = True
                            break
                    
                    # Only count this as a hit if it's not in a door
                    if not in_door:
                        hit_distance = distance
            
            # Store results
            self.vision_distances[i] = hit_distance
            self.vision_endpoints[i] = (
                x + hit_distance * dx,
                y + hit_distance * dy
            )
    
    def _ray_rect_intersection(self, ray_x, ray_y, ray_dx, ray_dy, rect):
        """Calculate intersection distance between ray and rectangle"""
        # Ray-rectangle intersection using parametric form
        # Ray: P = (ray_x, ray_y) + t * (ray_dx, ray_dy)
        # Rectangle: (rect.x, rect.y) to (rect.x + rect.width, rect.y + rect.height)
        
        if ray_dx == 0 and ray_dy == 0:
            return None
            
        t_min = 0
        t_max = float('inf')
        
        # Check X bounds
        if ray_dx != 0:
            t1 = (rect.x - ray_x) / ray_dx
            t2 = (rect.x + rect.width - ray_x) / ray_dx
            t_min = max(t_min, min(t1, t2))
            t_max = min(t_max, max(t1, t2))
        else:
            # Ray is vertical
            if ray_x < rect.x or ray_x > rect.x + rect.width:
                return None
                
        # Check Y bounds
        if ray_dy != 0:
            t1 = (rect.y - ray_y) / ray_dy
            t2 = (rect.y + rect.height - ray_y) / ray_dy
            t_min = max(t_min, min(t1, t2))
            t_max = min(t_max, max(t1, t2))
        else:
            # Ray is horizontal
            if ray_y < rect.y or ray_y > rect.y + rect.height:
                return None
        
        if t_min <= t_max and t_min > 0:
            # Ray hits the wall
            return t_min
        
        return None
    
    def draw_vision(self, screen, show_beams=True):
        """Draw the vision system"""
        if not show_beams:
            return
            
        x, y = int(self.state[0]), int(self.state[1])
        
        # Draw vision beams
        for i in range(self.vision_beams):
            end_x, end_y = self.vision_endpoints[i]
            distance = self.vision_distances[i]
            
            # Color based on distance and make beams more subtle
            if distance < self.vision_range:
                # Hit something - color based on distance
                intensity = min(200, int(200 * (1 - distance / self.vision_range)))
                color = (intensity, 200 - intensity, 0)  # Red to green, more subtle
            else:
                # Nothing hit - blue
                color = (0, 80, 200)
            
            # Draw thin line
            pygame.draw.line(screen, color, (x, y), (int(end_x), int(end_y)), 1)
        
        # Draw vision range circle - only show a small portion to avoid clutter
        if self.vision_range < 1000:  # Only draw if range is reasonable
            pygame.draw.circle(screen, (128, 128, 128), (x, y), self.vision_range, 1)
    
    def update(self, dt=0.1, walls=None, doors=None):
        """Update agent state using unicycle dynamics"""
        # Store previous state for collision detection
        self.prev_state = self.state.copy()
        
        v, omega = self.controls
        theta = self.state[2]
        
        # Unicycle model dynamics
        self.state[0] += v * cos(theta) * dt
        self.state[1] += v * sin(theta) * dt
        self.state[2] += omega * dt
        self.state[3] = v  # Update velocity state
        
        # Normalize angle to [-pi, pi]
        self.state[2] = (self.state[2] + pi) % (2 * pi) - pi
        
        # Handle collisions if walls are provided
        if walls is not None:
            self.handle_collision(walls, doors)
        
        # Boundary conditions - keep agent on screen
        self.state[0] = np.clip(self.state[0], self.radius, WIDTH - self.radius)
        self.state[1] = np.clip(self.state[1], self.radius, HEIGHT - self.radius)
    
    def set_controls(self, linear_vel, angular_vel):
        """Set control inputs"""
        self.controls = np.array([linear_vel, angular_vel])
    
    def apply_action(self, action: np.ndarray):
        """Apply action to agent (for RL compatibility)"""
        # Action is [linear_vel, angular_vel]
        self.set_controls(action[0], action[1])
    
    def handle_collision(self, walls, doors):
        """Handle collision with walls"""
        if doors is None:
            doors = []
            
        # Create a rect for collision detection
        agent_rect = pygame.Rect(
            int(self.state[0]) - self.radius,
            int(self.state[1]) - self.radius,
            2 * self.radius, 2 * self.radius
        )
        
        collision = False
        for wall in walls:
            if agent_rect.colliderect(wall):
                # Check if we're in a door
                in_door = False
                for door in doors:
                    if agent_rect.colliderect(door):
                        in_door = True
                        break
                
                if not in_door:
                    collision = True
                    break
        
        # If collision detected, revert to previous position
        if collision:
            self.state = self.prev_state.copy()
    
    def draw(self, screen):
        """Draw the agent"""
        x, y, theta, _ = self.state
        
        # Draw agent body (circle)
        pygame.draw.circle(screen, self.color, (int(x), int(y)), self.radius)
        
        # Draw direction indicator (line showing orientation)
        end_x = x + self.radius * cos(theta)
        end_y = y + self.radius * sin(theta)
        pygame.draw.line(screen, WHITE, (x, y), (end_x, end_y), 3)
        
        # Draw agent name
        font = pygame.font.SysFont('Arial', 12)
        text = font.render(self.agent_name, True, WHITE)
        screen.blit(text, (int(x) - text.get_width() // 2, int(y) - 25))

class TwoAgentEnvironment:
    """Environment class that manages two agents and game state"""
    
    def __init__(self, width: int = WIDTH, height: int = HEIGHT, training_mode: bool = False, render_mode: Optional[str] = None):
        self.width = width
        self.height = height
        self.training_mode = training_mode
        self.render_mode = render_mode  # None, "human", "rgb_array"
        
        # Initialize pygame only if rendering is needed
        self.screen = None
        self.clock = None
        self.font = None
        
        if self.render_mode == "human":
            pygame.init()
            self.screen = pygame.display.set_mode((width, height))
            pygame.display.set_caption("RL Training Visualization")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont('Arial', 16)
        
        # Create simulation environment
        self.sim_env = SimulationEnvironment(width, height)
        
        # Agent management
        self.agents = {}
        self.agent_ids = ["visitor", "escort"]
        
        # Environment state
        self.timestep = 0
        self.max_timesteps = 1000
        
        # Reset to initialize
        self.reset()
    
    def set_render_mode(self, mode: Optional[str]):
        """Change render mode during runtime"""
        if mode != self.render_mode:
            self.render_mode = mode
            
            if mode == "human" and self.screen is None:
                pygame.init()
                self.screen = pygame.display.set_mode((self.width, self.height))
                pygame.display.set_caption("RL Training Visualization")
                self.clock = pygame.time.Clock()
                self.font = pygame.font.SysFont('Arial', 16)
            elif mode is None and self.screen is not None:
                pygame.quit()
                self.screen = None
                self.clock = None
                self.font = None

    def set_training_mode(self, training_mode: bool):
        """Enable/disable training mode"""
        self.training_mode = training_mode
    
    def reset(self) -> Dict[str, np.ndarray]:
        """Reset environment and return initial observations"""
        self.timestep = 0
        
        # Find good spawn positions
        visitor_pos, escort_pos = self._find_spawn_positions()
        
        # Create agents
        self.agents = {
            "visitor": SimpleAgent(visitor_pos[0], visitor_pos[1], RED, "Visitor"),
            "escort": SimpleAgent(escort_pos[0], escort_pos[1], ORANGE, "Escort")
        }
        
        # Update vision for both agents
        for agent in self.agents.values():
            agent.cast_vision_rays(self.sim_env.get_all_walls(), self.sim_env.doors)
        
        # Return initial observations
        return self._get_observations()
    
    def step(self, actions: Dict[str, np.ndarray]) -> Tuple[Dict[str, np.ndarray], Dict[str, float], Dict[str, bool], Dict[str, bool], Dict[str, Any]]:
        """Step environment with actions for each agent"""
        # Apply actions to agents
        for agent_id, action in actions.items():
            if agent_id in self.agents:
                self.agents[agent_id].apply_action(action)
        
        # Update agents
        dt = 1.0 / TARGET_FPS
        for agent in self.agents.values():
            agent.update(dt, self.sim_env.get_all_walls(), self.sim_env.doors)
            agent.cast_vision_rays(self.sim_env.get_all_walls(), self.sim_env.doors)
        
        # Update timestep
        self.timestep += 1
        
        # Get observations, rewards, dones, and info
        observations = self._get_observations()
        rewards = self._get_rewards()
        dones = self._get_dones()
        truncated = self._get_truncated()
        info = self._get_info()
        
        return observations, rewards, dones, truncated, info
    
    def _get_observations(self) -> Dict[str, np.ndarray]:
        """Get observations for all agents"""
        observations = {}
        
        # Get states for cross-reference
        visitor_state = self.agents["visitor"].get_state() if "visitor" in self.agents else None
        escort_state = self.agents["escort"].get_state() if "escort" in self.agents else None
        
        # Each agent gets its own observation with the other agent's state
        if "visitor" in self.agents:
            observations["visitor"] = self.agents["visitor"].get_observation(escort_state)
        
        if "escort" in self.agents:
            observations["escort"] = self.agents["escort"].get_observation(visitor_state)
            
        return observations
    
    def _get_rewards(self) -> Dict[str, float]:
        """Calculate rewards for all agents - adversarial setup"""
        rewards = {}
        
        # Initialize rewards
        for agent_id in self.agents:
            rewards[agent_id] = 0.0
        
        # Get current game state
        line_of_sight = self._can_see_each_other()
        distance = self._get_agent_distance()
        episode_done = not line_of_sight  # Episode ends when line of sight is lost
        
        # ESCORT REWARDS (wants to maintain surveillance)
        if "escort" in self.agents:
            if line_of_sight:
                # Positive reward for maintaining line of sight
                rewards["escort"] += 1.0
                
                # Distance-based rewards: encourage optimal following distance
                optimal_distance = 120.0  # Target distance
                distance_error = abs(distance - optimal_distance)
                
                if distance < 50:  # Too close - penalty
                    rewards["escort"] -= 2.0
                elif distance > 300:  # Too far - penalty  
                    rewards["escort"] -= 1.0
                else:
                    # Reward for being in good range, bonus for optimal distance
                    distance_reward = max(0, 1.0 - (distance_error / optimal_distance))
                    rewards["escort"] += distance_reward
            
            if episode_done:  # Lost line of sight - game over penalty
                rewards["escort"] -= 10.0
        
        # VISITOR REWARDS (wants to escape surveillance)
        if "visitor" in self.agents:
            # Base penalty for being visible to escort
            if line_of_sight:
                rewards["visitor"] -= 0.5
                
                # DISTANCE-BASED REWARDS: Encourage visitor to increase distance
                if distance > 250:
                    rewards["visitor"] += 2.0  # Strong reward for being far
                elif distance > 200:
                    rewards["visitor"] += 1.2  # Good reward for decent distance
                elif distance > 150:
                    rewards["visitor"] += 0.6  # Small reward for moderate distance
                elif distance < 100:
                    rewards["visitor"] -= 1.0  # Penalty for being too close
                
                # MOVEMENT REWARD: Encourage visitor to move away from escort
                distance_change_reward = self._calculate_distance_change_reward("visitor")
                rewards["visitor"] += distance_change_reward
            
            # SUCCESS BONUS: Successfully broke line of sight
            if episode_done:
                rewards["visitor"] += 15.0
            
            # ENHANCED EXPLORATION REWARDS: Reward for finding open spaces and long sight lines
            exploration_reward = self._calculate_enhanced_exploration_reward("visitor")
            rewards["visitor"] += exploration_reward
            
            # STRATEGIC POSITIONING REWARDS: Reward for tactical positioning
            positioning_reward = self._calculate_strategic_positioning_reward("visitor")
            rewards["visitor"] += positioning_reward
            
        return rewards
    
    def _calculate_enhanced_exploration_reward(self, agent_id: str) -> float:
        """Enhanced reward system for encouraging exploration of open spaces"""
        if agent_id not in self.agents:
            return 0.0
        
        agent = self.agents[agent_id]
        vision_distances = agent.vision_distances
        vision_range = agent.vision_range
        
        # 1. OPENNESS METRICS
        total_openness = 0.0
        far_sight_lines = []
        open_directions = 0
        escape_routes = 0
        
        for i, distance in enumerate(vision_distances):
            # Normalize distance to [0, 1] where 1 is maximum vision range
            normalized_distance = min(distance / vision_range, 1.0)
            total_openness += normalized_distance
            
            # Count directions with significant openness (>60% of max range)
            if distance > vision_range * 0.6:
                open_directions += 1
                
            # Count escape routes (>80% of max range)
            if distance > vision_range * 0.8:
                escape_routes += 1
                far_sight_lines.append(distance)
        
        # Average openness across all vision beams
        avg_openness = total_openness / len(vision_distances)
        
        # 2. LONG SIGHT LINE BONUSES
        long_sight_bonus = 0.0
        if far_sight_lines:
            # Reward having many far sight lines
            far_sight_ratio = len(far_sight_lines) / len(vision_distances)
            long_sight_bonus = far_sight_ratio * 2.0  # Up to 2.0 bonus
            
            # Extra bonus for having VERY long sight lines
            ultra_long_lines = sum(1 for d in far_sight_lines if d > vision_range * 0.95)
            if ultra_long_lines > 0:
                long_sight_bonus += (ultra_long_lines / len(vision_distances)) * 1.5
        
        # 3. ESCAPE ROUTE DIVERSITY
        escape_diversity_bonus = 0.0
        if escape_routes > 0:
            # Reward having escape routes in multiple directions
            escape_diversity_bonus = min(escape_routes / 20.0, 1.0)  # Up to 1.0 bonus
            
            # Extra bonus for having escape routes spread around (not clustered)
            if escape_routes >= 8:  # At least 8 escape routes
                escape_diversity_bonus += 0.5
        
        # 4. VISIBILITY ADVANTAGE BONUS
        visibility_advantage = 0.0
        max_distance = max(vision_distances) if vision_distances else 0
        if max_distance > vision_range * 0.9:
            # Bonus for being in a position where you can see very far
            visibility_advantage = 1.0
            
            # Extra bonus if you can see far in multiple directions
            very_far_directions = sum(1 for d in vision_distances if d > vision_range * 0.85)
            if very_far_directions >= 5:
                visibility_advantage += 0.8
        
        # 5. COMBINE ALL REWARDS
        exploration_reward = (
            avg_openness * 0.5 +           # Base openness reward
            long_sight_bonus * 0.3 +       # Long sight line bonus
            escape_diversity_bonus * 0.8 + # Escape route diversity
            visibility_advantage * 0.6     # Visibility advantage
        )
        
        return exploration_reward
    
    def _calculate_strategic_positioning_reward(self, agent_id: str) -> float:
        """Reward visitor for strategic positioning that aids escape"""
        if agent_id not in self.agents or "escort" not in self.agents:
            return 0.0
        
        visitor = self.agents[agent_id]
        escort = self.agents["escort"]
        
        visitor_pos = visitor.state[:2]
        escort_pos = escort.state[:2]
        
        # 1. POSITIONING RELATIVE TO ESCORT
        distance_to_escort = np.linalg.norm(visitor_pos - escort_pos)
        
        # Reward for maintaining optimal distance for escape
        optimal_escape_distance = 150.0  # Sweet spot - not too close, not too far
        distance_error = abs(distance_to_escort - optimal_escape_distance)
        distance_reward = max(0, 1.0 - (distance_error / optimal_escape_distance)) * 0.5
        
        # 2. ANGULAR POSITIONING ADVANTAGE
        # Reward visitor for positioning where they have more escape options than escort
        visitor_escape_options = sum(1 for d in visitor.vision_distances if d > visitor.vision_range * 0.7)
        
        # Estimate escort's potential escape options if they were at visitor's position
        # This is a rough approximation - reward visitor for being in better positions
        angular_advantage = 0.0
        if visitor_escape_options > 15:  # If visitor has many escape options
            angular_advantage = min(visitor_escape_options / 50.0, 0.8)  # Up to 0.8 bonus
        
        # 3. DOORWAY AND CHOKEPOINT UTILIZATION
        # Reward visitor for being near doorways (potential escape routes)
        doorway_bonus = self._calculate_doorway_proximity_bonus(visitor_pos)
        
        # 4. ROOM POSITIONING STRATEGY
        # Reward visitor for being in rooms with multiple exits
        room_strategy_bonus = self._calculate_room_strategy_bonus(visitor_pos)
        
        # Combine strategic positioning rewards
        positioning_reward = (
            distance_reward +        # Optimal distance from escort
            angular_advantage +      # Having more escape options
            doorway_bonus +         # Being near escape routes
            room_strategy_bonus     # Being in strategically good rooms
        )
        
        return positioning_reward
    
    def _calculate_doorway_proximity_bonus(self, agent_pos: np.ndarray) -> float:
        """Reward visitor for being near doorways (escape routes)"""
        if not hasattr(self.sim_env, 'doors'):
            return 0.0
        
        max_bonus = 0.0
        proximity_threshold = 80.0  # Distance threshold for doorway bonus
        
        for door in self.sim_env.doors:
            # Calculate distance to door center
            door_center = np.array([door.centerx, door.centery])
            distance_to_door = np.linalg.norm(agent_pos - door_center)
            
            if distance_to_door < proximity_threshold:
                # Bonus inversely proportional to distance
                bonus = (1.0 - distance_to_door / proximity_threshold) * 0.4
                max_bonus = max(max_bonus, bonus)
        
        return max_bonus
    
    def _calculate_room_strategy_bonus(self, agent_pos: np.ndarray) -> float:
        """Reward visitor for being in rooms with good strategic value"""
        # This is a simplified version - could be enhanced with actual room detection
        x, y = agent_pos
        
        # Identify which area/room the visitor is in based on position
        # These are rough approximations based on the house layout
        
        # Central areas (hallways, open spaces) - good for escape
        if (0.4 * self.width < x < 0.9 * self.width and 
            0.3 * self.height < y < 0.6 * self.height):
            return 0.6  # High bonus for central areas
        
        # Large rooms with multiple exits (living room, dining room)
        if ((0.2 * self.width < x < 0.6 * self.width and 0.4 * self.height < y < 0.8 * self.height) or
            (0.7 * self.width < x < 0.95 * self.width and 0.1 * self.height < y < 0.3 * self.height)):
            return 0.4  # Medium bonus for large rooms
        
        # Corner rooms (potential traps) - small penalty
        if ((x < 0.2 * self.width or x > 0.9 * self.width) and 
            (y < 0.2 * self.height or y > 0.8 * self.height)):
            return -0.2  # Small penalty for corners
        
        return 0.0  # Neutral for other areas
    
    def _calculate_distance_change_reward(self, agent_id: str) -> float:
        """Reward visitor for increasing distance from escort"""
        if agent_id not in self.agents or "escort" not in self.agents:
            return 0.0
        
        # Track previous distance to calculate change
        if not hasattr(self, '_prev_distance'):
            self._prev_distance = self._get_agent_distance()
            return 0.0
        
        current_distance = self._get_agent_distance()
        distance_change = current_distance - self._prev_distance
        self._prev_distance = current_distance
        
        # Reward for increasing distance (moving away from escort)
        if distance_change > 0:
            # Positive reward for moving away, scaled by magnitude
            return min(distance_change * 0.1, 1.0)  # Up to 1.0 reward
        elif distance_change < 0:
            # Small penalty for moving closer
            return max(distance_change * 0.05, -0.5)  # Up to -0.5 penalty
        
        return 0.0  # No change in distance
    
    def _get_dones(self) -> Dict[str, bool]:
        """Check if any agent reached a terminal state (episode end)"""
        dones = {}
        
        # Common condition: max timesteps reached
        if self.timestep >= self.max_timesteps:
            for agent_id in self.agents:
                dones[agent_id] = True
        
        # Specific condition for losing line of sight
        if not self._can_see_each_other():
            if "visitor" in self.agents:
                dones["visitor"] = True
            if "escort" in self.agents:
                dones["escort"] = True
        
        return dones
    
    def _get_truncated(self) -> Dict[str, bool]:
        """Check if any agent's episode was truncated (prematurely ended)"""
        truncated = {}
        
        # For now, same as done condition - can be made more sophisticated
        for agent_id in self.agents:
            truncated[agent_id] = False  # No truncation by default
        
        return truncated
    
    def _get_info(self) -> Dict[str, Any]:
        """Get additional info for the current step (empty for now)"""
        info = {}
        
        return info
    
    def render(self, screen, font, show_vision: bool = True):
        """Render the environment and agents"""
        # Draw environment
        self.sim_env.draw(screen, font)
        
        # Draw vision beams for both agents (behind everything else)
        if show_vision:
            for agent in self.agents.values():
                agent.draw_vision(screen, show_beams=True)
        
        # Draw line of sight between agents
        if len(self.agents) == 2:
            visitor_pos = self.agents["visitor"].state[:2]
            escort_pos = self.agents["escort"].state[:2]
            
            if self._line_of_sight(visitor_pos, escort_pos):
                # Draw green line if they can see each other
                pygame.draw.line(screen, GREEN, visitor_pos, escort_pos, 2)
            else:
                # Draw red dashed line if they can't see each other
                self._draw_dashed_line(screen, visitor_pos, escort_pos, RED)
        
        # Draw agents
        for agent in self.agents.values():
            agent.draw(screen)
    
    def _draw_dashed_line(self, screen, start_pos, end_pos, color):
        """Draw a dashed line between two positions"""
        distance = np.linalg.norm(np.array(end_pos) - np.array(start_pos))
        segments = int(distance / 10)
        for i in range(0, segments, 2):  # Every other segment
            t1 = i / segments
            t2 = min((i + 1) / segments, 1)
            x1 = start_pos[0] + t1 * (end_pos[0] - start_pos[0])
            y1 = start_pos[1] + t1 * (end_pos[1] - start_pos[1])
            x2 = start_pos[0] + t2 * (end_pos[0] - start_pos[0])
            y2 = start_pos[1] + t2 * (end_pos[1] - start_pos[1])
            pygame.draw.line(screen, color, (x1, y1), (x2, y2), 2)

def find_free_spawn_positions(environment, agent_radius=10, min_distance=100, max_distance=200):
    """Find two good spawn positions that are in free space, can see each other, and are between min_distance and max_distance apart"""
    # Generate more varied spawn candidates with some randomization
    base_candidates = [
        # Living room area variations
        (WIDTH * 0.25, HEIGHT * 0.40), (WIDTH * 0.20, HEIGHT * 0.35), (WIDTH * 0.30, HEIGHT * 0.45),
        # Kitchen area variations
        (WIDTH * 0.60, HEIGHT * 0.15), (WIDTH * 0.55, HEIGHT * 0.20), (WIDTH * 0.65, HEIGHT * 0.10),
        # Center hallway variations
        (WIDTH * 0.80, HEIGHT * 0.40), (WIDTH * 0.75, HEIGHT * 0.35), (WIDTH * 0.85, HEIGHT * 0.45),
        # Dining room variations
        (WIDTH * 0.80, HEIGHT * 0.15), (WIDTH * 0.75, HEIGHT * 0.10), (WIDTH * 0.85, HEIGHT * 0.20),
        # Lower open area variations
        (WIDTH * 0.30, HEIGHT * 0.80), (WIDTH * 0.25, HEIGHT * 0.75), (WIDTH * 0.35, HEIGHT * 0.85),
        # Master bedroom variations
        (WIDTH * 0.92, HEIGHT * 0.15), (WIDTH * 0.88, HEIGHT * 0.12), (WIDTH * 0.95, HEIGHT * 0.18),
        # Game room variations
        (WIDTH * 0.92, HEIGHT * 0.40), (WIDTH * 0.88, HEIGHT * 0.35), (WIDTH * 0.95, HEIGHT * 0.45),
        # Library variations
        (WIDTH * 0.92, HEIGHT * 0.60), (WIDTH * 0.88, HEIGHT * 0.55), (WIDTH * 0.95, HEIGHT * 0.65),
        # Upper bedroom variations
        (WIDTH * 0.15, HEIGHT * 0.15), (WIDTH * 0.12, HEIGHT * 0.12), (WIDTH * 0.18, HEIGHT * 0.18),
        # Study area variations
        (WIDTH * 0.12, HEIGHT * 0.45), (WIDTH * 0.08, HEIGHT * 0.40), (WIDTH * 0.16, HEIGHT * 0.50),
        # Additional varied positions
        (WIDTH * 0.35, HEIGHT * 0.25), (WIDTH * 0.40, HEIGHT * 0.30), (WIDTH * 0.32, HEIGHT * 0.22),
        (WIDTH * 0.45, HEIGHT * 0.35), (WIDTH * 0.50, HEIGHT * 0.40), (WIDTH * 0.42, HEIGHT * 0.32),
        (WIDTH * 0.65, HEIGHT * 0.25), (WIDTH * 0.70, HEIGHT * 0.30), (WIDTH * 0.62, HEIGHT * 0.22),
        (WIDTH * 0.75, HEIGHT * 0.30), (WIDTH * 0.78, HEIGHT * 0.35), (WIDTH * 0.72, HEIGHT * 0.25),
        (WIDTH * 0.20, HEIGHT * 0.60), (WIDTH * 0.25, HEIGHT * 0.65), (WIDTH * 0.18, HEIGHT * 0.55),
        (WIDTH * 0.40, HEIGHT * 0.65), (WIDTH * 0.45, HEIGHT * 0.70), (WIDTH * 0.38, HEIGHT * 0.60),
        # Corridor and transitional spaces
        (WIDTH * 0.55, HEIGHT * 0.50), (WIDTH * 0.60, HEIGHT * 0.45), (WIDTH * 0.50, HEIGHT * 0.55),
        (WIDTH * 0.70, HEIGHT * 0.50), (WIDTH * 0.68, HEIGHT * 0.55), (WIDTH * 0.72, HEIGHT * 0.45),
    ]
    
    # Add small random variations to each candidate to increase variety
    spawn_candidates = []
    for base_x, base_y in base_candidates:
        # Add 3 variations of each base position with small random offsets
        for _ in range(3):
            offset_x = random.uniform(-WIDTH * 0.03, WIDTH * 0.03)  # ±3% screen width
            offset_y = random.uniform(-HEIGHT * 0.03, HEIGHT * 0.03)  # ±3% screen height
            new_x = max(50, min(WIDTH - 50, base_x + offset_x))
            new_y = max(50, min(HEIGHT - 50, base_y + offset_y))
            spawn_candidates.append((new_x, new_y))
    
    # Shuffle the candidates to add randomness to selection order
    random.shuffle(spawn_candidates)
    
    def is_position_free(x, y, walls, doors):
        """Check if a position is free of walls"""
        agent_rect = pygame.Rect(
            int(x) - agent_radius, int(y) - agent_radius,
            2 * agent_radius, 2 * agent_radius
        )
        
        for wall in walls:
            if agent_rect.colliderect(wall):
                # Check if we're in a door
                in_door = False
                for door in doors:
                    if agent_rect.colliderect(door):
                        in_door = True
                        break
                if not in_door:
                    return False
        return True
    
    def can_see_each_other(pos1, pos2, walls, doors):
        """Simple line-of-sight check between two positions"""
        x1, y1 = pos1
        x2, y2 = pos2
        
        # Simple ray casting - check if line between agents intersects walls
        steps = 50
        for i in range(steps + 1):
            t = i / steps
            x = x1 + t * (x2 - x1)
            y = y1 + t * (y2 - y1)
            
            # Check if this point intersects a wall
            point_rect = pygame.Rect(int(x) - 2, int(y) - 2, 4, 4)
            for wall in walls:
                if point_rect.colliderect(wall):
                    # Check if we're in a door
                    in_door = False
                    for door in doors:
                        if point_rect.colliderect(door):
                            in_door = True
                            break
                    if not in_door:
                        return False
        return True
    
    def calculate_distance(pos1, pos2):
        """Calculate distance between two positions"""
        return ((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)**0.5
    
    # Find two positions that work well together
    walls = environment.get_all_walls()
    doors = environment.doors
    
    best_pairs = []  # Store valid pairs with their distances
    
    for i, pos1 in enumerate(spawn_candidates):
        if is_position_free(pos1[0], pos1[1], walls, doors):
            for j, pos2 in enumerate(spawn_candidates[i+1:], i+1):
                if is_position_free(pos2[0], pos2[1], walls, doors):
                    distance = calculate_distance(pos1, pos2)
                    # Check if they can see each other and are within the desired distance range
                    if (min_distance <= distance <= max_distance and 
                        can_see_each_other(pos1, pos2, walls, doors)):
                        best_pairs.append((pos1, pos2, distance))
    
    # Sort by distance (prefer closer pairs that still have line of sight)
    best_pairs.sort(key=lambda x: x[2])
    
    if best_pairs:
        # Add better randomization - choose from more options and weight by variety
        num_options = min(8, len(best_pairs))  # Choose from top 8 options instead of 3
        
        # Add some preference for medium distances (not always the closest)
        weighted_pairs = []
        for i, pair in enumerate(best_pairs[:num_options]):
            # Give higher weight to positions that aren't the absolute closest
            weight = 1.0 + (i * 0.5)  # Slightly prefer positions that aren't #1 closest
            weighted_pairs.extend([pair] * int(weight * 2))
        
        chosen_pair = random.choice(weighted_pairs)
        print(f"Found spawn positions with distance: {chosen_pair[2]:.1f} pixels")
        return chosen_pair[0], chosen_pair[1]
    
    # Fallback: find two positions that are at least free of walls and within distance range
    print("No perfect pairs found, trying fallback positions...")
    for pos1 in spawn_candidates:
        if is_position_free(pos1[0], pos1[1], walls, doors):
            for pos2 in spawn_candidates:
                if (pos2 != pos1 and 
                    is_position_free(pos2[0], pos2[1], walls, doors)):
                    fallback_distance = calculate_distance(pos1, pos2)
                    if min_distance <= fallback_distance <= max_distance:
                        print(f"Fallback spawn positions with distance: {fallback_distance:.1f} pixels")
                        return pos1, pos2
    
    # Final fallback: use positions that are at least min_distance apart, even without line of sight
    print("Using final fallback positions (may not have line of sight)...")
    for pos1 in spawn_candidates:
        if is_position_free(pos1[0], pos1[1], walls, doors):
            for pos2 in spawn_candidates:
                if (pos2 != pos1 and 
                    is_position_free(pos2[0], pos2[1], walls, doors)):
                    final_distance = calculate_distance(pos1, pos2)
                    if final_distance >= min_distance:
                        print(f"Final fallback distance: {final_distance:.1f} pixels")
                        return pos1, pos2
    
    # Absolute final fallback: use well-separated default positions
    print("Using absolute default positions...")
    visitor_pos = (WIDTH * 0.15, HEIGHT * 0.15)  # Upper left (bedroom)
    escort_pos = (WIDTH * 0.85, HEIGHT * 0.85)   # Lower right (far corner)
    final_distance = calculate_distance(visitor_pos, escort_pos)
    print(f"Default separation distance: {final_distance:.1f} pixels")
    
    return visitor_pos, escort_pos

def respawn_agents(environment):
    """Find new spawn positions and create new agents"""
    # Track recent spawn positions to avoid repetition
    if not hasattr(respawn_agents, 'recent_spawns'):
        respawn_agents.recent_spawns = []
    
    max_attempts = 10
    for attempt in range(max_attempts):
        visitor_pos, escort_pos = find_free_spawn_positions(environment)
        
        # Check if this combination is too similar to recent spawns
        current_spawn = (visitor_pos, escort_pos)
        is_too_similar = False
        
        for recent_visitor, recent_escort in respawn_agents.recent_spawns:
            visitor_dist = ((visitor_pos[0] - recent_visitor[0])**2 + (visitor_pos[1] - recent_visitor[1])**2)**0.5
            escort_dist = ((escort_pos[0] - recent_escort[0])**2 + (escort_pos[1] - recent_escort[1])**2)**0.5
            
            # If both agents are within 80 pixels of a recent spawn, it's too similar
            if visitor_dist < 80 and escort_dist < 80:
                is_too_similar = True
                break
        
        if not is_too_similar:
            # Store this spawn in recent history (keep last 5)
            respawn_agents.recent_spawns.append(current_spawn)
            if len(respawn_agents.recent_spawns) > 5:
                respawn_agents.recent_spawns.pop(0)
            break
    
    # Create new agents at the found positions
    visitor = SimpleAgent(visitor_pos[0], visitor_pos[1], RED, "Visitor")
    escort = SimpleAgent(escort_pos[0], escort_pos[1], ORANGE, "Escort")
    
    # Calculate and display spawn distance
    spawn_distance = ((visitor_pos[0] - escort_pos[0])**2 + (visitor_pos[1] - escort_pos[1])**2)**0.5
    
    print(f"\n🔄 RESPAWNED AGENTS (Attempt {attempt + 1}):")
    print(f"Visitor at ({visitor_pos[0]:.0f}, {visitor_pos[1]:.0f})")
    print(f"Escort at ({escort_pos[0]:.0f}, {escort_pos[1]:.0f})")
    print(f"Distance: {spawn_distance:.1f} pixels (target: 100-200) {'✅' if 100 <= spawn_distance <= 200 else '❌'}")
    
    return visitor, escort

def main():
    """Main simulation loop"""
    pygame.init()
    
    # Set up display
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Simple Two-Agent Simulation")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont('Arial', 16)
    
    # Create environment - check if we're in training mode
    training_mode = TRAINING_MODE  # Use config flag
    env = TwoAgentEnvironment(WIDTH, HEIGHT, training_mode=training_mode)
    
    # Simulation state
    running = True
    paused = False
    show_vision = True
    reset_message_timer = 0
    
    print("Simple Two-Agent Simulation Started!")
    if training_mode:
        print("🚀 TRAINING MODE: Frame rate limiting disabled for maximum speed")
    print("\nControls:")
    print("  Visitor Agent (Red):")
    print("    ↑ : Move Forward")
    print("    ↓ : Move Backward") 
    print("    ← : Turn Left")
    print("    → : Turn Right")
    print("")
    print("  Escort Agent (Orange):")
    print("    W : Move Forward")
    print("    S : Move Backward")
    print("    A : Turn Left") 
    print("    D : Turn Right")
    print("")
    print("  Other:")
    print("    SPACE : Pause/Unpause")
    print("    V : Toggle Vision Beams")
    print("    R : Reset/Respawn Agents in New Locations")
    print("    T : Toggle Training Mode (disables frame rate limiting)")
    print("    ESC : Quit")
    
    while running:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                    print("Simulation", "PAUSED" if paused else "RESUMED")
                elif event.key == pygame.K_v:
                    show_vision = not show_vision
                    print("Vision beams", "ENABLED" if show_vision else "DISABLED")
                elif event.key == pygame.K_t:
                    # Toggle training mode
                    training_mode = not training_mode
                    env.set_training_mode(training_mode)
                    print("Training mode", "ENABLED" if training_mode else "DISABLED")
                    print("Frame rate limiting", "DISABLED" if training_mode else "ENABLED")
                elif event.key == pygame.K_r:
                    # Reset environment
                    env.reset()
                    reset_message_timer = 120
                    print("Environment reset!")
        
        # Handle continuous key input and convert to actions
        if not paused:
            keys = pygame.key.get_pressed()
            
            # Convert key input to actions
            actions = {}
            
            # Visitor controls (Arrow keys)
            visitor_linear_vel = 0
            visitor_angular_vel = 0
            
            if keys[pygame.K_UP]:
                visitor_linear_vel = LEADER_LINEAR_VEL
            if keys[pygame.K_DOWN]:
                visitor_linear_vel = -LEADER_LINEAR_VEL
            if keys[pygame.K_LEFT]:
                visitor_angular_vel = -LEADER_ANGULAR_VEL
            if keys[pygame.K_RIGHT]:
                visitor_angular_vel = LEADER_ANGULAR_VEL
                
            actions["visitor"] = np.array([visitor_linear_vel, visitor_angular_vel])
            
            # Escort controls (WASD keys)
            escort_linear_vel = 0
            escort_angular_vel = 0
            
            if keys[pygame.K_w]:
                escort_linear_vel = FOLLOWER_LINEAR_VEL_MAX
            if keys[pygame.K_s]:
                escort_linear_vel = -FOLLOWER_LINEAR_VEL_MAX
            if keys[pygame.K_a]:
                escort_angular_vel = -LEADER_ANGULAR_VEL
            if keys[pygame.K_d]:
                escort_angular_vel = LEADER_ANGULAR_VEL
                
            actions["escort"] = np.array([escort_linear_vel, escort_angular_vel])
            
            # Step environment
            observations, rewards, dones, truncated, info = env.step(actions)
            
            # Update reset message timer
            if reset_message_timer > 0:
                reset_message_timer -= 1
        
        # Clear screen
        screen.fill(BLACK)
        
        # Render environment
        env.render(screen, font, show_vision)
        
        # Draw status information
        status_y = 10
        
        # Get current agent states for display
        if "visitor" in env.agents and "escort" in env.agents:
            visitor = env.agents["visitor"]
            escort = env.agents["escort"]
            
            visitor_linear_vel, visitor_angular_vel = visitor.controls
            escort_linear_vel, escort_angular_vel = escort.controls
            
            # Show control status
            visitor_status = f"Visitor: L={visitor_linear_vel:.1f}, A={visitor_angular_vel:.1f}"
            escort_status = f"Escort: L={escort_linear_vel:.1f}, A={escort_angular_vel:.1f}"
            
            line_of_sight = env._can_see_each_other()
            sight_status = f"Line of Sight: {'YES' if line_of_sight else 'NO'}"
            
            current_distance = env._get_agent_distance()
            distance_status = f"Distance: {current_distance:.1f}px"
            
            visitor_surface = font.render(visitor_status, True, RED)
            escort_surface = font.render(escort_status, True, ORANGE)
            sight_surface = font.render(sight_status, True, GREEN if line_of_sight else RED)
            distance_surface = font.render(distance_status, True, WHITE)
            
            screen.blit(visitor_surface, (10, status_y))
            screen.blit(escort_surface, (10, status_y + 25))
            screen.blit(sight_surface, (10, status_y + 50))
            screen.blit(distance_surface, (10, status_y + 75))
            status_y += 100
        
        # Show timestep
        timestep_text = f"Timestep: {env.timestep}/{env.max_timesteps}"
        timestep_surface = font.render(timestep_text, True, WHITE)
        screen.blit(timestep_surface, (10, status_y))
        
        # Show training mode status
        if training_mode:
            training_text = "🚀 TRAINING MODE (No FPS limit)"
            training_surface = font.render(training_text, True, CYAN)
            screen.blit(training_surface, (10, status_y + 25))
            status_y += 50
        
        # Show pause status
        if paused:
            pause_text = font.render("PAUSED - Press SPACE to resume", True, YELLOW)
            screen.blit(pause_text, (WIDTH - pause_text.get_width() - 10, 10))
        
        # Show vision status
        vision_text = f"Vision: {'ON' if show_vision else 'OFF'} (V to toggle)"
        vision_surface = font.render(vision_text, True, GREEN if show_vision else RED)
        screen.blit(vision_surface, (WIDTH - vision_surface.get_width() - 10, 35))
        
        # Show reset message if recently reset
        if reset_message_timer > 0:
            reset_text = "🔄 ENVIRONMENT RESET!"
            reset_font = pygame.font.SysFont('Arial', 24, bold=True)
            reset_surface = reset_font.render(reset_text, True, YELLOW)
            screen.blit(reset_surface, (WIDTH//2 - reset_surface.get_width()//2, 50))
        
        # Show FPS
        fps = clock.get_fps()
        fps_text = font.render(f"FPS: {fps:.1f}", True, WHITE)
        screen.blit(fps_text, (WIDTH - fps_text.get_width() - 10, HEIGHT - 25))
        
        # Update display
        pygame.display.flip()
        
        # Apply frame rate limiting only if not in training mode
        if not training_mode:
            clock.tick(TARGET_FPS)
        else:
            # In training mode, just tick without limiting FPS
            clock.tick()
    
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
