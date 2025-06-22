#!/usr/bin/env python3
"""
Simple Two-Agent Simulation

A stripped-down version of the multi-agent simulation with just:
- Environment (walls, doors, windows)
- Visitor Agent (controlled with arrow keys)
- Escort Agent (controlled with WASD keys)

No map graphs, pathfinding, or complex features - just basic movement and visualization.

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
    
    def cast_vision_rays(self, walls, doors):
        """Cast 100 rays in 360 degrees around the agent to detect obstacles"""
        x, y = self.state[0], self.state[1]
        
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
                # Use simple ray-rectangle intersection
                distance = self._ray_rect_intersection(x, y, dx, dy, wall, doors)
                if distance is not None and distance < hit_distance:
                    hit_distance = distance
            
            # Store results
            self.vision_distances[i] = hit_distance
            self.vision_endpoints[i] = (
                x + hit_distance * dx,
                y + hit_distance * dy
            )
    
    def _ray_rect_intersection(self, ray_x, ray_y, ray_dx, ray_dy, rect, doors):
        """Calculate intersection distance between ray and rectangle"""
        # Check if this wall has a door that the ray passes through
        def ray_passes_through_door(ray_x, ray_y, ray_dx, ray_dy, door_rect, max_t):
            """Check if ray passes through a door within max_t distance"""
            # Simple check: sample points along the ray and see if any are in the door
            steps = 20
            for i in range(steps):
                t = (max_t * i) / steps
                test_x = ray_x + t * ray_dx
                test_y = ray_y + t * ray_dy
                test_point = pygame.Rect(int(test_x)-1, int(test_y)-1, 2, 2)
                if test_point.colliderect(door_rect):
                    return True
            return False
        
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
            # Check if ray passes through a door in this wall
            for door in doors:
                if door.colliderect(rect):  # Door is in this wall
                    if ray_passes_through_door(ray_x, ray_y, ray_dx, ray_dy, door, t_min):
                        return None  # Ray passes through door, no collision
            
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
    
    # Create environment
    environment = SimulationEnvironment(WIDTH, HEIGHT)
    
    # Find good spawn positions
    visitor_pos, escort_pos = find_free_spawn_positions(environment)
    
    # Create agents at the found positions
    visitor = SimpleAgent(visitor_pos[0], visitor_pos[1], RED, "Visitor")
    escort = SimpleAgent(escort_pos[0], escort_pos[1], ORANGE, "Escort")
    
    # Calculate and display spawn distance
    spawn_distance = ((visitor_pos[0] - escort_pos[0])**2 + (visitor_pos[1] - escort_pos[1])**2)**0.5
    
    print(f"Spawned Visitor at ({visitor_pos[0]:.0f}, {visitor_pos[1]:.0f})")
    print(f"Spawned Escort at ({escort_pos[0]:.0f}, {escort_pos[1]:.0f})")
    print(f"Distance between agents: {spawn_distance:.1f} pixels (target: 100-200)")
    print(f"Within distance range: {'✅ YES' if 100 <= spawn_distance <= 200 else '❌ NO'}")
    
    # Simulation state
    running = True
    paused = False
    show_vision = True  # Toggle for vision beams
    reset_message_timer = 0  # Timer for showing reset message
    
    print("Simple Two-Agent Simulation Started!")
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
    print("    ESC : Quit")
    print("")
    print("Visual Features:")
    print("  🟢 Green Line: Agents can see each other")
    print("  🔴 Red Dashed Line: Agents cannot see each other")
    print("  Vision Beams: 100 rays per agent, 360° coverage")
    print("    - Red to Green: Close to far obstacles")
    print("    - Blue: No obstacle in range")
    print("    - Range: 800 pixels")
    print("  Line of sight is blocked by walls but not doors")
    print("")
    print("Reset Feature:")
    print("  🔄 Press 'R' to respawn agents in new random locations")
    print("  ✅ Maintains 100-200px distance and line of sight constraints")
    print("")
    
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
                elif event.key == pygame.K_r:
                    # Reset/respawn agents
                    visitor, escort = respawn_agents(environment)
                    reset_message_timer = 120  # Show message for 2 seconds at 60fps
                    print("Agents respawned in new locations!")
        
        # Handle continuous key input
        keys = pygame.key.get_pressed()
        
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
            
        visitor.set_controls(visitor_linear_vel, visitor_angular_vel)
        
        # Escort controls (WASD keys)
        escort_linear_vel = 0
        escort_angular_vel = 0
        
        if keys[pygame.K_w]:
            escort_linear_vel = FOLLOWER_LINEAR_VEL_MAX
        if keys[pygame.K_s]:
            escort_linear_vel = -FOLLOWER_LINEAR_VEL_MAX
        if keys[pygame.K_a]:
            escort_angular_vel = -LEADER_ANGULAR_VEL  # Use same magnitude as visitor for consistency
        if keys[pygame.K_d]:
            escort_angular_vel = LEADER_ANGULAR_VEL   # Use same magnitude as visitor for consistency
            
        escort.set_controls(escort_linear_vel, escort_angular_vel)
        
        # Update simulation
        if not paused:
            dt = clock.get_time() / 1000.0  # Convert to seconds
            
            # Update agents
            visitor.update(dt, environment.get_all_walls(), environment.doors)
            escort.update(dt, environment.get_all_walls(), environment.doors)
            
            # Update vision for both agents
            visitor.cast_vision_rays(environment.get_all_walls(), environment.doors)
            escort.cast_vision_rays(environment.get_all_walls(), environment.doors)
            
            # Update reset message timer
            if reset_message_timer > 0:
                reset_message_timer -= 1
        
        # Draw everything
        # Draw environment
        environment.draw(screen, font)
        
        # Draw vision beams for both agents (behind everything else)
        if show_vision:
            visitor.draw_vision(screen, show_beams=True)
            escort.draw_vision(screen, show_beams=True)
        
        # Check and draw line of sight between agents
        def can_see_each_other(pos1, pos2, walls, doors):
            """Check if two agents can see each other"""
            x1, y1 = pos1
            x2, y2 = pos2
            
            steps = 50
            for i in range(steps + 1):
                t = i / steps
                x = x1 + t * (x2 - x1)
                y = y1 + t * (y2 - y1)
                
                point_rect = pygame.Rect(int(x) - 2, int(y) - 2, 4, 4)
                for wall in walls:
                    if point_rect.colliderect(wall):
                        in_door = False
                        for door in doors:
                            if point_rect.colliderect(door):
                                in_door = True
                                break
                        if not in_door:
                            return False
            return True
        
        # Draw line of sight
        visitor_pos = (visitor.state[0], visitor.state[1])
        escort_pos = (escort.state[0], escort.state[1])
        
        if can_see_each_other(visitor_pos, escort_pos, environment.get_all_walls(), environment.doors):
            # Draw green line if they can see each other
            pygame.draw.line(screen, GREEN, visitor_pos, escort_pos, 2)
            line_of_sight = True
        else:
            # Draw red dashed line if they can't see each other
            # Simple dashed line implementation
            distance = ((escort_pos[0] - visitor_pos[0])**2 + (escort_pos[1] - visitor_pos[1])**2)**0.5
            segments = int(distance / 10)
            for i in range(0, segments, 2):  # Every other segment
                t1 = i / segments
                t2 = min((i + 1) / segments, 1)
                x1 = visitor_pos[0] + t1 * (escort_pos[0] - visitor_pos[0])
                y1 = visitor_pos[1] + t1 * (escort_pos[1] - visitor_pos[1])
                x2 = visitor_pos[0] + t2 * (escort_pos[0] - visitor_pos[0])
                y2 = visitor_pos[1] + t2 * (escort_pos[1] - visitor_pos[1])
                pygame.draw.line(screen, RED, (x1, y1), (x2, y2), 2)
            line_of_sight = False
        
        # Draw agents
        visitor.draw(screen)
        escort.draw(screen)
        
        # Draw status information
        status_y = 10
        
        # Show control status for both agents
        visitor_status = f"Visitor: L={visitor_linear_vel:.1f}, A={visitor_angular_vel:.1f}"
        escort_status = f"Escort: L={escort_linear_vel:.1f}, A={escort_angular_vel:.1f}"
        sight_status = f"Line of Sight: {'YES' if line_of_sight else 'NO'}"
        
        # Calculate current distance between agents
        current_distance = ((visitor.state[0] - escort.state[0])**2 + (visitor.state[1] - escort.state[1])**2)**0.5
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
        
        # Show which keys are pressed
        pressed_keys = []
        if keys[pygame.K_UP]: pressed_keys.append("↑")
        if keys[pygame.K_DOWN]: pressed_keys.append("↓")
        if keys[pygame.K_LEFT]: pressed_keys.append("←")
        if keys[pygame.K_RIGHT]: pressed_keys.append("→")
        if keys[pygame.K_w]: pressed_keys.append("W")
        if keys[pygame.K_s]: pressed_keys.append("S")
        if keys[pygame.K_a]: pressed_keys.append("A")
        if keys[pygame.K_d]: pressed_keys.append("D")
        
        if pressed_keys:
            keys_text = f"Pressed: {', '.join(pressed_keys)}"
            keys_surface = font.render(keys_text, True, WHITE)
            screen.blit(keys_surface, (10, status_y))
            status_y += 25
        
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
            reset_text = "🔄 AGENTS RESPAWNED!"
            reset_font = pygame.font.SysFont('Arial', 24, bold=True)
            reset_surface = reset_font.render(reset_text, True, YELLOW)
            screen.blit(reset_surface, (WIDTH//2 - reset_surface.get_width()//2, 50))
        
        # Show FPS
        fps = clock.get_fps()
        fps_text = font.render(f"FPS: {fps:.1f}", True, WHITE)
        screen.blit(fps_text, (WIDTH - fps_text.get_width() - 10, HEIGHT - 25))
        
        # Show agent positions (debug info)
        visitor_pos_text = f"Visitor: ({visitor.state[0]:.0f}, {visitor.state[1]:.0f})"
        escort_pos_text = f"Escort: ({escort.state[0]:.0f}, {escort.state[1]:.0f})"
        
        visitor_pos_surface = font.render(visitor_pos_text, True, RED)
        escort_pos_surface = font.render(escort_pos_text, True, ORANGE)
        
        screen.blit(visitor_pos_surface, (10, HEIGHT - 50))
        screen.blit(escort_pos_surface, (10, HEIGHT - 25))
        
        # Update display
        pygame.display.flip()
        clock.tick(TARGET_FPS)
    
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
