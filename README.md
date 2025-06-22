# Simple Two-Agent Simulation - Standalone

This is a standalone version of the simple two-agent simulation extracted from the PivotedTracking project.

## Description

A simple simulation featuring two agents in an indoor environment:
- **Visitor Agent (Red)**: Controlled with arrow keys
- **Escort Agent (Orange)**: Controlled with WASD keys

## Features

- Indoor environment with walls, doors, and windows
- 360-degree vision system for both agents
- Line-of-sight detection between agents
- Collision detection with walls
- Intelligent spawn positioning
- Real-time controls and visualization

## Requirements

- Python 3.8+
- pygame>=2.1.0
- numpy>=1.21.0

## Installation

1. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

Run the simulation:
```bash
python run_simple_simulation.py
```

## Controls

### Visitor Agent (Red):
- ↑ : Move Forward
- ↓ : Move Backward
- ← : Turn Left
- → : Turn Right

### Escort Agent (Orange):
- W : Move Forward
- S : Move Backward
- A : Turn Left
- D : Turn Right

### Other Controls:
- SPACE : Pause/Unpause
- V : Toggle Vision Beams
- R : Reset/Respawn Agents in New Locations
- ESC : Quit

## Visual Features

- 🟢 **Green Line**: Agents can see each other
- 🔴 **Red Dashed Line**: Agents cannot see each other
- **Vision Beams**: 100 rays per agent, 360° coverage
  - Red to Green: Close to far obstacles
  - Blue: No obstacle in range
  - Range: 800 pixels
- Line of sight is blocked by walls but not doors

## File Structure

```
standalone_simulation/
├── run_simple_simulation.py          # Main launcher
├── simple_two_agent_simulation.py    # Core simulation logic
├── requirements.txt                  # Python dependencies
├── README.md                        # This file
└── multitrack/                      # Support modules
    ├── __init__.py
    ├── models/
    │   ├── __init__.py
    │   └── simulation_environment.py  # Environment management
    └── utils/
        ├── __init__.py
        └── config.py                 # Configuration constants
```

## License

This standalone version maintains the licensing of the original PivotedTracking project.
