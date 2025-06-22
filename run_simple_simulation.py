#!/usr/bin/env python3
"""
Simple launcher for the two-agent simulation.
This provides a convenient way to start the simulation.
"""
import os
import sys

# Make sure the project directory is in the path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Import and run the simulation
from simple_two_agent_simulation import main

if __name__ == "__main__":
    print("🎮 Starting Simple Two-Agent Simulation...")
    print("=" * 50)
    main()
