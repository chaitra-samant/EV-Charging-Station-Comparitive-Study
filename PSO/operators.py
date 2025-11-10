"""
Particle Swarm Optimization operators: initialization, chromosome derivation
"""

import numpy as np

def initialize_swarm(n_candidates, swarm_size):
    """
    Initialize swarm positions and velocities.
    
    Args:
        n_candidates: Total number of candidate locations
        swarm_size: Number of particles
    
    Returns:
        positions: (swarm_size, n_candidates) array in [0,1]
        velocities: (swarm_size, n_candidates) array
    """
    positions = np.random.uniform(0, 1, (swarm_size, n_candidates))
    velocities = np.random.uniform(-0.1, 0.1, (swarm_size, n_candidates))
    return positions, velocities

def get_chromosome(position, p):
    """
    Derive binary chromosome from continuous position.
    Selects top p positions with highest values.
    
    Args:
        position: (n_candidates,) array in [0,1]
        p: Number of stations to select
    
    Returns:
        chromosome: Binary array with exactly p ones
    """
    indices = np.argsort(position)[-p:]  # Top p indices
    chromosome = np.zeros(len(position), dtype=int)
    chromosome[indices] = 1
    return chromosome
