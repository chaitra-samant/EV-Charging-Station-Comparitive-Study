"""
Fitness function for p-Median EV Charging Station Location Problem
"""

import numpy as np
from sklearn.metrics.pairwise import haversine_distances

def calculate_distance_matrix(coords1, coords2):
    """
    Calculate haversine distances between two sets of coordinates.
    
    Args:
        coords1: (n, 2) array of [lat, lon] in degrees
        coords2: (m, 2) array of [lat, lon] in degrees
    
    Returns:
        distance_matrix: (n, m) array of distances in kilometers
    """
    # Convert to radians
    coords1_rad = np.radians(coords1)
    coords2_rad = np.radians(coords2)
    
    # Haversine formula (Earth radius = 6371 km)
    distances = haversine_distances(coords1_rad, coords2_rad) * 6371
    return distances

def fitness_pmedian(chromosome, distance_matrix, demand_weights, p):
    """
    Calculate fitness for p-median problem.
    
    Objective: Minimize total weighted distance from demand points to nearest station
    Formula: Σᵢ wᵢ × min_j(dᵢⱼ) where station j is selected
    
    Args:
        chromosome: Binary array [0,1,0,1,...] where 1 = station selected
        distance_matrix: (n_demand, n_candidates) distances in km
        demand_weights: (n_demand,) array of demand at each point
        p: Number of stations to select
    
    Returns:
        fitness: Total cost (to be minimised). Returns inf if invalid.
    """
    selected_stations = np.where(chromosome == 1)[0]
    
    # Constraint: Must have exactly p stations
    if len(selected_stations) != p:
        return float('inf')
    
    # For each demand point, find distance to nearest selected station
    distances_to_selected = distance_matrix[:, selected_stations]
    min_distances = distances_to_selected.min(axis=1)
    
    # Total weighted cost
    total_cost = np.sum(demand_weights * min_distances)
    
    return total_cost

def calculate_coverage_metrics(chromosome, distance_matrix, demand_weights, p, threshold_km=5.0):
    """
    Calculate additional metrics for analysis.
    
    Returns:
        dict with metrics: avg_distance, max_distance, coverage_percentage
    """
    selected_stations = np.where(chromosome == 1)[0]
    
    if len(selected_stations) != p:
        return None
    
    distances_to_selected = distance_matrix[:, selected_stations]
    min_distances = distances_to_selected.min(axis=1)
    
    # Weighted average distance
    avg_distance = np.sum(demand_weights * min_distances) / demand_weights.sum()
    
    # Maximum distance any demand point has to travel
    max_distance = min_distances.max()
    
    # Coverage: % of demand within threshold distance
    within_threshold = min_distances <= threshold_km
    coverage = (demand_weights[within_threshold].sum() / demand_weights.sum()) * 100
    
    return {
        'avg_distance_km': avg_distance,
        'max_distance_km': max_distance,
        'coverage_percent': coverage,
        'num_stations': p
    }
