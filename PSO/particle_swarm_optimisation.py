"""
Main Particle Swarm Optimization implementation for EV Charging Station Location
"""

import numpy as np
import pandas as pd
import time
from fitness import calculate_distance_matrix, fitness_pmedian, calculate_coverage_metrics
from operators import *
from config import PSO_PARAMS
import os
import warnings
warnings.filterwarnings('ignore')

def particle_swarm_optimization(candidate_coords, demand_coords, demand_weights, p, verbose=True):
    """
    Main PSO loop
    
    Args:
        candidate_coords: (n_candidates, 2) array [lat, lon]
        demand_coords: (n_demand, 2) array [lat, lon]
        demand_weights: (n_demand,) demand at each point
        p: Number of stations to select
        verbose: Print progress
    
    Returns:
        best_solution: Binary array of selected stations
        best_cost: Final fitness value
        convergence_history: List of best costs per iteration
        metrics: Dictionary of performance metrics
    """
    # Parameters
    swarm_size = PSO_PARAMS['swarm_size']
    w = PSO_PARAMS['inertia_weight']
    c1 = PSO_PARAMS['cognitive_coeff']
    c2 = PSO_PARAMS['social_coeff']
    max_iter = PSO_PARAMS['max_iterations']
    
    n_candidates = len(candidate_coords)
    
    # Calculate distance matrix 
    if verbose:
        print("Calculating distance matrix...")
    distance_matrix = calculate_distance_matrix(demand_coords, candidate_coords)
    
    # Initialize swarm
    if verbose:
        print(f"Initializing swarm (size={swarm_size})...")
    positions, velocities = initialize_swarm(n_candidates, swarm_size)
    
    # Evaluate initial fitness
    chromosomes = [get_chromosome(pos, p) for pos in positions]
    fitness_scores = np.array([
        fitness_pmedian(chrom, distance_matrix, demand_weights, p) 
        for chrom in chromosomes
    ])
    
    # Personal bests
    pbest_positions = positions.copy()
    pbest_scores = fitness_scores.copy()
    
    # Global best
    gbest_idx = np.argmin(pbest_scores)
    gbest_position = pbest_positions[gbest_idx].copy()
    gbest_score = pbest_scores[gbest_idx]
    
    # Tracking
    convergence_history = [gbest_score]
    no_improvement_count = 0
    
    # Optimization loop
    start_time = time.time()
    
    for iteration in range(1, max_iter + 1):
        # Update each particle
        for i in range(swarm_size):
            # Update velocity
            r1 = np.random.rand(n_candidates)
            r2 = np.random.rand(n_candidates)
            velocities[i] = (w * velocities[i] + 
                             c1 * r1 * (pbest_positions[i] - positions[i]) + 
                             c2 * r2 * (gbest_position - positions[i]))
            
            # Clamp velocity (optional, but helps stability)
            np.clip(velocities[i], -1.0, 1.0, out=velocities[i])
            
            # Update position
            positions[i] += velocities[i]
            np.clip(positions[i], 0.0, 1.0, out=positions[i])
            
            # Get chromosome and fitness
            chrom = get_chromosome(positions[i], p)
            score = fitness_pmedian(chrom, distance_matrix, demand_weights, p)
            
            # Update personal best
            if score < pbest_scores[i]:
                pbest_scores[i] = score
                pbest_positions[i] = positions[i].copy()
                
                # Update global best
                if score < gbest_score:
                    gbest_score = score
                    gbest_position = positions[i].copy()
                    no_improvement_count = 0
        
        # Track convergence
        convergence_history.append(gbest_score)
        
        # Check improvement
        if convergence_history[-1] >= convergence_history[-2]:
            no_improvement_count += 1
        else:
            no_improvement_count = 0
        
        # Print progress
        if verbose and iteration % 10 == 0:
            avg_cost = np.mean(pbest_scores[pbest_scores != float('inf')])
            elapsed = time.time() - start_time
            print(f"Iter {iteration:3d}: Best={gbest_score:,.2f} km  Avg={avg_cost:,.2f} km  Time={elapsed:.1f}s")
        
        # Early stopping
        patience = PSO_PARAMS.get('early_stopping_patience', max_iter)
        if no_improvement_count >= patience:
            if verbose:
                print(f"Early stopping at iteration {iteration} (no improvement for {no_improvement_count} iterations)")
            break
    
    elapsed = time.time() - start_time
    
    # Get best solution
    best_solution = get_chromosome(gbest_position, p)
    
    # Calculate final metrics
    metrics = calculate_coverage_metrics(best_solution, distance_matrix, demand_weights, p)
    metrics['total_cost_km'] = gbest_score
    metrics['time_seconds'] = elapsed
    metrics['generations'] = len(convergence_history)  # Note: using 'generations' for compatibility with viz
    
    if verbose:
        print(f"\n\n\n")
        print(f"Optimization Complete")
        print(f"Stations Selected: {p}")
        print(f"Total Cost: {gbest_score:,.2f} km")
        print(f"Avg Distance: {metrics['avg_distance_km']:.3f} km per EV")
        print(f"Max Distance: {metrics['max_distance_km']:.2f} km")
        print(f"Coverage (≤5km): {metrics['coverage_percent']:.1f}%")
        print(f"Time: {elapsed:.2f}s")
        print(f"Iterations: {len(convergence_history)}")
    
    return best_solution, gbest_score, convergence_history, metrics


# ===== MAIN EXECUTION =====
if __name__ == "__main__":
    print("\n\n\n")
    print("EV Charging Station Optimization - Delhi")
    print("Particle Swarm Optimization Implementation")
    print("\n\n\n")
    
    # Results Directory
    os.makedirs('results/data', exist_ok=True)
    os.makedirs('results/plots', exist_ok=True)
    os.makedirs('results/maps', exist_ok=True)
    
    # Load Delhi dataset
    df = pd.read_csv('../dataset/delhi_ev_charging.csv')
    
    print(f"\nDataset loaded: {len(df)} charging stations")
    print(f"Coordinate range:")
    print(f"  Latitude:  {df['lattitude'].min():.4f} → {df['lattitude'].max():.4f}")
    print(f"  Longitude: {df['longitude'].min():.4f} → {df['longitude'].max():.4f}")
    
    # Setup problem
    candidate_coords = df[['lattitude', 'longitude']].values
    demand_coords = candidate_coords.copy()  # Same points for simplified model
    demand_weights = np.ones(len(df)) * 50  # Assume 50 EVs per location
    
    print(f"\nProblem Setup:")
    print(f"  Candidate locations: {len(candidate_coords)}")
    print(f"  Demand points: {len(demand_coords)}")
    print(f"  Total demand: {demand_weights.sum():.0f} EVs")
    print(f"  Testing p values: {PSO_PARAMS['num_stations']}")
    
    # Run PSO for multiple p values
    print(f"\n\n\n")
    print("Starting Optimization Runs")
    print(f"{'='*60}")
    
    results = []
    
    for p in PSO_PARAMS['num_stations']:
        print(f"\n{'='*60}")
        print(f"Running PSO for p={p} stations")
        print(f"{'='*60}")
        
        best_sol, best_cost, history, metrics = particle_swarm_optimization(
            candidate_coords, demand_coords, demand_weights, p, verbose=True
        )
        
        results.append(metrics)
        
        # Save convergence results
        conv_df = pd.DataFrame({
            'generation': range(len(history)),  # Note: 'generation' for compatibility
            'best_cost': history
        })
        conv_df.to_csv(f'results/data/convergence_p{p}.csv', index=False)
        print(f"Saved: results/data/convergence_p{p}.csv")
        
        # Save selected station details
        selected_indices = np.where(best_sol == 1)[0]
        selected_stations = df.iloc[selected_indices].copy()
        selected_stations.to_csv(f'results/data/selected_stations_p{p}.csv', index=False)
        print(f"Saved: results/data/selected_stations_p{p}.csv")
    
    # Save summary results
    results_df = pd.DataFrame(results)
    results_df.to_csv('results/pso_results.csv', index=False)
    
    
    print("\n Summary Results:")
    print("─"*60)
    summary_cols = ['num_stations', 'total_cost_km', 'avg_distance_km', 
                    'coverage_percent', 'time_seconds', 'generations']
    print(results_df[summary_cols].to_string(index=False))
