"""
Main Genetic Algorithm implementation for EV Charging Station Location
"""

import numpy as np
import pandas as pd
import time
from fitness import calculate_distance_matrix, fitness_pmedian, calculate_coverage_metrics
from operators import *
from config import GA_PARAMS
import os
import warnings
warnings.filterwarnings('ignore')

def genetic_algorithm(candidate_coords, demand_coords, demand_weights, p, verbose=True):
    """
    Main GA loop
    
    Args:
        candidate_coords: (n_candidates, 2) array [lat, lon]
        demand_coords: (n_demand, 2) array [lat, lon]
        demand_weights: (n_demand,) demand at each point
        p: Number of stations to select
        verbose: Print progress
    
    Returns:
        best_solution: Binary array of selected stations
        best_cost: Final fitness value
        convergence_history: List of best costs per generation
        metrics: Dictionary of performance metrics
    """
    # Parameters
    pop_size = GA_PARAMS['population_size']
    crossover_rate = GA_PARAMS['crossover_rate']
    mutation_rate = GA_PARAMS['mutation_rate']
    max_iter = GA_PARAMS['max_iterations']
    tournament_size = GA_PARAMS['tournament_size']
    
    n_candidates = len(candidate_coords)
    
    # Calculate distance matrix 
    if verbose:
        print("Calculating distance matrix...")
    distance_matrix = calculate_distance_matrix(demand_coords, candidate_coords)
    
    # Initialize population
    if verbose:
        print(f"Initializing population (size={pop_size})...")
    population = initialize_population(n_candidates, p, pop_size)
    
    # Tracking cost
    convergence_history = []
    best_cost = float('inf')
    best_solution = None
    no_improvement_count = 0
    
    # Evolution loop
    start_time = time.time()
    
    for generation in range(max_iter):
        # Evaluate fitness for all individuals
        fitness_scores = np.array([
            fitness_pmedian(chrom, distance_matrix, demand_weights, p) 
            for chrom in population
        ])
        
        # Track best solution
        gen_best_idx = np.argmin(fitness_scores)
        gen_best_cost = fitness_scores[gen_best_idx]
        
        if gen_best_cost < best_cost:
            best_cost = gen_best_cost
            best_solution = population[gen_best_idx].copy()
            no_improvement_count = 0
        else:
            no_improvement_count += 1
        
        convergence_history.append(best_cost)
        
        # Print progress
        if verbose and generation % 10 == 0:
            avg_cost = np.mean(fitness_scores[fitness_scores != float('inf')])
            elapsed = time.time() - start_time
            print(f"Gen {generation:3d}: Best={best_cost:,.2f} km  Avg={avg_cost:,.2f} km  Time={elapsed:.1f}s")
        
        # Early stopping
        patience = GA_PARAMS.get('early_stopping_patience', max_iter)
        if no_improvement_count >= patience:
            if verbose:
                print(f"Early stopping at generation {generation} (no improvement for {no_improvement_count} generations)")
            break
        
        # Create next generation
        new_population = []
        
        for _ in range(pop_size):
            # Tournament Selection 
            parent1 = tournament_selection(population, fitness_scores, tournament_size)
            parent2 = tournament_selection(population, fitness_scores, tournament_size)
            
            # Crossover
            if np.random.rand() < crossover_rate:
                child = single_point_crossover(parent1, parent2, p)
            else:
                child = parent1.copy()
            
            # Mutation
            child = swap_mutation(child, p, mutation_rate)
            
            new_population.append(child)
        
        population = np.array(new_population)
    
    elapsed = time.time() - start_time
    
    # Calculate final metrics
    metrics = calculate_coverage_metrics(best_solution, distance_matrix, demand_weights, p)
    metrics['total_cost_km'] = best_cost
    metrics['time_seconds'] = elapsed
    metrics['generations'] = len(convergence_history)
    
    if verbose:
        print(f"\n\n\n")
        print(f"Optimization Complete")
        print(f"Stations Selected: {p}")
        print(f"Total Cost: {best_cost:,.2f} km")
        print(f"Avg Distance: {metrics['avg_distance_km']:.3f} km per EV")
        print(f"Max Distance: {metrics['max_distance_km']:.2f} km")
        print(f"Coverage (≤5km): {metrics['coverage_percent']:.1f}%")
        print(f"Time: {elapsed:.2f}s")
        print(f"Generations: {len(convergence_history)}")
    
    return best_solution, best_cost, convergence_history, metrics


# ===== MAIN EXECUTION =====
if __name__ == "__main__":
    print("\n\n\n")
    print("EV Charging Station Optimization - Delhi")
    print("Genetic Algorithm Implementation")
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
    print(f"  Testing p values: {GA_PARAMS['num_stations']}")
    
    # Run GA for multiple p values
    print(f"\n\n\n")
    print("Starting Optimization Runs")
    print(f"{'='*60}")
    
    results = []
    
    for p in GA_PARAMS['num_stations']:
        print(f"\n{'='*60}")
        print(f"Running GA for p={p} stations")
        print(f"{'='*60}")
        
        best_sol, best_cost, history, metrics = genetic_algorithm(
            candidate_coords, demand_coords, demand_weights, p, verbose=True
        )
        
        results.append(metrics)
        
        # Save convergence results
        conv_df = pd.DataFrame({
            'generation': range(len(history)),
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
    results_df.to_csv('results/ga_results.csv', index=False)
    
    
    print("\n Summary Results:")
    print("─"*60)
    summary_cols = ['num_stations', 'total_cost_km', 'avg_distance_km', 
                    'coverage_percent', 'time_seconds', 'generations']
    print(results_df[summary_cols].to_string(index=False))
