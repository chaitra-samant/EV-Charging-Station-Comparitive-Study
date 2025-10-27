"""
Genetic Algorithm operators: initialization, selection, crossover, mutation
And Simulated Annealing for hybrid
"""

import numpy as np
from fitness import fitness_pmedian

def initialize_population(n_candidates, p, pop_size):
    """
    Create initial population of valid chromosomes.
    Each chromosome has exactly p ones (selected stations).
    
    Args:
        n_candidates: Total number of candidate locations
        p: Number of stations to select
        pop_size: Population size
    
    Returns:
        population: (pop_size, n_candidates) binary array
    """
    population = []
    for _ in range(pop_size):
        chromosome = np.zeros(n_candidates, dtype=int)
        selected_indices = np.random.choice(n_candidates, p, replace=False)
        chromosome[selected_indices] = 1
        population.append(chromosome)
    return np.array(population)

# Paper recommends Tournament Selection
def tournament_selection(population, fitness_scores, tournament_size=3):
    """
    Args:
        population: Array of chromosomes
        fitness_scores: Fitness for each chromosome
        tournament_size: Number of individuals in tournament
    
    Returns:
        selected_chromosome: Winner of tournament
    """
    # Randomly pick tournament_size individuals
    indices = np.random.choice(len(population), tournament_size, replace=False)
    tournament_fitness = fitness_scores[indices]
    
    # Select best (minimum fitness for minimization)
    winner_idx = indices[np.argmin(tournament_fitness)]
    return population[winner_idx].copy()

def roulette_wheel_selection(population, fitness_scores):
    """
    Roulette wheel selection (fitness proportionate).
    For minimization, invert fitness scores.
    """
    # Convert to maximization (inverse fitness)
    # Add small epsilon to avoid division by zero
    max_fitness = fitness_scores.max() + 1
    inverted_fitness = max_fitness - fitness_scores
    
    # Normalize to probabilities
    probabilities = inverted_fitness / inverted_fitness.sum()
    
    # Select based on probabilities
    idx = np.random.choice(len(population), p=probabilities)
    return population[idx].copy()

def single_point_crossover(parent1, parent2, p):
    """
    Single-point crossover with repair mechanism.
    
    Args:
        parent1, parent2: Binary chromosomes
        p: Required number of stations
    
    Returns:
        child: Valid chromosome with exactly p stations
    """
    point = np.random.randint(1, len(parent1))
    child = np.concatenate([parent1[:point], parent2[point:]])
    
    # Repair to ensure exactly p stations
    child = repair_chromosome(child, p)
    return child

def uniform_crossover(parent1, parent2, p, crossover_rate=0.5):
    """
    Uniform crossover: Each gene has 50% chance from each parent.
    """
    mask = np.random.rand(len(parent1)) < crossover_rate
    child = np.where(mask, parent1, parent2)
    
    # Repair
    child = repair_chromosome(child, p)
    return child

def swap_mutation(chromosome, p, mutation_rate):
    """
    Swap mutation: Flip one 0 to 1 and one 1 to 0.
    Maintains exactly p stations.
    
    Args:
        chromosome: Binary chromosome
        p: Required number of stations
        mutation_rate: Probability of mutation
    
    Returns:
        mutated_chromosome
    """
    if np.random.rand() < mutation_rate:
        ones_idx = np.where(chromosome == 1)[0]
        zeros_idx = np.where(chromosome == 0)[0]
        
        if len(ones_idx) > 0 and len(zeros_idx) > 0:
            # Swap one selected with one unselected
            chromosome[np.random.choice(ones_idx)] = 0
            chromosome[np.random.choice(zeros_idx)] = 1
    
    return chromosome

def repair_chromosome(chromosome, p):
    """
    Repair chromosome to have exactly p stations.
    
    Strategy:
    - If too many stations: randomly remove excess
    - If too few: randomly add missing ones
    """
    current_stations = np.sum(chromosome)
    
    if current_stations > p:
        # Remove excess stations
        ones_idx = np.where(chromosome == 1)[0]
        remove_idx = np.random.choice(ones_idx, current_stations - p, replace=False)
        chromosome[remove_idx] = 0
    
    elif current_stations < p:
        # Add missing stations
        zeros_idx = np.where(chromosome == 0)[0]
        add_idx = np.random.choice(zeros_idx, p - current_stations, replace=False)
        chromosome[add_idx] = 1
    
    return chromosome

def simulated_annealing(chromosome, distance_matrix, demand_weights, p, initial_temp, cooling_rate, min_temp, max_iter_per_temp):
    """
    Simulated Annealing local search for refining a chromosome.
    
    Args:
        chromosome: Starting binary chromosome
        distance_matrix: Precomputed distances
        demand_weights: Demand weights
        p: Number of stations
        initial_temp: Starting temperature
        cooling_rate: Cooling factor (0-1)
        min_temp: Stopping temperature
        max_iter_per_temp: Iterations per temperature level
    
    Returns:
        refined_chromosome: Best found solution
    """
    current = chromosome.copy()
    current_fitness = fitness_pmedian(current, distance_matrix, demand_weights, p)
    
    best = current.copy()
    best_fitness = current_fitness
    
    temp = initial_temp
    
    while temp > min_temp:
        for _ in range(max_iter_per_temp):
            # Generate neighbor via swap
            neighbor = swap_mutation(current.copy(), p, 1.0)  # Force one swap
            neighbor_fitness = fitness_pmedian(neighbor, distance_matrix, demand_weights, p)
            
            if neighbor_fitness == float('inf'):
                continue  # Skip invalid
            
            delta = neighbor_fitness - current_fitness
            
            if delta < 0 or np.random.rand() < np.exp(-delta / temp):
                current = neighbor
                current_fitness = neighbor_fitness
                
                if current_fitness < best_fitness:
                    best = current.copy()
                    best_fitness = current_fitness
        
        temp *= cooling_rate
    
    return best
