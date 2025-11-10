"""
Setting Hyperparameters for Hybrid GA-SA
"""

GA_PARAMS = {
    
    'population_size': 100,
    
    'crossover_rate': 0.80,
    'mutation_rate': 0.05,
    'tournament_size': 3,
    
    'max_iterations': 100,
    'early_stopping_patience': 20,
    
    # p-median values
    'num_stations': [10, 20, 30],
    
    'selection_method': 'tournament'
}

SA_PARAMS = {
    'initial_temp': 1000.0,
    'cooling_rate': 0.95,
    'min_temp': 1.0,
    'max_iter_per_temp': 10,
    'apply_sa_prob': 0.2  # Probability to apply SA to each individual in new population
}

# Delhi City Coordinates
DELHI_BOUNDS = {
    'lat_min': 28.4,
    'lat_max': 28.9,
    'lon_min': 76.8,
    'lon_max': 77.4,
    'center': [28.6139, 77.2090]
}
