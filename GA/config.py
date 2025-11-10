"""
Setting Hyperparameters for GA
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

# Delhi City Coordinates
DELHI_BOUNDS = {
    'lat_min': 28.4,
    'lat_max': 28.9,
    'lon_min': 76.8,
    'lon_max': 77.4,
    'center': [28.6139, 77.2090]
}
