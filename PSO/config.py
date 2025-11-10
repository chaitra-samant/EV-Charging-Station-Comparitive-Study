"""
Setting Hyperparameters for PSO
"""

PSO_PARAMS = {
    
    'swarm_size': 100,
    
    'inertia_weight': 0.8,
    'cognitive_coeff': 1.5,
    'social_coeff': 1.5,
    
    'max_iterations': 100,
    'early_stopping_patience': 20,
    
    # p-median values
    'num_stations': [10, 20, 30],
    
}

# Delhi City Coordinates
DELHI_BOUNDS = {
    'lat_min': 28.4,
    'lat_max': 28.9,
    'lon_min': 76.8,
    'lon_max': 77.4,
    'center': [28.6139, 77.2090]
}
