import mlrose
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import mlrose
import numpy as np

# Define the optimization problem - Traveling Salesperson (TSP)
tsp_coords = np.array([[0, 0], [1, 2], [2, 4], [3, 1]])
fitness_tsp = mlrose.TravellingSales(coords=tsp_coords)
problem_tsp = mlrose.TSPOpt(length=len(tsp_coords), fitness_fn=fitness_tsp, maximize=True)

# Define the optimization problem - N-Queens (NQP)
fitness_nqp = mlrose.Queens()
problem_nqp = mlrose.DiscreteOpt(length=8, fitness_fn=fitness_nqp, maximize=True, max_val=2)

# Define the optimization problem - Graph Coloring (GCP)
edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
fitness_gcp = mlrose.MaxKColor(edges)
problem_gcp = mlrose.DiscreteOpt(length=4, fitness_fn=fitness_gcp, maximize=True, max_val=2)

# Define the algorithms
algorithms_tsp = [
    ('Random Hill Climbing', mlrose.random_hill_climb(problem=problem_tsp, max_attempts=10, max_iters=1000, restarts=5, curve= True, random_state=None)),
    ('Simulated Annealing', mlrose.simulated_annealing(problem=problem_tsp, schedule=mlrose.GeomDecay(init_temp=1.0, decay=0.99), max_attempts=10, max_iters=1000, curve= True, random_state=None)),
    ('Genetic Algorithm', mlrose.genetic_alg(problem=problem_tsp, pop_size=200, mutation_prob=0.1, max_attempts=10, max_iters=1000, curve= True, random_state=None)),
    ('MIMIC', mlrose.mimic(problem=problem_tsp, pop_size=200, keep_pct=0.2, max_attempts=10, max_iters=1000, curve= True, random_state=None))
]

algorithms_nqp = [
    ('Random Hill Climbing', mlrose.random_hill_climb(problem=problem_nqp, max_attempts=10, max_iters=1000, restarts=5, curve= True, random_state=None)),
    ('Simulated Annealing', mlrose.simulated_annealing(problem=problem_nqp, schedule=mlrose.GeomDecay(init_temp=1.0, decay=0.99), max_attempts=10, max_iters=1000, curve= True, random_state=None)),
    ('Genetic Algorithm', mlrose.genetic_alg(problem=problem_nqp, pop_size=200, mutation_prob=0.1, max_attempts=10, max_iters=1000, curve= True, random_state=None)),
    ('MIMIC', mlrose.mimic(problem=problem_nqp, pop_size=200, keep_pct=0.2, max_attempts=10, max_iters=1000, curve= True, random_state=None))
]

algorithms_gcp = [
    ('Random Hill Climbing', mlrose.random_hill_climb(problem=problem_gcp, max_attempts=10, max_iters=1000, restarts=5, curve= True, random_state=None)),
    ('Simulated Annealing', mlrose.simulated_annealing(problem=problem_gcp, schedule=mlrose.GeomDecay(init_temp=1.0, decay=0.99), max_attempts=10, max_iters=1000, curve= True, random_state=None)),
    ('Genetic Algorithm', mlrose.genetic_alg(problem=problem_gcp, pop_size=200, mutation_prob=0.1, max_attempts=10, max_iters=1000, curve= True, random_state=None)),
    ('MIMIC', mlrose.mimic(problem=problem_gcp, pop_size=200, keep_pct=0.2, max_attempts=10, max_iters=1000, curve= True, random_state=None))
]