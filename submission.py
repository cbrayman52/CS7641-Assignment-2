import mlrose_hiive
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score

import tsp
import nqp
import gcp
import run_algorithms

###############################################################################
######################### Traveling Salesperson (TSP) #########################
###############################################################################
# Define the number of cities and area of the city
num_cities = 20
area = 20

# Define an optimization problem using TSPGenerator
problem_tsp = mlrose_hiive.TSPGenerator.generate(seed=None, number_of_cities=num_cities, area_width=area, area_height=area)
problem_tsp.maximize = -1.0

# Display the city coordinates
tsp.display_city_coordinates(problem_tsp.coords, area)

# Perform hyperparameter tuning
run_algorithms.hyperparameter_tuning(problem_tsp, 'TSP')

# Define optimization algorithms with optimized parameters
algorithms = {
    'Random_Hill_Climbing': {
        'runner': mlrose_hiive.RHCRunner,
        'params': {
            'experiment_name'   : 'RHC',
            'seed'              : None,
            'iteration_list'    : [1000],
            'restart_list'      : [5],
            'max_attempts'      : 20,
            'generate_curves'   : True
        }
    },
    'Simulated_Annealing': {
        'runner': mlrose_hiive.SARunner,
        'params': {
            'experiment_name'   : 'SA',
            'seed'              : None,
            'iteration_list'    : [1000],
            'temperature_list'  : [100],
            'decay_list'        : [mlrose_hiive.GeomDecay],
            'max_attempts'      : 10,
            'generate_curves'   : True
        }
    },
    'Genetic_Algorithm': {
        'runner': mlrose_hiive.GARunner,
        'params': {
            'experiment_name'   : 'GA',
            'seed'              : None,
            'iteration_list'    : [1000],
            'population_sizes'  : [200],
            'mutation_rates'    : [0.2],
            'hamming_factors'   : None,
            'hamming_factors_decays' : None,
            'max_attempts'      : 10,
            'generate_curves'   : True
        }
    },
    'MIMIC': {
        'runner': mlrose_hiive.MIMICRunner,
        'params': {
            'experiment_name'   : 'MIMIC',
            'seed'              : None,
            'iteration_list'    : [1000],
            'population_sizes'  : [200],
            'keep_percent_list' : [0.5],
            'max_attempts'      : 10,
            'generate_curves'   : True
        }
    }
}
results = run_algorithms.run_algorithms(problem_tsp, algorithms)

# Display final results
tsp.display_results(results, problem_tsp.coords, area)

###############################################################################
############################### N-Queens (NQP) ################################
###############################################################################
# Define the number of queens
num_queens = 8

# Define an optimization problem using QueensGenerator
problem_nqp = mlrose_hiive.QueensGenerator.generate(seed=None, size=num_queens)
problem_nqp.maximize = -1.0

# Initialize the state (queens' positions)
init_state = np.random.permutation(num_queens)

# Plot the initial position
nqp.plot_board(init_state, num_queens, 'Initial Position of N-Queens', int(mlrose_hiive.Queens().evaluate(init_state)))
plt.savefig('Images/NQP/NQP_Initial_State.png')
plt.close()

# Perform hyperparameter tuning
run_algorithms.hyperparameter_tuning(problem_nqp, 'NQP')

# Define optimization algorithms with optimized parameters
algorithms = {
    'Random_Hill_Climbing': {
        'runner': mlrose_hiive.RHCRunner,
        'params': {
            'experiment_name'   : 'RHC',
            'seed'              : None,
            'iteration_list'    : [1000],
            'restart_list'      : [5],
            'max_attempts'      : 10,
            'generate_curves'   : True,
            'init_state'        : init_state
        }
    },
    'Simulated_Annealing': {
        'runner': mlrose_hiive.SARunner,
        'params': {
            'experiment_name'   : 'SA',
            'seed'              : None,
            'iteration_list'    : [1000],
            'temperature_list'  : [100],
            'decay_list'        : [mlrose_hiive.GeomDecay],
            'max_attempts'      : 10,
            'generate_curves'   : True,
            'init_state'        : init_state
        }
    },
    'Genetic_Algorithm': {
        'runner': mlrose_hiive.GARunner,
        'params': {
            'experiment_name'   : 'GA',
            'seed'              : None,
            'iteration_list'    : [1000],
            'population_sizes'  : [200],
            'mutation_rates'    : [0.2],
            'hamming_factors'   : None,
            'hamming_factors_decays' : None,
            'max_attempts'      : 10,
            'generate_curves'   : True
        }
    },
    'MIMIC': {
        'runner': mlrose_hiive.MIMICRunner,
        'params': {
            'experiment_name'   : 'MIMIC',
            'seed'              : None,
            'iteration_list'    : [1000],
            'population_sizes'  : [200],
            'keep_percent_list' : [0.5],
            'max_attempts'      : 10,
            'generate_curves'   : True
        }
    }
}
results = run_algorithms.run_algorithms(problem_nqp, algorithms)

# Show final results
nqp.display_results(results, num_queens)

###############################################################################
############################ Graph Coloring (GCP) #############################
###############################################################################
# Number of nodes in graph
num_nodes = 12

# Define an optimization problem using MaxKColorGenerator
problem_gcp = mlrose_hiive.MaxKColorGenerator.generate(seed=None, number_of_nodes=num_nodes, max_connections_per_node=4, max_colors=None)
problem_gcp.maximize = 1.0

# Display Graph
gcp.display_graph(problem_gcp)

# Perform hyperparameter tuning
run_algorithms.hyperparameter_tuning(problem_gcp, 'GCP')

# Define optimization algorithms with optimized parameters
algorithms = {
    'Random_Hill_Climbing': {
        'runner': mlrose_hiive.RHCRunner,
        'params': {
            'experiment_name'   : 'RHC',
            'seed'              : None,
            'iteration_list'    : [1000],
            'restart_list'      : [5],
            'max_attempts'      : 10,
            'generate_curves'   : True
        }
    },
    'Simulated_Annealing': {
        'runner': mlrose_hiive.SARunner,
        'params': {
            'experiment_name'   : 'SA',
            'seed'              : None,
            'iteration_list'    : [1000],
            'temperature_list'  : [100],
            'decay_list'        : [mlrose_hiive.GeomDecay],
            'max_attempts'      : 10,
            'generate_curves'   : True
        }
    },
    'Genetic_Algorithm': {
        'runner': mlrose_hiive.GARunner,
        'params': {
            'experiment_name'   : 'GA',
            'seed'              : None,
            'iteration_list'    : [1000],
            'population_sizes'  : [200],
            'mutation_rates'    : [0.2],
            'hamming_factors'   : None,
            'hamming_factors_decays' : None,
            'max_attempts'      : 10,
            'generate_curves'   : True
        }
    },
    'MIMIC': {
        'runner': mlrose_hiive.MIMICRunner,
        'params': {
            'experiment_name'   : 'MIMIC',
            'seed'              : None,
            'iteration_list'    : [1000],
            'population_sizes'  : [200],
            'keep_percent_list' : [0.5],
            'max_attempts'      : 10,
            'generate_curves'   : True
        }
    }
}
results = run_algorithms.run_algorithms(problem_gcp, algorithms)

# Show final results
gcp.display_results(results, problem_gcp.source_graph, num_nodes)

###############################################################################
############################ Neural Networks (NN) #############################
###############################################################################
# Download dataset
wine_quality = pd.read_csv(r'Dataset/wine_quality.csv')

# Clean data
wine_quality = wine_quality.drop_duplicates()   # Remove rows that contain duplicate data
wine_quality = wine_quality.dropna()            # Remove rows that contain missing values

# Manually handle outliers
lower_limits = {'fixed acidity': 4.0,  'volatile acidity': 0.1, 'citric acid': 0.0, 'residual sugar': 0.0,  'chlorides': 0.0, 'free sulfur dioxide': 0,  'total sulfur dioxide': 0,   'density': 0.9, 'ph': 3.0, 'sulphates': 0.3, 'alcohol': 8.0}
upper_limits = {'fixed acidity': 15.0, 'volatile acidity': 1.1, 'citric acid': 0.8, 'residual sugar': 10.0, 'chlorides': 0.3, 'free sulfur dioxide': 60, 'total sulfur dioxide': 170, 'density': 1.1, 'ph': 4.0, 'sulphates': 1.5, 'alcohol': 13.6}
for column in wine_quality.columns:
        lower_limit = lower_limits.get(column, None)
        upper_limit = upper_limits.get(column, None)        
        if lower_limit is not None and upper_limit is not None:
            wine_quality[column] = np.where((wine_quality[column] < lower_limit) | (wine_quality[column] > upper_limit), wine_quality[column].mean(), wine_quality[column])

#Making binary classificaion for the response variable.
#Dividing wine as good and bad by giving the limit for the quality
bins = (2, 6.5, 8)
group_names = ['bad', 'good']
wine_quality['quality'] = pd.cut(wine_quality['quality'], bins = bins, labels = group_names)

#Now lets assign a labels to our quality variable
label_quality = LabelEncoder()

#Bad becomes 0 and good becomes 1 
wine_quality['quality'] = label_quality.fit_transform(wine_quality['quality'])
wine_quality['quality'].value_counts()

#Now seperate the dataset as response variable and feature variabes
x = wine_quality.drop('quality', axis = 1)
y = wine_quality['quality']

# Split the dataset
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=None)

# Preprocess the data (Standard Scaling)
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Define the optimization problem using NeuralNetwork
problem_nn = mlrose_hiive.NeuralNetwork(hidden_nodes=[5, 5],
                                        activation='relu',
                                        algorithm='random_hill_climb',
                                        max_iters=100,
                                        bias=True,
                                        is_classifier=True,
                                        learning_rate=0.1,
                                        early_stopping=False,
                                        clip_max=1e+10,
                                        restarts=0,
                                        schedule=mlrose_hiive.GeomDecay(),
                                        pop_size=200,
                                        mutation_prob=0.1,
                                        max_attempts=10,
                                        random_state=None,
                                        curve=True)

# Predict labels for train set and assess accuracy
problem_nn.fit(x_train_scaled, y_train)
y_train_pred = problem_nn.predict(x_train_scaled)
y_train_accuracy = accuracy_score(y_train, y_train_pred)
print(y_train_accuracy)

# Predict labels for test set and assess accuracy
y_test_pred = problem_nn.predict(x_test_scaled)
y_test_accuracy = accuracy_score(y_test, y_test_pred)
print(y_test_accuracy)