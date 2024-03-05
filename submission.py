import mlrose_hiive
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score

import nn
import tsp
import nqp
import gcp
import run_algorithms
import GCP_Custom

alg_names = ['Random Hill Climb', 'Simulated Annealing', 'Genetic Algorithm', 'MIMIC']
###############################################################################
######################### Traveling Salesperson (TSP) #########################
###############################################################################
# Define the number of cities and area of the city
num_cities = [10, 50, 100]
area = 40
results = []

for index, cities in enumerate(num_cities):

    # Define an optimization problem using TSPGenerator
    problem_tsp = mlrose_hiive.TSPGenerator.generate(seed=None, number_of_cities=cities, area_width=area, area_height=area)
    problem_tsp.maximize = -1.0

    # Display the city coordinates
    tsp.display_city_coordinates(problem_tsp.coords, area, cities)

    # Perform hyperparameter tuning
    run_algorithms.hyperparameter_tuning(problem_tsp, 'TSP', cities)

    # Define optimization algorithms with optimized parameters
    algorithms = {
        'Random Hill Climbing': {
            'runner': mlrose_hiive.RHCRunner,
            'params': {
                'experiment_name'   : 'RHC',
                'seed'              : None,
                'iteration_list'    : [500],
                'restart_list'      : [5],
                'max_attempts'      : 100,
                'generate_curves'   : True
            }
        },
        'Simulated Annealing': {
            'runner': mlrose_hiive.SARunner,
            'params': {
                'experiment_name'   : 'SA',
                'seed'              : None,
                'iteration_list'    : [500],
                'temperature_list'  : [10],
                'decay_list'        : [mlrose_hiive.GeomDecay],
                'max_attempts'      : 100,
                'generate_curves'   : True
            }
        },
        'Genetic Algorithm': {
            'runner': mlrose_hiive.GARunner,
            'params': {
                'experiment_name'   : 'GA',
                'seed'              : None,
                'iteration_list'    : [500],
                'population_sizes'  : [200],
                'mutation_rates'    : [0.4],
                'hamming_factors'   : None,
                'hamming_factors_decays' : None,
                'max_attempts'      : 100,
                'generate_curves'   : True
            }
        },
        'MIMIC': {
            'runner': mlrose_hiive.MIMICRunner,
            'params': {
                'experiment_name'   : 'MIMIC',
                'seed'              : None,
                'iteration_list'    : [500],
                'population_sizes'  : [400],
                'keep_percent_list' : [0.6],
                'max_attempts'      : 100,
                'generate_curves'   : True
            }
        }
    }
    results.append(run_algorithms.run_algorithms(problem_tsp, algorithms))

    # Display final results
    tsp.display_results(results[index], problem_tsp.coords, area, cities)

rhc_size = [round(index['Random Hill Climbing']['Best_Fitness'], 3) for index in results]
sa_size = [round(index['Simulated Annealing']['Best_Fitness'], 3) for index in results]
ga_size = [round(index['Genetic Algorithm']['Best_Fitness'], 3) for index in results]
mimic_size = [round(index['MIMIC']['Best_Fitness'], 3) for index in results]
 
# Concatenate arrays and reshape
best_fitness_arrays = np.concatenate([rhc_size, sa_size, ga_size, mimic_size])
reshaped_array = np.reshape(best_fitness_arrays, (4, 3))

# Create subplots
fig, axes = plt.subplots(1, 4, figsize=(16, 4))

# Plot each set of best_fitness arrays on a separate subplot
for i, (fitness_values, alg_name) in enumerate(zip(reshaped_array, alg_names)):
    bars = axes[i].bar(range(1, 4), fitness_values)
    axes[i].set_xlabel('Population Size')
    axes[i].set_ylabel('Best Fitness')
    axes[i].set_title(f'{alg_name}')

    # Set x-ticks and labels to only display num_cities values
    axes[i].set_xticks(range(1, 4))
    axes[i].set_xticklabels(num_cities)

    # Add annotations for fitness values
    for bar, value in zip(bars, fitness_values):
        axes[i].text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                     f'{value}', ha='center', va='bottom')

# Set super title
plt.suptitle('Traveling Salesperson - Fitness vs Population Size', fontsize=16, y=1.05)

# Adjust layout and show the plot
plt.tight_layout()
plt.savefig('Images/TSP/Fitness vs. Population Size.png')
plt.close()

###############################################################################
############################### N-Queens (NQP) ################################
###############################################################################
# Define the number of queens
num_queens = [10, 50, 100]
results = []

for index, queens in enumerate(num_queens):

    # Define an optimization problem using QueensGenerator
    problem_nqp = mlrose_hiive.QueensGenerator.generate(seed=None, size=queens)
    problem_nqp.maximize = -1.0

    # Initialize the state (queens' positions)
    init_state = np.random.permutation(queens)

    # Plot the initial position
    nqp.plot_board(init_state, queens, 'Initial Position of N-Queens', int(mlrose_hiive.Queens().evaluate(init_state)))
    plt.savefig(f'Images/NQP/NQP Initial State-{queens}.png')
    plt.close()

    # Perform hyperparameter tuning
    run_algorithms.hyperparameter_tuning(problem_nqp, 'NQP', queens)

    # Define optimization algorithms with optimized parameters
    algorithms = {
        'Random_Hill_Climbing': {
            'runner': mlrose_hiive.RHCRunner,
            'params': {
                'experiment_name'   : 'RHC',
                'seed'              : None,
                'iteration_list'    : [500],
                'restart_list'      : [15],
                'max_attempts'      : 500,
                'generate_curves'   : True,
            }
        },
        'Simulated_Annealing': {
            'runner': mlrose_hiive.SARunner,
            'params': {
                'experiment_name'   : 'SA',
                'seed'              : None,
                'iteration_list'    : [500],
                'temperature_list'  : [10],
                'decay_list'        : [mlrose_hiive.GeomDecay],
                'max_attempts'      : 500,
                'generate_curves'   : True,
            }
        },
        'Genetic_Algorithm': {
            'runner': mlrose_hiive.GARunner,
            'params': {
                'experiment_name'   : 'GA',
                'seed'              : None,
                'iteration_list'    : [500],
                'population_sizes'  : [300],
                'mutation_rates'    : [0.4],
                'hamming_factors'   : None,
                'hamming_factors_decays' : None,
                'max_attempts'      : 500,
                'generate_curves'   : True
            }
        },
        'MIMIC': {
            'runner': mlrose_hiive.MIMICRunner,
            'params': {
                'experiment_name'   : 'MIMIC',
                'seed'              : None,
                'iteration_list'    : [500],
                'population_sizes'  : [400],
                'keep_percent_list' : [0.4],
                'max_attempts'      : 500,
                'generate_curves'   : True
            }
        }
    }
    results.append(run_algorithms.run_algorithms(problem_nqp, algorithms))

    # Show final results
    nqp.display_results(results[index], queens)

rhc_size = [round(index['Random_Hill_Climbing']['Best_Fitness'], 3) for index in results]
sa_size = [round(index['Simulated_Annealing']['Best_Fitness'], 3) for index in results]
ga_size = [round(index['Genetic_Algorithm']['Best_Fitness'], 3) for index in results]
mimic_size = [round(index['MIMIC']['Best_Fitness'], 3) for index in results]
 
# Concatenate arrays and reshape
best_fitness_arrays = np.concatenate([rhc_size, sa_size, ga_size, mimic_size])
reshaped_array = np.reshape(best_fitness_arrays, (4, 3))

# Create subplots
fig, axes = plt.subplots(1, 4, figsize=(16, 4))

# Plot each set of best_fitness arrays on a separate subplot
for i, (fitness_values, alg_name) in enumerate(zip(reshaped_array, alg_names)):
    bars = axes[i].bar(range(1, 4), fitness_values)
    axes[i].set_xlabel('Population Size')
    axes[i].set_ylabel('Best Fitness')
    axes[i].set_title(f'{alg_name}')

    # Set x-ticks and labels to only display num_queens values
    axes[i].set_xticks(range(1, 4))
    axes[i].set_xticklabels(num_queens)

    # Add annotations for fitness values
    for bar, value in zip(bars, fitness_values):
        axes[i].text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                     f'{value}', ha='center', va='bottom')

# Set super title
plt.suptitle('N-Queens - Fitness vs Population Size', fontsize=16, y=1.05)

# Adjust layout and show the plot
plt.tight_layout()
plt.savefig('Images/NQP/Fitness vs. Population Size.png')
plt.close()

###############################################################################
############################ Graph Coloring (GCP) #############################
###############################################################################
# Number of nodes in graph
num_nodes = [10, 50, 100]
results = []

for index, nodes in enumerate(num_nodes):
    # Define an optimization problem using MaxKColorGenerator
    problem_gcp = GCP_Custom.MaxKColorGenerator.generate(seed=None, number_of_nodes=nodes, max_connections_per_node=3, max_colors=3)
    problem_gcp.maximize = -1.0

    # Display Graph
    gcp.display_graph(problem_gcp, nodes)

    # Perform hyperparameter tuning
    run_algorithms.hyperparameter_tuning(problem_gcp, 'GCP', nodes)

    # Define optimization algorithms with optimized parameters
    algorithms = {
        'Random_Hill_Climbing': {
            'runner': mlrose_hiive.RHCRunner,
            'params': {
                'experiment_name'   : 'RHC',
                'seed'              : None,
                'iteration_list'    : [500],
                'restart_list'      : [15],
                'max_attempts'      : 100,
                'generate_curves'   : True
            }
        },
        'Simulated_Annealing': {
            'runner': mlrose_hiive.SARunner,
            'params': {
                'experiment_name'   : 'SA',
                'seed'              : None,
                'iteration_list'    : [500],
                'temperature_list'  : [10],
                'decay_list'        : [mlrose_hiive.GeomDecay],
                'max_attempts'      : 100,
                'generate_curves'   : True
            }
        },
        'Genetic_Algorithm': {
            'runner': mlrose_hiive.GARunner,
            'params': {
                'experiment_name'   : 'GA',
                'seed'              : None,
                'iteration_list'    : [500],
                'population_sizes'  : [300],
                'mutation_rates'    : [0.4],
                'hamming_factors'   : None,
                'hamming_factors_decays' : None,
                'max_attempts'      : 100,
                'generate_curves'   : True
            }
        },
        'MIMIC': {
            'runner': mlrose_hiive.MIMICRunner,
            'params': {
                'experiment_name'   : 'MIMIC',
                'seed'              : None,
                'iteration_list'    : [500],
                'population_sizes'  : [400],
                'keep_percent_list' : [0.4],
                'max_attempts'      : 100,
                'generate_curves'   : True
            }
        }
    }
    results.append(run_algorithms.run_algorithms(problem_gcp, algorithms))

    # Show final results
    gcp.display_results(results[index], problem_gcp.source_graph, nodes)

rhc_size = [round(index['Random_Hill_Climbing']['Best_Fitness'], 3) for index in results]
sa_size = [round(index['Simulated_Annealing']['Best_Fitness'], 3) for index in results]
ga_size = [round(index['Genetic_Algorithm']['Best_Fitness'], 3) for index in results]
mimic_size = [round(index['MIMIC']['Best_Fitness'], 3) for index in results]
 
# Concatenate arrays and reshape
best_fitness_arrays = np.concatenate([rhc_size, sa_size, ga_size, mimic_size])
reshaped_array = np.reshape(best_fitness_arrays, (4, 3))

# Create subplots
fig, axes = plt.subplots(1, 4, figsize=(16, 4))

# Plot each set of best_fitness arrays on a separate subplot
for i, (fitness_values, alg_name) in enumerate(zip(reshaped_array, alg_names)):
    bars = axes[i].bar(range(1, 4), fitness_values)
    axes[i].set_xlabel('Population Size')
    axes[i].set_ylabel('Best Fitness')
    axes[i].set_title(f'{alg_name}')

    # Set x-ticks and labels to only display num_nodes values
    axes[i].set_xticks(range(1, 4))
    axes[i].set_xticklabels(num_nodes)

    # Add annotations for fitness values
    for bar, value in zip(bars, fitness_values):
        axes[i].text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                     f'{value}', ha='center', va='bottom')

# Set super title
plt.suptitle('Graph Coloring - Fitness vs Population Size', fontsize=16, y=1.05)

# Adjust layout and show the plot
plt.tight_layout()
plt.savefig('Images/GCP/Fitness vs. Population Size.png')
plt.close()

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

# Alter dataset to be binary classificaion
bins = (2, 6.5, 8)
group_names = ['bad', 'good']
wine_quality['quality'] = pd.cut(wine_quality['quality'], bins = bins, labels = group_names)

# Assign labels to quality variable
label_quality = LabelEncoder()

# Bad becomes 0 and good becomes 1 
wine_quality['quality'] = label_quality.fit_transform(wine_quality['quality'])
value_count = wine_quality['quality'].value_counts()

# Seperate the dataset as response variable and feature variabes
x = wine_quality.drop('quality', axis = 1)
y = wine_quality['quality']

# Split the dataset
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=None)

# Preprocess the data (Standard Scaling)
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

y_train = y_train.values
y_test = y_test.values

# Use backpropagation to train neural network
# Initialize MLPClassifier
mlp = MLPClassifier(
    hidden_layer_sizes=(10,), 
    activation='relu', 
    solver='adam', 
    alpha=0.0001, 
    batch_size='auto', 
    learning_rate='constant', 
    learning_rate_init=0.01, 
    power_t=0.5, 
    max_iter=100, 
    shuffle=True, 
    random_state=4, 
    tol=1e-4, 
    verbose=False, 
    warm_start=False, 
    momentum=0.9, 
    nesterovs_momentum=True, 
    early_stopping=False, 
    validation_fraction=0.1, 
    beta_1=0.9, 
    beta_2=0.999, 
    epsilon=1e-8
)
nn.train_nn(mlp, x_train_scaled, y_train)

# Testing parameters
params_rhc = ({'max_iters': [100, 500, 1000], 'restarts': [5, 15, 30]})
params_sa = ({'max_iters': [100, 500, 1000], 
              'schedule': [mlrose_hiive.ArithDecay(), mlrose_hiive.ExpDecay(), mlrose_hiive.GeomDecay()]})
params_ga = ({'pop_size': [100, 200, 300, 400], 'mutation_prob': [0.2, 0.4, 0.6, 0.8]})

# Plot Curves
graph_names = ['RHC', 'SA', 'GA']
run_stats = []
curves = []
run_stats_results, curves_results = nn.plot_loss(mlrose_hiive.random_hill_climb, params_rhc, 'RHC', x_train_scaled, y_train, x_test_scaled, y_test)
run_stats.append(run_stats_results)
curves.append(curves_results)
run_stats_results, curves_results = nn.plot_loss(mlrose_hiive.simulated_annealing, params_sa, 'SA', x_train_scaled, y_train, x_test_scaled, y_test)
run_stats.append(run_stats_results)
curves.append(curves_results)
run_stats_results, curves_results = nn.plot_loss(mlrose_hiive.genetic_alg, params_ga, 'GA', x_train_scaled, y_train, x_test_scaled, y_test)
run_stats.append(run_stats_results)
curves.append(curves_results)

for index, curve in enumerate(curves):
    plt.plot(curve['Iteration'], curve['Fitness'], label=graph_names[index])
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title(f'Loss vs Iteration - Random Optimization Algorithms')
plt.legend(graph_names)
plt.savefig(f'Images/NN/Loss vs Iteration.png')
plt.close()

# Use Random Hill Climb to train neural network
problem_nn_rhc = nn.CustomNeuralNetwork(hidden_nodes=[10],
                                        activation='relu',
                                        algorithm='random_hill_climb',
                                        max_iters=1000,
                                        bias=False,
                                        is_classifier=True,
                                        learning_rate=0.01,
                                        early_stopping=False,
                                        clip_max=100,
                                        restarts=15,
                                        random_state=4,
                                        curve=True)

# Use Simulated Annealing to train neural network
problem_nn_sa = nn.CustomNeuralNetwork(hidden_nodes=[10],
                                       activation='relu',
                                       algorithm='simulated_annealing',
                                       max_iters=1000,
                                       bias=False,
                                       is_classifier=True,
                                       learning_rate=0.01,
                                       early_stopping=False,
                                       clip_max=100,
                                       schedule=mlrose_hiive.GeomDecay(),
                                       random_state=4,
                                       curve=True)

# Use Genetic Algorithm to train neural network
problem_nn_ga = nn.CustomNeuralNetwork(hidden_nodes=[10],
                                       activation='relu',
                                       algorithm='genetic_alg',
                                       max_iters=1000,
                                       bias=False,
                                       is_classifier=True,
                                       learning_rate=0.01,
                                       early_stopping=False,
                                       clip_max=100,
                                       pop_size=300,
                                       mutation_prob=0.4,
                                       random_state=4,
                                       curve=True)

# Fit object
problem_nn_rhc.fit(x_train_scaled, y_train)
problem_nn_sa.fit(x_train_scaled, y_train)
problem_nn_ga.fit(x_train_scaled, y_train)
mlp.fit(x_train_scaled, y_train)

y_train_pred = []
y_test_pred = []
y_train_accuracy = []
y_test_accuracy = []

# Predict labels for train set and assess accuracy
y_train_pred.append(problem_nn_rhc.predict(x_train_scaled))
y_train_accuracy.append(accuracy_score(y_train, y_train_pred[0]))

y_train_pred.append(problem_nn_sa.predict(x_train_scaled))
y_train_accuracy.append(accuracy_score(y_train, y_train_pred[1]))

y_train_pred.append(problem_nn_ga.predict(x_train_scaled))
y_train_accuracy.append(accuracy_score(y_train, y_train_pred[2]))

y_train_pred.append(mlp.predict(x_train_scaled))
y_train_accuracy.append(accuracy_score(y_train, y_train_pred[3]))

# Predict labels for test set and assess accuracy
y_test_pred.append(problem_nn_rhc.predict(x_test_scaled))
y_test_accuracy.append(accuracy_score(y_test, y_test_pred[0]))

y_test_pred.append(problem_nn_sa.predict(x_test_scaled))
y_test_accuracy.append(accuracy_score(y_test, y_test_pred[1]))

y_test_pred.append(problem_nn_ga.predict(x_test_scaled))
y_test_accuracy.append(accuracy_score(y_test, y_test_pred[2]))

y_test_pred.append(mlp.predict(x_test_scaled))
y_test_accuracy.append(accuracy_score(y_test, y_test_pred[3]))

final_graphs = ['RHC', 'SA', 'GA', 'Backprop']

indices = np.arange(len(final_graphs))

# Bar width for better visualization
bar_width = 0.35

# Create the bar plot
plt.bar(indices, y_train_accuracy, bar_width, label='Training Accuracy', color='blue')
plt.bar(indices + bar_width, y_test_accuracy, bar_width, label='Test Accuracy', color='orange')

# Adding labels and title
plt.xlabel('Random Optimization Algorithms')
plt.ylabel('Accuracy')
plt.title('Random Optimization Algorithm Performance vs Backprop')
plt.xticks(indices + bar_width/2, final_graphs)  # Set x-axis ticks at the center of each group
plt.legend(loc='lower right')

# Add annotations above each bar
for i, (bar1, bar2) in enumerate(zip(y_train_accuracy, y_test_accuracy)):
    plt.annotate(f'{bar1:.2f}', ((indices - bar_width/2)[i]+0.15, bar1), 
             textcoords="offset points", ha='center', va='bottom')
    plt.annotate(f'{bar2:.2f}', ((indices + bar_width/2)[i]+0.15, bar2), 
             textcoords="offset points", ha='center', va='bottom')

# Show the plot
plt.savefig('Images/NN/Final Prediction Scores.png')
plt.close()