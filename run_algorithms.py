import mlrose_hiive
import ast
import time
import numpy as np
import matplotlib.pyplot as plt

def run_algorithms(problem, algorithms):
    # Define a dictionary to store results
    results_dict = {}

    # Run each algorithm and update the results dictionary
    for algorithm_name, algorithm_info in algorithms.items():
        runner_class = algorithm_info['runner']
        algorithm_params = algorithm_info['params']

        # Create a runner
        runner = runner_class(problem=problem, **algorithm_params)

        # Record the start time
        start_time = time.time()

        # Run the algorithm
        run_stats, curves = runner.run()

        # Record the end time
        end_time = time.time()

        # Inverse the fitness to convert to maximization problem
        if problem.maximize == -1.0:
            run_stats['Fitness'] = (1.0 / run_stats['Fitness']) * 500
            curves['Fitness'] = (1.0 / curves['Fitness']) * 500
            run_stats['Fitness'].replace(np.inf, 1000, inplace=True)
            curves['Fitness'].replace(np.inf, 1000, inplace=True)

        # Extract results
        best_state_index = run_stats['Fitness'].idxmax()  # Index of the row with the best fitness
        best_state = run_stats.loc[best_state_index, 'State']
        best_fitness = run_stats['Fitness'].max()  # Best fitness value in the curve
        fitness_curve = curves['Fitness']
        function_evals = curves['FEvals']
        iterations = curves.index
        
        # Generate evenly spaced X values over the elapsed time
        elapsed_time = end_time - start_time
        wall_time = np.linspace(0, elapsed_time, len(function_evals))

        results_dict[algorithm_name] = {
            'Best_State': ast.literal_eval(best_state),
            'Best_Fitness': best_fitness,
            'Fitness_Curve': fitness_curve,
            'FEvals': function_evals,
            'Iteration': iterations,
            'Time': wall_time
        }

    return results_dict

def hyperparameter_tuning(problem, problem_id, pop_size):
    algorithms = {
        'Random Hill Climbing': {
            'runner': mlrose_hiive.RHCRunner,
            'params': {
                'experiment_name'   : 'RHC',
                'output_directory'  : 'Output/' + problem_id,
                'seed'              : 4,
                'iteration_list'    : [10, 50, 100, 500, 1000],
                'restart_list'      : [5, 15, 30, 50],
                'max_attempts'      : 1000,
                'generate_curves'   : True
            }
        },
        'Simulated Annealing': {
            'runner': mlrose_hiive.SARunner,
            'params': {
                'experiment_name'   : 'SA',
                'output_directory'  : 'Output/' + problem_id,
                'seed'              : 4,
                'iteration_list'    : [10, 50, 100, 500, 1000],
                'temperature_list'  : [1, 10, 50, 100, 500, 1000],
                'decay_list'        : [mlrose_hiive.GeomDecay, mlrose_hiive.ArithDecay, mlrose_hiive.ExpDecay],
                'max_attempts'      : 1000,
                'generate_curves'   : True
            }
        },
        'Genetic Algorithm': {
            'runner': mlrose_hiive.GARunner,
            'params': {
                'experiment_name'   : 'GA',
                'output_directory'  : 'Output/' + problem_id,
                'seed'              : 4,
                'iteration_list'    : [10, 50, 100, 500, 1000],
                'population_sizes'  : [100, 200, 300, 400, 500],
                'mutation_rates'    : [0.2, 0.4, 0.6, 0.8],
                'hamming_factors'   : None,
                'hamming_factors_decays' : None,
                'max_attempts'      : 1000,
                'generate_curves'   : True
            }
        },
        'MIMIC': {
            'runner': mlrose_hiive.MIMICRunner,
            'params': {
                'experiment_name'   : 'MIMIC',
                'output_directory'  : 'Output/' + problem_id,
                'seed'              : 4,
                'iteration_list'    : [10, 50, 100, 500, 1000],
                'population_sizes'  : [100, 200, 300, 400, 500],
                'keep_percent_list' : [0.2, 0.4, 0.6, 0.8],
                'max_attempts'      : 1000,
                'generate_curves'   : True
            }
        }
    }

    # Define the parameters to optimize
    params = {
        'Random Hill Climbing': {
            'Iteration'             : [10, 50, 100, 500, 1000],
            'Restarts'              : [5, 15, 30, 50],
        },
        'Simulated Annealing': {
            'Iteration'             : [10, 50, 100, 500, 1000],
            'schedule_init_temp'    : [1, 10, 50, 100, 500, 1000],
            'schedule_type'         : ['geometric', 'arithmetic', 'exponential'],
        },
        'Genetic Algorithm': {
            'Iteration'             : [10, 50, 100, 500, 1000],
            'Population Size'       : [100, 200, 300, 400, 500],
            'Mutation Rate'         : [0.2, 0.4, 0.6, 0.8],
        },
        'MIMIC': {
            'Iteration'             : [10, 50, 100, 500, 1000],
            'Population Size'       : [100, 200, 300, 400, 500],
            'Keep Percent'          : [0.2, 0.4, 0.6, 0.8],
        }
    }

    # Run each algorithm and update the results dictionary
    for algorithm_name, algorithm_info in algorithms.items():
        runner_class = algorithm_info['runner']
        algorithm_params = algorithm_info['params']

        # Create a runner
        runner = runner_class(problem=problem, **algorithm_params)

        # Run the algorithm
        run_stats, curves = runner.run()
        
        # Inverse the fitness to convert to maximization problem
        if problem.maximize == -1.0:
            run_stats['Fitness'] = (1.0 / run_stats['Fitness']) * 1000
            run_stats['Fitness'].replace(np.inf, 1000, inplace=True)

        # Create Subplots
        fig, axes = plt.subplots(1, len(params[algorithm_name]), figsize=(10, 5))
        axes[0].set_ylabel('Fitness', fontsize=12)

        # Loop through each parameter to optimize
        for idx, (param_name, param_range) in enumerate(params[algorithm_name].items()):
            
            # Initialize lists to store values for plotting
            x_values = np.arange(len(param_range))
            y_values = []

            # Iterate through each parameter value
            for param_value in param_range:
                # Filter stats for the specific parameter value
                filtered_stats = run_stats[run_stats[param_name] == param_value]
                
                # Get the maximum fitness value for this parameter value
                max_fitness = filtered_stats['Fitness'].max()
                
                # Append values to the lists
                y_values.append(max_fitness)

                # Annotate the bar with fitness value
                axes[idx].annotate(f'{max_fitness:.2f}', (x_values[param_range.index(param_value)], max_fitness),
                   textcoords="offset points", xytext=(0,3), ha='center')

            # Plot a bar for each parameter value
            axes[idx].bar(x_values, y_values, label=param_name, width=0.3)  # Adjust width as needed
            axes[idx].set_xticks(x_values)  # Set ticks to match the number of parameter values
            axes[idx].set_xticklabels(param_range)  # Set tick labels to parameter values
            axes[idx].set_xlabel(param_name, fontsize=10)
        
        # Set title for the entire subplot
        fig.suptitle(f'{algorithm_name} - Fitness vs Parameter Values - Population Size = {pop_size}', fontsize=15)
        
        # Adjust layout for better spacing
        plt.tight_layout()
        
        # Show the plot
        plt.savefig('Images/'f'{problem_id}/{algorithm_name}-{pop_size}.png')
        plt.close()