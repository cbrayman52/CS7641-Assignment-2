import mlrose_hiive
import ast
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

        # Run the algorithm
        run_stats, curves = runner.run()

        # Extract results
        best_state_index = run_stats['Fitness'].idxmin()  # Index of the row with the best fitness
        best_state = run_stats.loc[best_state_index, 'State']
        best_fitness = run_stats['Fitness'].min()  # Best fitness value in the curve
        fitness_curve = curves['Fitness']

        results_dict[algorithm_name] = {
            'Best_State': ast.literal_eval(best_state),
            'Best_Fitness': best_fitness,
            'Fitness_Curve': fitness_curve
        }

    return results_dict

def hyperparameter_tuning(problem, problem_id):
    algorithms = {
        'Random_Hill_Climbing': {
            'runner': mlrose_hiive.RHCRunner,
            'params': {
                'experiment_name'   : 'RHC',
                'output_directory'  : 'Output/' + problem_id,
                'seed'              : 4,
                'iteration_list'    : [10, 20, 50, 100],
                'restart_list'      : [5, 10, 15, 20],
                'max_attempts'      : 20,
                'generate_curves'   : True
            }
        },
        'Simulated_Annealing': {
            'runner': mlrose_hiive.SARunner,
            'params': {
                'experiment_name'   : 'SA',
                'output_directory'  : 'Output/' + problem_id,
                'seed'              : 4,
                'iteration_list'    : [10, 20, 50, 100],
                'temperature_list'  : [10, 20, 50, 100],
                'decay_list'        : [mlrose_hiive.GeomDecay, mlrose_hiive.ArithDecay, mlrose_hiive.ExpDecay],
                'max_attempts'      : 10,
                'generate_curves'   : True
            }
        },
        'Genetic_Algorithm': {
            'runner': mlrose_hiive.GARunner,
            'params': {
                'experiment_name'   : 'GA',
                'output_directory'  : 'Output/' + problem_id,
                'seed'              : 4,
                'iteration_list'    : [10, 20, 50, 100],
                'population_sizes'  : [100, 200, 300, 400],
                'mutation_rates'    : [0.2, 0.4, 0.6, 0.8],
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
                'output_directory'  : 'Output/' + problem_id,
                'seed'              : 4,
                'iteration_list'    : [10, 20, 50, 100],
                'population_sizes'  : [100, 200, 300, 400],
                'keep_percent_list' : [0.2, 0.4, 0.6, 0.8],
                'max_attempts'      : 10,
                'generate_curves'   : True
            }
        }
    }

    # Define the parameters to optimize
    params = {
        'Random_Hill_Climbing': {
            'Iteration'             : [10, 20, 50, 100],
            'Restarts'              : [5, 10, 15, 20],
        },
        'Simulated_Annealing': {
            'Iteration'             : [10, 20, 50, 100],
            'schedule_init_temp'    : [10, 20, 50, 100],
            'schedule_type'         : ['geometric', 'arithmetic', 'exponential'],
        },
        'Genetic_Algorithm': {
            'Iteration'             : [10, 20, 50, 100],
            'Population Size'       : [100, 200, 300, 400],
            'Mutation Rate'         : [0.2, 0.4, 0.6, 0.8],
        },
        'MIMIC': {
            'Iteration'             : [10, 20, 50, 100],
            'Population Size'       : [100, 200, 300, 400],
            'Keep Percent'          : [0.2, 0.4, 0.6, 0.8],
        }
    }

    # Define a dictionary to store results
    results_dict = {}

    # Run each algorithm and update the results dictionary
    for algorithm_name, algorithm_info in algorithms.items():
        runner_class = algorithm_info['runner']
        algorithm_params = algorithm_info['params']

        # Create a runner
        runner = runner_class(problem=problem, **algorithm_params)

        # Run the algorithm
        run_stats, curves = runner.run()

        # Create Subplots
        fig, axes = plt.subplots(1, len(params[algorithm_name]), figsize=(15, 5))
        axes[0].set_ylabel('Fitness', fontsize=12)

        # Loop through each parameter to optimize
        for idx, (param_name, param_range) in enumerate(params[algorithm_name].items()):
            
            # Initialize lists to store values for plotting
            x_values = np.arange(len(param_range))
            y_values = []

            for param_value in param_range:
                # Filter stats for the specific parameter value
                filtered_stats = run_stats[run_stats[param_name] == param_value]
                
                # Get the maximum fitness value for this parameter value
                max_fitness = filtered_stats['Fitness'].min()
                
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
            
            # Add legend
            # axes[idx].legend()
        
        # Set title for the entire subplot
        fig.suptitle(f'{algorithm_name} - Fitness vs Parameter Values', fontsize=15)
        
        # Adjust layout for better spacing
        plt.tight_layout()
        
        # Show the plot
        f'{problem_id}_{algorithm_name}_HP.png'
        plt.savefig('Images/'f'{problem_id}/{algorithm_name}_HP.png')
        plt.close()