import matplotlib.pyplot as plt

def display_city_coordinates(coords, area):

    # Extract x and y coordinates separately
    x_coordinates, y_coordinates = zip(*coords)

    # Plot the coordinates
    plt.scatter(x_coordinates, y_coordinates, color='blue', marker='o', label='Coordinates')

    # Add grid lines
    plt.grid(True)

    # Annotate each point with its corresponding number
    for i, (x, y) in enumerate(coords):
        plt.annotate(str(i+1), (x, y), textcoords="offset points", xytext=(0,5), ha='center')

    # Add labels, title, and legend
    plt.xlim(-1, area+1)
    plt.ylim(-1, area+1)
    plt.xticks(range(0, area+1, 2))
    plt.yticks(range(0, area+1, 2))
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Scatter Plot of Coordinates')
    plt.legend(loc='best')
    plt.savefig('Images/TSP/TSP_Initial_State.png')
    plt.close()


def display_results(results_dict, city_coordinates, area):
    
    # Plot fitness curves for each algorithm
    for algorithm_name, results in results_dict.items():
        fitness_curve = results['Fitness_Curve']
        iterations = range(1, len(fitness_curve) + 1)
        
        plt.plot(iterations, fitness_curve, label=algorithm_name)

    # Add labels, title, and legend
    plt.xlabel('Iteration')
    plt.ylabel('Fitness Value')
    plt.title('Fitness Curve Comparison for Optimization Algorithms')
    plt.legend()
    plt.savefig('Images/TSP/TSP_Fitness_Curve.png')
    plt.close()

    # Extract x and y coordinates of cities
    x_cities, y_cities = zip(*city_coordinates)

    # Get the number of algorithms and determine the layout of subplots
    num_algorithms = len(results_dict)
    num_rows = 2
    num_cols = (num_algorithms + num_rows - 1) // num_rows

    # Create subplots
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 8))

    # Flatten axes
    axes = axes.flatten() if num_algorithms > 1 else [axes]

    # Iterate through all algorithms in results_dict
    for i, (algorithm_name, results) in enumerate(zip(results_dict.keys(), results_dict.values())):
        best_tsp_state = results['Best_State']

        # Plot the cities
        axes[i].scatter(x_cities, y_cities, color='blue', marker='o', label='Cities')

        # Add grid lines
        axes[i].grid(True)

        # Annotate each point with its corresponding number
        for j, (x, y) in enumerate(city_coordinates):
            axes[i].annotate(str(j+1), (x, y), textcoords="offset points", xytext=(0,5), ha='center')

        # Plot the best state with dashed red lines
        best_tsp_state = list(best_tsp_state) + [best_tsp_state[0]]  # Connect the last city to the first city
        x_best_state = [x_cities[i] for i in best_tsp_state]
        y_best_state = [y_cities[i] for i in best_tsp_state]

        # Plot the best state with dashed red lines
        algorithm_label = f'{algorithm_name} - Best State'
        axes[i].plot(x_best_state, y_best_state, linestyle='--', color='red', label=algorithm_label)

        # Set title for each subplot
        fitness = int(results['Best_Fitness'])
        axes[i].set_title(f'TSP - {algorithm_name}\nFitness: {fitness}')

        # Set x-axis and y-axis limits
        axes[i].set_xlim(-1, area+1)
        axes[i].set_ylim(-1, area+1)

        # Add custom x and y ticks
        axes[i].set_xticks(range(0, area+1, 2))
        axes[i].set_yticks(range(0, area+1, 2))

    # Adjust layout for better spacing
    plt.tight_layout()
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, hspace=0.3, wspace=0.2)

    # Save the plot
    plt.savefig('Images/TSP/TSP_Solutions.png')
    plt.close()