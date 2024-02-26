import string
import numpy as np
import matplotlib.pyplot as plt

# Function to plot the chessboard with queens
def plot_board(state, num_queens, title, fitness):
    
    board = np.zeros((num_queens, num_queens))
    for i, j in enumerate(np.array(state).astype(int)):
        board[i, j] = 1

    # Display plot as chessboard
    plt.imshow(board, cmap='binary', interpolation='none')

    # Add grid lines
    for i in range(num_queens + 1):
        plt.axvline(x=i - 0.5, color='grey', linestyle='-', linewidth=2)
        plt.axhline(y=i - 0.5, color='grey', linestyle='-', linewidth=2)

    # Add row labels
    plt.yticks(np.arange(num_queens), np.arange(1, num_queens + 1))
    plt.gca().invert_yaxis()  # Invert the order of y ticks

    # Add column labels
    row_labels = [char for char in string.ascii_lowercase[:num_queens]]
    plt.xticks(np.arange(num_queens), row_labels)

    # Add title
    plt.title(f"{title}\nFitness: {fitness}")


def hyperparameter_tuning(num_queens, fitness_nqp):
    return


def display_results(results_dict, num_queens):
    
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
    plt.savefig('Images/NQP/NQP_Fitness_Curve.png')
    plt.close()

    # Plot the final positions of the board for each algorithm
    plt.figure(figsize=(8, 8))
    for i, (algorithm_name, result) in enumerate(results_dict.items(), 1):
        plt.subplot(2, 2, i)
        plot_board(result['Best_State'], num_queens, f'{algorithm_name} - Final Position', int(results_dict[algorithm_name]['Best_Fitness']))

    plt.tight_layout()
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, hspace=0.3, wspace=0.2)

    plt.savefig('Images/NQP/NQP_Final_Positions.png')
    plt.close()