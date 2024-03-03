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
    if num_queens < 27:
        plt.xticks(np.arange(num_queens), row_labels)
    else:
        plt.xticks(np.arange(num_queens), np.arange(1, num_queens + 1))

    # Add title
    plt.title(f'{title}\nFitness: {fitness}')


def display_results(results_dict, num_queens):
    
    # Plot fitness curves for each algorithm
    for algorithm_name, results in results_dict.items():
        fitness_curve = results['Fitness_Curve']
        iterations = range(1, len(fitness_curve) + 1)
        
        plt.plot(iterations, fitness_curve, label=algorithm_name)

    # Add labels, title, and legend
    plt.xlabel('Iteration')
    plt.ylabel('Fitness Value')
    plt.title('N-Queens - Fitness Curve')
    plt.legend()
    plt.savefig(f'Images/NQP/NQP Fitness Curve-{num_queens}.png')
    plt.close()

    # Plot function evaluation curves for each algorithm
    for algorithm_name, results in results_dict.items():  
        fevals = results['FEvals']
        times = results['Time']
        iterations = results['Iteration']
        final_iterations = iterations.stop
        final_FEvals = round(fevals.iloc[-1], 3)
        final_time = round(times[-1], 3)

        # Plot FEvals/Time
        plt.subplot(1, 2, 1)
        plt.plot(times, fevals, label=algorithm_name)
        plt.annotate(f'({final_time}, {final_FEvals})', (final_time, final_FEvals), textcoords="offset points", xytext=(0,0), ha='center')
        plt.title('FEvals/Time')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Function Evaluations')

        # Plot FEvals/Iterations
        plt.subplot(1, 2, 2)
        plt.plot(iterations, fevals, label=algorithm_name)
        plt.annotate(f'({final_iterations}, {final_FEvals})', (final_iterations, final_FEvals), textcoords="offset points", xytext=(0,0), ha='center')
        plt.title('FEvals/Iterations')
        plt.xlabel('Iterations')
        
        # Add an overall title
        plt.suptitle(f'{algorithm_name}')
        plt.tight_layout()
        plt.savefig(f'Images/NQP/{algorithm_name} FEvals-{num_queens}')
        plt.close()

    # Plot the final positions of the board for each algorithm
    plt.figure(figsize=(8, 8))
    for i, (algorithm_name, result) in enumerate(results_dict.items(), 1):
        plt.subplot(2, 2, i)
        plot_board(result['Best_State'], num_queens, f'{algorithm_name} - Final Position', round(results_dict[algorithm_name]['Best_Fitness'], 2))

    plt.tight_layout()
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, hspace=0.3, wspace=0.2)

    plt.savefig(f'Images/NQP/NQP Final Positions-{num_queens}.png')
    plt.close()