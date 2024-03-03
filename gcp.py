import networkx as nx
import matplotlib.pyplot as plt

def display_graph(problem, pop_size):

    # Create a graph from the Max K-Color problem
    graph = nx.Graph()
    for edge in problem.source_graph.edges:
        graph.add_edge(*edge)

    # Visualize the graph
    pos = nx.spring_layout(graph)
    nx.draw(graph, pos, with_labels=True, node_color='skyblue', node_size=800, font_size=10, font_color='black', font_weight='bold', edge_color='black', linewidths=1, alpha=0.7)

    # Display the plot
    plt.title(f'Graph Layout - Population Size = {pop_size}')
    plt.savefig(f'Images/GCP/GCP Initial State-{pop_size}.png')
    plt.close()

def plot_graph_subplot(ax, G, node_colors, title, fitness):
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, font_weight='bold', node_color=node_colors, cmap=plt.cm.rainbow, ax=ax)
    ax.set_title(f'{title}\nFitness: {fitness}')

def display_results(results_dict, graph, num_nodes):
    
    # Plot fitness curves for each algorithm
    for algorithm_name, results in results_dict.items():
        fitness_curve = results['Fitness_Curve']
        iterations = range(1, len(fitness_curve) + 1)
        
        plt.plot(iterations, fitness_curve, label=algorithm_name)

    # Add labels, title, and legend
    plt.xlabel('Iteration')
    plt.ylabel('Fitness Value')
    plt.title('Graph Coloring - Fitness Curve')
    plt.legend()
    plt.savefig(f'Images/GCP/GCP Fitness Curve-{num_nodes}.png')
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
        plt.savefig(f'Images/GCP/{algorithm_name} FEvals-{num_nodes}')
        plt.close()

    # Create subplots
    num_algorithms = len(results_dict)
    fig, axes = plt.subplots(1, num_algorithms, figsize=(15, 5))

    # Visualize the best state for each algorithm
    for i, (algorithm_name, result) in enumerate(results_dict.items()):
        best_state = result['Best_State']
        fitness_value = round(result['Best_Fitness'], 2)
        plot_graph_subplot(axes[i], graph, best_state, f'Best State - {algorithm_name}', fitness_value)

    # Adjust layout
    plt.tight_layout()
    plt.savefig(f'Images/GCP/GCP Final Solutions-{num_nodes}.png')
    plt.close()