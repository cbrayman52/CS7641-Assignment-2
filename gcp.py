import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def display_graph(problem):

    # Create a graph from the Max K-Color problem
    graph = nx.Graph()
    for edge in problem.source_graph.edges:
        graph.add_edge(*edge)

    # Visualize the graph
    pos = nx.spring_layout(graph)
    nx.draw(graph, pos, with_labels=True, node_color='skyblue', node_size=800, font_size=10, font_color='black', font_weight='bold', edge_color='black', linewidths=1, alpha=0.7)

    # Display the plot
    plt.title('Graph Layout')
    plt.savefig('Images/GCP/GCP_Graph_Layout.png')
    plt.close()

def plot_graph_subplot(ax, G, node_colors, title, fitness):
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, font_weight='bold', node_color=node_colors, cmap=plt.cm.rainbow, ax=ax)
    ax.set_title(f"{title}\nFitness: {fitness}")

def display_results(results_dict, graph, num_nodes):
    
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
    plt.savefig('Images/GCP/GCP_Fitness_Curve.png', bbox_inches='tight')
    plt.close()

    # Create subplots
    num_algorithms = len(results_dict)
    fig, axes = plt.subplots(1, num_algorithms, figsize=(15, 5))

    # Visualize the best state for each algorithm
    for i, (algorithm_name, result) in enumerate(results_dict.items()):
        best_state = result['Best_State']
        node_colors = [best_state[node] for node in range(num_nodes)]
        fitness_value = round(result['Best_Fitness'], 2)
        plot_graph_subplot(axes[i], graph, node_colors, f"Best State - {algorithm_name}", fitness_value)

    # Adjust layout
    plt.tight_layout()
    plt.savefig('Images/GCP/GCP_Solutions.png')
    plt.close()