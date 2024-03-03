import mlrose_hiive
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import sklearn.metrics as skmt
import matplotlib.pyplot as plt

class CustomNeuralNetwork(mlrose_hiive.NeuralNetwork):
    def __init__(self, hidden_nodes=None,
                 activation='relu',
                 algorithm='random_hill_climb',
                 max_iters=100,
                 bias=False,
                 is_classifier=True,
                 learning_rate=0.1,
                 early_stopping=True,
                 clip_max=10,
                 restarts=0,
                 schedule=mlrose_hiive.GeomDecay(),
                 pop_size=300,
                 mutation_prob=0.4,
                 max_attempts=100,
                 random_state=None,
                 curve=True):
        self.classes_ = [0, 1]
        super().__init__(
            hidden_nodes=hidden_nodes,
            activation=activation,
            algorithm=algorithm,
            max_iters=max_iters,
            bias=bias,
            is_classifier=is_classifier,
            learning_rate=learning_rate,
            early_stopping=early_stopping,
            clip_max=clip_max,
            restarts=restarts,
            schedule=schedule,
            pop_size=pop_size,
            mutation_prob=mutation_prob,
            max_attempts=max_attempts,
            random_state=random_state,
            curve=curve)

def train_nn(mlp, x_train, y_train):

    # Initialize arrays to record metrics during training
    training_accuracies = []
    cross_val_accuracies = []
    loss_values = []

    # Number of epochs
    epochs = 100  # You can adjust this based on your preference

    # Training loop
    for epoch in range(epochs):
        # Split the dataset into training and validation sets
        x_train_epoch, x_val, y_train_epoch, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=epoch)

        # Partial fit the model for one epoch
        mlp.partial_fit(x_train_epoch, y_train_epoch, classes=np.unique(y_train))

        # Training accuracy
        y_train_pred = mlp.predict(x_train_epoch)
        training_accuracy = accuracy_score(y_train_epoch, y_train_pred)
        training_accuracies.append(training_accuracy)

        # Cross-validation accuracy
        y_val_pred = mlp.predict(x_val)
        cross_val_accuracy = accuracy_score(y_val, y_val_pred)
        cross_val_accuracies.append(cross_val_accuracy)

        # Loss during training
        loss_values.append(mlp.loss_)

    # Plotting
    plt.figure(figsize=(8, 6))

    # Plot Training Accuracy
    plt.plot(range(1, epochs + 1), training_accuracies, label='Training Accuracy', linestyle='-')

    # Plot Cross-validation Accuracy
    plt.plot(range(1, epochs + 1), cross_val_accuracies, label='Cross-validation Accuracy', linestyle='-')

    # Plot Loss
    plt.plot(range(1, epochs + 1), loss_values, label='Training Loss', linestyle='-')

    plt.title('Training and Validation Metrics - Backprop')
    plt.xlabel('Epochs')
    plt.ylabel('Metric Value')
    plt.legend()
    plt.grid(True)
    plt.savefig('Images/NN/BackProp.png')
    plt.close()

def plot_loss(algorithm, params, title, x_train, y_train, x_test, y_test):

    nn_accuracy_runner = mlrose_hiive.NNGSRunner(x_train=x_train,
                                        y_train=y_train,
                                        x_test=x_test,
                                        y_test=y_test,
                                        experiment_name=title,
                                        output_directory='Output/NN',
                                        algorithm=algorithm,
                                        grid_search_parameters=params,
                                        grid_search_scorer_method=skmt.balanced_accuracy_score,
                                        iteration_list=[100, 500, 1000],
                                        hidden_layer_sizes=[[10]],
                                        bias=False,
                                        early_stopping=False,
                                        clip_max=100,
                                        max_attempts=100,
                                        n_jobs=-1,
                                        cv=5,
                                        generate_curves=True,
                                        seed=4)
    nn_accuracy_runner.classifier.activation = mlrose_hiive.relu
    nn_accuracy_runner.classifier.max_iters = 1000

    run_stats, curves, cv_results, grid_search_results = nn_accuracy_runner.run()

    # Edit curves for RHC to take the best restart
    if title == 'RHC':
        best_restart_index = run_stats['Fitness'].idxmin()
        best_restart = run_stats.at[best_restart_index, 'current_restart']
        curves = curves[curves['current_restart'] == best_restart]

    if title == 'SA':
        # Define the repeating pattern of replacement values
        replacement_values = ['ArithDecay', 'ExpDecay', 'GeomDecay']
        pattern = replacement_values * (len(cv_results) // len(replacement_values))

        # Assign the repeating pattern to the 'Column' in a circular manner
        cv_results['param_schedule'] = pattern[:len(cv_results)]
        params['schedule'] = ['ArithDecay', 'ExpDecay', 'GeomDecay']    

    # Create Subplots
    fig, axes = plt.subplots(1, len(params), figsize=(10, 5))
    axes[0].set_ylabel('Score', fontsize=12)

    # Loop through each parameter to optimize
    for idx, (key, value) in enumerate(params.items()):

        training_scores = []
        test_scores = []

        # Set the width of the bars
        bar_width = 0.35

        # Set the positions of the bars on the x-axis
        indices = np.arange(len(value))

        for param_value in value:
            # Filter results for the specific parameter value
            filtered_results = cv_results[cv_results['param_'+key] == param_value]
            
            # Get the maximum fitness value for this parameter value
            max_train = filtered_results['mean_train_score'].max()
            max_test = filtered_results['mean_test_score'].max()

            # Append values to the lists
            training_scores.append(max_train)
            test_scores.append(max_test)

            # Annotate the bar with fitness value
            axes[idx].annotate(f'{max_train:.3f}', ((indices - bar_width/2)[value.index(param_value)], max_train),
                textcoords="offset points", xytext=(0,3), ha='center')
            
            axes[idx].annotate(f'{max_test:.3f}', ((indices + bar_width/2)[value.index(param_value)], max_test),
                textcoords="offset points", xytext=(0,3), ha='center')

        # Create the grouped bar chart
        axes[idx].bar(indices - bar_width/2, training_scores, bar_width, label='Training Scores')
        axes[idx].bar(indices + bar_width/2, test_scores, bar_width, label='Test Scores')

        # Adding labels and title
        axes[idx].set_xticks(indices)  # Set ticks to match the number of parameter values
        axes[idx].set_xticklabels(value)  # Set tick labels to parameter values
        axes[idx].set_xlabel(key)
    
    # Add legend only in first plot
    axes[0].legend()

    # Set title for the entire subplot
    fig.suptitle(f'NN Weight Optimization - {title} - Accuracy vs Parameter Values', fontsize=15)

    # Adjust layout for better spacing
    plt.tight_layout()

    # Show the plot
    plt.savefig(f'Images/NN/{title}-Parameter Tuning.png')
    plt.close()

    return run_stats, curves