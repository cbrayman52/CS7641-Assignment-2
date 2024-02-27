import mlrose_hiive
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def run_algorithms(x_train, y_train, x_test, y_test):
    
    grid_search_parameters = ({
            'max_iters': [1, 2, 4, 8, 16, 32, 64, 128],                     # nn params
            'learning_rate': [0.001, 0.002, 0.003],                         # nn params
            'schedule': [mlrose_hiive.ArithDecay(1), mlrose_hiive.ArithDecay(100), mlrose_hiive.ArithDecay(1000)]  # sa params
    })

    nnr = mlrose_hiive.NNGSRunner(x_train=x_train,
                    y_train=y_train,
                    x_test=x_test,
                    y_test=y_test,
                    experiment_name='RHC',
                    output_directory= 'Output/NN',
                    algorithm=mlrose_hiive.algorithms.sa.simulated_annealing,
                    grid_search_parameters=grid_search_parameters,
                    iteration_list=[1, 10, 50, 100, 250, 500, 1000, 2500, 5000, 10000],
                    hidden_layer_sizes=[[44,44]],
                    bias=True,
                    early_stopping=False,
                    clip_max=1e+10,
                    max_attempts=500,
                    generate_curves=True,
                    seed=4)

    results = nnr.run()