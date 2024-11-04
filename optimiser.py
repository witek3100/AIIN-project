import numpy as np
import pygad
import random

from trainer import Trainer

def load_training_params(genes: str):
    params = {
        'random_horizontal_flip': int(genes[0]),
        'random_vertical_flip': int(genes[1]),
        'random_affine': int(genes[2]),
        'color_jitter': int(genes[3]),
        'batch_size': int(np.interp(genes[4], [0, 1], [4, 64])),
        'learning_rate': np.interp(genes[5], [0, 1], [1e-6, 1e-2]),
        'num_conv_layers': int(np.interp(int(genes[6]), [0, 1], [1, 3])),
        'conv_kernels_sizes': [
            int(np.interp(int(genes[7 + i]), [0, 1], [1, 4])) for i in range(3)
        ],
        'pool_kernels_sizes': [
            int(np.interp(int(genes[10 + i]), [0, 1], [2, 4])) for i in range(3)
        ],
        'conv_neurons': [
            int(np.interp(int(genes[13 + i]), [0, 1], [12, 36])) for i in range(3)
        ],
        'paddings': [
            int(np.interp(int(genes[16 + i]), [0, 1], [1, 3])) for i in range(3)
        ],
    }
    print(params)

    return params

def create_solution():
    num_genes = 19
    return [random.uniform(0, 1) for _ in range(num_genes)]

def fitness_function(ga_instance, solution, solution_idx):
    params = load_training_params(solution)

    trainer = Trainer(**params)
    score = trainer.run()

    return score

num_individuals = 10
population = [create_solution() for _ in range(num_individuals)]

ga_instance = pygad.GA(
    num_generations=20,
    num_parents_mating=5,
    fitness_func=fitness_function,
    sol_per_pop=num_individuals,
    initial_population=population
)
ga_instance.run()

solution, solution_fitness, solution_idx = ga_instance.best_solution()
print("Best Solution:", solution)
print("Best Fitness:", solution_fitness)


