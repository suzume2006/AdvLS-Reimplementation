"""
Pure genetic-algorithm operators for AdvLS -- no model or image libraries
needed here, just the crossover/mutation/selection math. Kept separate from
step4_ga.py so these can be tested without needing torch/torchvision installed.
"""

import random

CROSSOVER_PROB = 0.7    # P_c from the paper
MUTATION_PROB = 0.1     # P_m from the paper
REPLACE_FRACTION = 0.1  # replace worst 10% with best 10%, per the paper's selection strategy


def mutate(vector, image_width, image_height):
    """Randomly perturb each value in the chromosome with probability MUTATION_PROB."""
    new_vector = list(vector)
    for i in range(0, len(new_vector), 5):
        if random.random() < MUTATION_PROB:
            new_vector[i] = random.randint(0, image_width - 1)      # m
        if random.random() < MUTATION_PROB:
            new_vector[i + 1] = random.randint(0, image_height - 1) # n
        if random.random() < MUTATION_PROB:
            new_vector[i + 2] = random.randint(0, 255)              # r
        if random.random() < MUTATION_PROB:
            new_vector[i + 3] = random.randint(0, 255)              # g
        if random.random() < MUTATION_PROB:
            new_vector[i + 4] = random.randint(0, 255)              # b
    return new_vector


def crossover(vector_a, vector_b):
    """With probability CROSSOVER_PROB, swap each spot (5-value chunk) between
    two parent chromosomes to produce one child."""
    child = list(vector_a)
    for i in range(0, len(child), 5):
        if random.random() < CROSSOVER_PROB:
            child[i:i + 5] = vector_b[i:i + 5]
    return child


def select_and_replace(fitness_scores, population):
    """
    fitness_scores: list of (confidence, index) tuples, one per individual.
    population: list of chromosomes (flat vectors), modified in place.
    Replaces the worst REPLACE_FRACTION of individuals with copies of the
    best REPLACE_FRACTION -- the paper's selection strategy (Section 3.4).
    """
    sorted_scores = sorted(fitness_scores, key=lambda t: t[0])  # ascending: lowest confidence = most adversarial = best
    population_size = len(population)
    num_replace = max(1, int(population_size * REPLACE_FRACTION))

    best_indices = [idx for _, idx in sorted_scores[:num_replace]]
    worst_indices = [idx for _, idx in sorted_scores[-num_replace:]]

    for worst_i, best_i in zip(worst_indices, best_indices):
        population[worst_i] = list(population[best_i])

    return population
