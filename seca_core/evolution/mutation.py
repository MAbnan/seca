"""
Mutation Operator for SECA

This module mutates neural architecture genomes to introduce variation.
"""

import random
from seca_core.evolution.genome import clone_genome

FILTER_OPTIONS = [16, 32, 48, 64]
KERNEL_OPTIONS = [3, 5]
POOL_OPTIONS = [1, 2]

DENSE_OPTIONS = [32, 64, 128]
DROPOUT_OPTIONS = [0.0, 0.2, 0.4]


def mutate(genome):
    """
    Apply mutation to a genome.

    Parameters
    ----------
    genome : dict

    Returns
    -------
    mutated_genome : dict
    """

    g = clone_genome(genome)

    mutation_type = random.choice([
        "filters",
        "kernel",
        "pool",
        "dense",
        "dropout",
        "add_stage",
        "remove_stage"
    ])

    # mutate convolution filters
    if mutation_type == "filters":
        stage = random.choice(g["stages"])
        stage["filters"] = random.choice(FILTER_OPTIONS)

    # mutate kernel size
    elif mutation_type == "kernel":
        stage = random.choice(g["stages"])
        stage["kernel"] = random.choice(KERNEL_OPTIONS)

    # mutate pooling
    elif mutation_type == "pool":
        stage = random.choice(g["stages"])
        stage["pool"] = random.choice(POOL_OPTIONS)

    # mutate dense units
    elif mutation_type == "dense":
        g["dense"] = random.choice(DENSE_OPTIONS)

    # mutate dropout
    elif mutation_type == "dropout":
        g["dropout"] = random.choice(DROPOUT_OPTIONS)

    # add convolution stage
    elif mutation_type == "add_stage":

        new_stage = {
            "filters": random.choice(FILTER_OPTIONS),
            "kernel": random.choice(KERNEL_OPTIONS),
            "pool": random.choice(POOL_OPTIONS)
        }

        g["stages"].append(new_stage)

    # remove convolution stage
    elif mutation_type == "remove_stage":

        if len(g["stages"]) > 1:
            idx = random.randrange(len(g["stages"]))
            g["stages"].pop(idx)

    return g