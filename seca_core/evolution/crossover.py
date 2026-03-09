"""
Crossover Operator for SECA

This module combines two parent genomes to produce a child genome.
"""

import random
from seca_core.evolution.genome import clone_genome


def crossover(parent1, parent2):
    """
    Perform crossover between two genomes.

    Parameters
    ----------
    parent1 : dict
    parent2 : dict

    Returns
    -------
    child : dict
    """

    p1 = clone_genome(parent1)
    p2 = clone_genome(parent2)

    child = {}

    # Combine dense layer
    child["dense"] = random.choice([p1["dense"], p2["dense"]])

    # Combine dropout
    child["dropout"] = random.choice([p1["dropout"], p2["dropout"]])

    # Combine convolution stages
    stages1 = p1["stages"]
    stages2 = p2["stages"]

    split1 = random.randint(0, len(stages1) - 1)
    split2 = random.randint(0, len(stages2) - 1)

    child_stages = stages1[:split1] + stages2[split2:]

    # Ensure at least one stage
    if len(child_stages) == 0:
        child_stages = [random.choice(stages1)]

    child["stages"] = child_stages

    return child