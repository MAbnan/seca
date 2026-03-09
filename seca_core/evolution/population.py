"""
Population Management for SECA

This module initializes and manages the population of neural architectures.
"""

from seca_core.evolution.genome import random_genome


def initialize_population(population_size):
    """
    Create an initial population of genomes.

    Parameters
    ----------
    population_size : int

    Returns
    -------
    population : list
    """

    population = []

    for _ in range(population_size):
        genome = random_genome()
        population.append(genome)

    return population