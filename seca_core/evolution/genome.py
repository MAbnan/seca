"""
Genome Representation for SECA

A genome defines the architecture of a neural network.
Each genome contains:

- number of dense units
- dropout rate
- convolution stages
"""

import random
import copy


# Allowed architecture search space
FILTER_OPTIONS = [16, 32, 48, 64]
KERNEL_OPTIONS = [3, 5]
POOL_OPTIONS = [1, 2]

DENSE_OPTIONS = [32, 64, 128]
DROPOUT_OPTIONS = [0.0, 0.2, 0.4]

MIN_STAGES = 1
MAX_STAGES = 5


def random_stage():
    """Generate a random convolution stage."""

    return {
        "filters": random.choice(FILTER_OPTIONS),
        "kernel": random.choice(KERNEL_OPTIONS),
        "pool": random.choice(POOL_OPTIONS)
    }


def random_genome():
    """Generate a random neural architecture genome."""

    num_stages = random.randint(MIN_STAGES, MAX_STAGES)

    genome = {
        "dense": random.choice(DENSE_OPTIONS),
        "dropout": random.choice(DROPOUT_OPTIONS),
        "stages": [random_stage() for _ in range(num_stages)]
    }

    return genome


def clone_genome(genome):
    """Create a deep copy of a genome."""
    return copy.deepcopy(genome)


def print_genome(genome):
    """Pretty print genome."""

    print("\nGenome Architecture")
    print("-------------------")

    for i, stage in enumerate(genome["stages"]):
        print(
            f"Conv Stage {i}: "
            f"filters={stage['filters']} "
            f"kernel={stage['kernel']} "
            f"pool={stage['pool']}"
        )

    print(f"Dense Units: {genome['dense']}")
    print(f"Dropout: {genome['dropout']}")