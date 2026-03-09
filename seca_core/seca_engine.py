"""
SECA Engine
-----------

Core controller for the Self-Evolving Cognitive Architecture (SECA).

Responsibilities:
1. Initialize architecture population
2. Train architectures
3. Evaluate fitness
4. Apply cognitive self-evaluation
5. Evolve architectures
"""

import random
import time

from seca_core.evolution.population import initialize_population
from seca_core.evolution.selection import select_top
from seca_core.evolution.mutation import mutate
from seca_core.evolution.crossover import crossover

from seca_core.learning.model_builder import build_model
from seca_core.learning.trainer import train_model

from seca_core.evaluation.fitness import compute_fitness

from seca_core.cognition.regulation import regulate_evolution


class SECAEngine:

    def __init__(
        self,
        x_train,
        y_train,
        x_test,
        y_test,
        population_size=6,
        generations=5,
        mutation_rate=0.3
    ):

        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate

    def evaluate_genome(self, genome):

        """Train and evaluate a neural architecture."""

        model = build_model(genome)

        history, val_acc = train_model(
            model,
            self.x_train,
            self.y_train,
            self.x_test,
            self.y_test
        )

        params = model.count_params()

        fitness = compute_fitness(val_acc, params)

        return {
            "genome": genome,
            "accuracy": val_acc,
            "params": params,
            "fitness": fitness
        }

    def run_evolution(self, on_individual=None, on_generation=None):

        """Run the SECA evolutionary process."""

        history = []

        population = initialize_population(self.population_size)

        best_genome = None
        best_stats = None

        for generation in range(self.generations):

            print(f"\n=== Generation {generation} ===")

            scores = []

            for idx, genome in enumerate(population):

                result = self.evaluate_genome(genome)

                scores.append(result)

                print(
                    f"Ind {idx}: "
                    f"acc={result['accuracy']:.4f}, "
                    f"params={result['params']}, "
                    f"fit={result['fitness']:.4f}"
                )

                if on_individual:
                    on_individual(generation, idx, result)

            # sort by fitness
            scores = sorted(scores, key=lambda x: x["fitness"], reverse=True)

            best = scores[0]

            best_genome = best["genome"]
            best_stats = {
                "accuracy": best["accuracy"],
                "params": best["params"],
                "fitness": best["fitness"]
            }

            print(
                f"> Best this gen: "
                f"acc={best_stats['accuracy']:.4f}, "
                f"params={best_stats['params']}"
            )

            history.append(best_stats)

            if on_generation:
                on_generation(generation, best_stats, best_genome)

            # cognitive regulation
            evolve = regulate_evolution(scores)

            if not evolve:
                print("Cognitive regulation: stopping evolution early.")
                break

            # selection
            parents = select_top(scores, k=2)

            new_population = []

            while len(new_population) < self.population_size:

                parent1 = random.choice(parents)["genome"]
                parent2 = random.choice(parents)["genome"]

                child = crossover(parent1, parent2)

                if random.random() < self.mutation_rate:
                    child = mutate(child)

                new_population.append(child)

            population = new_population

        return best_genome, best_stats, history