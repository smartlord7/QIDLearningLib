#!/usr/bin/env python3
"""
A modular evolutionary algorithm for QID selection where the user
can supply the metric functions and coefficients, as well as a custom
fitness aggregator. The EA prints and plots the evolution of all metrics.
"""

import concurrent.futures
import random
import logging
import numpy as np
import pandas as pd
import matplotlib

from optimizer.metric import Metric
from optimizer.util import load_data_from_folder

matplotlib.use('TkAgg')  # Set the backend before importing pyplot
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')



# =============================================================================
# Evolutionary Algorithm Class
# =============================================================================

class EvolutionaryAlgorithm:
    """
    The evolutionary algorithm (EA) class.
    Users supply:
       - data: a pandas DataFrame,
       - metrics: a list of Metric objects (each with its own coefficient and compute function),
       - alpha: a scaling parameter,
       - population and generation parameters,
       - initial mutation/crossover rates, etc.

    Optionally, you can supply your own aggregator function that computes the fitness from
    a dictionary of metric values. The default aggregator sums (coefficient * metric value)
    for each metric and then divides by (alpha * penalty) if an "Attribute Length Penalty"
    metric is present.
    """

    def __init__(self, data, metrics, alpha=1.0, population_size=50, generations=30,
                 initial_mutation_rate=0.2, initial_crossover_rate=0.3, elite_size=1,
                 tournament_size=5, aggregator=None, interactive_plot=True):
        self.data = data
        self.metrics = metrics
        self.alpha = alpha
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = initial_mutation_rate
        self.crossover_rate = initial_crossover_rate
        self.elite_size = elite_size
        self.tournament_size = tournament_size
        self.interactive_plot = interactive_plot
        self.num_features = data.shape[1]
        logging.info(f"EA initialized: population_size={population_size}, generations={generations}, "
                     f"alpha={alpha}, mutation_rate={initial_mutation_rate}, crossover_rate={initial_crossover_rate}")

        # Define the default aggregator if none was supplied.
        if aggregator is None:
            def default_aggregator(metric_values, individual, data, alpha):
                total = 0.0
                for m in self.metrics:
                    total += m.coefficient * metric_values[m.name]
                penalty = metric_values.get("Attribute Length Penalty", 1.0)
                fitness = total / (alpha * penalty)
                logging.info(f"Aggregator: Metric values: {metric_values}, Total: {total}, Penalty: {penalty}, "
                             f"Fitness: {fitness}")
                return fitness

            self.aggregator = default_aggregator
        else:
            self.aggregator = aggregator

    # --- EA Operator Functions (as static or instance methods) ---

    @staticmethod
    def recombine(parent1, parent2, crossover_rate):
        """
        One-point crossover. With probability crossover_rate a crossover is performed.
        """
        if random.random() < crossover_rate:
            point = random.randint(1, len(parent1) - 1)
            child1 = np.concatenate((parent1[:point], parent2[point:]))
            child2 = np.concatenate((parent2[:point], parent1[point:]))
            logging.info(f"Recombination: Crossover performed at point {point}")
            return child1, child2
        logging.info("Recombination: Crossover not performed, returning copies of parents")
        return parent1.copy(), parent2.copy()

    @staticmethod
    def mutate(individual, mutation_rate):
        """
        Bit-flip mutation.
        """
        original = individual.copy()
        for i in range(len(individual)):
            if random.random() < mutation_rate:
                individual[i] = 1 - individual[i]
        logging.info(f"Mutation: Before: {original}, After: {individual}")
        return individual

    def tournament_selection(self, population, fitnesses):
        """
        Tournament selection from the population.
        """
        tournament = random.sample(list(zip(population, fitnesses)), self.tournament_size)
        winner = max(tournament, key=lambda x: x[1])
        logging.info(f"Tournament Selection: Participants: {[f for _, f in tournament]}, Winner fitness: {winner[1]}")
        return winner[0]

    def elitism(self, population, fitnesses):
        """
        Select the top elite_size individuals.
        """
        sorted_indices = np.argsort(fitnesses)[-self.elite_size:]
        elite = [population[i] for i in sorted_indices]
        logging.info(f"Elitism: Selected elite individuals with fitness values: {[fitnesses[i] for i in sorted_indices]}")
        return elite

    def evaluate_individual(self, individual, best_individual):
        """
        Compute all metric values and return the fitness (as computed by the aggregator)
        along with the metric values.
        """
        metric_values = {}
        for m in self.metrics:
            metric_values[m.name] = m.compute(individual, best_individual, self.data)
        fitness = self.aggregator(metric_values, individual, self.data, self.alpha)
        logging.info(f"Evaluated individual: Fitness: {fitness}, Metrics: {metric_values}")
        return fitness, metric_values

    def run(self):
        """
        Run the evolutionary algorithm and return the best individual, its fitness, and
        a history dictionary (containing evolution of fitness and all metrics).
        """
        logging.info("Starting EA run...")
        # Initialize a random binary population.
        population = [np.random.randint(0, 2, self.num_features) for _ in range(self.population_size)]
        logging.info(f"Initial population of {self.population_size} individuals created.")
        best_individual = None
        best_fitness = float('-inf')
        history = {
            "fitness_mean": [],
            "fitness_best": [],
            "metrics": {m.name: {"mean": [], "best": []} for m in self.metrics}
        }

        # --- Set up interactive plotting if enabled ---
        if self.interactive_plot:
            num_metrics = len(self.metrics)
            num_columns = 2
            num_rows = (num_metrics + num_columns - 1) // num_columns
            fig, axs = plt.subplots(num_rows, num_columns, figsize=(15, 10))
            if num_rows == 1:
                axs = np.expand_dims(axs, axis=0)
            elif num_columns == 1:
                axs = np.expand_dims(axs, axis=1)
            for i, m in enumerate(self.metrics):
                row, col = i // num_columns, i % num_columns
                axs[row, col].set_title(m.name)
                axs[row, col].set_xlabel("Generation")
                axs[row, col].set_ylabel("Value")
                axs[row, col].grid(True)
            fig_fitness = plt.figure(figsize=(8, 6))
            ax_fitness = fig_fitness.add_subplot()
            ax_fitness.set_title("Fitness Evolution")
            ax_fitness.set_xlabel("Generation")
            ax_fitness.set_ylabel("Fitness")
            ax_fitness.grid(True)
            plt.ion()
            plt.tight_layout()
            logging.info("Interactive plotting initialized.")

        # --- Evolution Loop ---
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.population_size) as executor:
            prev_best_fitness = float('-inf')
            for gen in range(self.generations):
                logging.info(f"--- Generation {gen} ---")
                # Evaluate the fitness of all individuals concurrently.
                futures = {executor.submit(self.evaluate_individual, ind, best_individual): ind
                           for ind in population}
                results = [future.result() for future in concurrent.futures.as_completed(futures)]
                fitnesses = [res[0] for res in results]
                metric_values_list = [res[1] for res in results]

                mean_fitness = np.mean(fitnesses)
                current_best_fitness = max(fitnesses)
                best_idx = np.argmax(fitnesses)
                logging.info(f"Generation {gen}: Mean Fitness = {mean_fitness}, Current Best Fitness = {current_best_fitness}")

                if current_best_fitness > best_fitness:
                    best_fitness = current_best_fitness
                    best_individual = population[best_idx].copy()
                    logging.info(f"New overall best individual found with fitness {best_fitness}")

                history["fitness_mean"].append(mean_fitness)
                history["fitness_best"].append(current_best_fitness)

                # Save metric history.
                for m in self.metrics:
                    values = [mv[m.name] for mv in metric_values_list]
                    mean_metric = np.mean(values)
                    best_metric = max(values) if m.maximize else min(values)
                    history["metrics"][m.name]["mean"].append(mean_metric)
                    history["metrics"][m.name]["best"].append(best_metric)
                    logging.info(f"Generation {gen}: Metric [{m.name}] - Mean: {mean_metric}, Best: {best_metric}")

                # --- Dynamic Adjustment of Rates ---
                if current_best_fitness > prev_best_fitness:
                    self.mutation_rate *= 0.99  # lower mutation if improving
                    self.crossover_rate *= 1.01
                    logging.info("Improvement detected: decreasing mutation rate, increasing crossover rate.")
                else:
                    self.mutation_rate *= 1.01  # increase mutation if not improving
                    self.crossover_rate *= 0.99
                    logging.info("No improvement: increasing mutation rate, decreasing crossover rate.")
                self.mutation_rate = min(max(self.mutation_rate, 0.01), 1.0)
                self.crossover_rate = min(max(self.crossover_rate, 0.01), 1.0)
                logging.info(f"Adjusted rates: Mutation Rate = {self.mutation_rate}, Crossover Rate = {self.crossover_rate}")
                prev_best_fitness = current_best_fitness

                # --- Create Next Generation ---
                elite = self.elitism(population, fitnesses)
                new_population = elite.copy()
                logging.info("Generating offspring for next generation...")
                while len(new_population) < self.population_size:
                    parent1 = self.tournament_selection(population, fitnesses)
                    parent2 = self.tournament_selection(population, fitnesses)
                    child1, child2 = self.recombine(parent1, parent2, self.crossover_rate)
                    child1 = self.mutate(child1, self.mutation_rate)
                    child2 = self.mutate(child2, self.mutation_rate)
                    new_population.extend([child1, child2])
                population = new_population[:self.population_size]
                logging.info(f"Generation {gen} completed. Population updated.")

                # --- Update Interactive Plots ---
                if self.interactive_plot:
                    for i, m in enumerate(self.metrics):
                        row, col = i // num_columns, i % num_columns
                        axs[row, col].cla()
                        axs[row, col].plot(range(gen + 1), history["metrics"][m.name]["mean"],
                                            label="Mean", color="blue", alpha=0.5)
                        axs[row, col].plot(range(gen + 1), history["metrics"][m.name]["best"],
                                            label="Best", color="red", alpha=0.5)
                        axs[row, col].set_title(m.name)
                        axs[row, col].set_xlabel("Generation")
                        axs[row, col].set_ylabel("Value")
                        axs[row, col].grid(True)
                        axs[row, col].legend()
                    ax_fitness.cla()
                    ax_fitness.plot(range(gen + 1), history["fitness_mean"],
                                    label="Mean Fitness", color="blue", alpha=0.5)
                    ax_fitness.plot(range(gen + 1), history["fitness_best"],
                                    label="Best Fitness", color="red", alpha=0.5)
                    ax_fitness.set_title("Fitness Evolution")
                    ax_fitness.set_xlabel("Generation")
                    ax_fitness.set_ylabel("Fitness")
                    ax_fitness.grid(True)
                    ax_fitness.legend()
                    plt.pause(0.01)
            if self.interactive_plot:
                plt.ioff()
                plt.show()
        logging.info("EA run completed.")
        return best_individual, best_fitness, history


# =============================================================================
# Post-run Plotting Function
# =============================================================================

def plot_evolution(history, metrics, generations):
    """
    Plot the evolution of all metrics and the fitness over generations.
    """
    logging.info("Plotting evolution of metrics and fitness.")
    num_metrics = len(metrics)
    num_columns = 2
    num_rows = (num_metrics + num_columns - 1) // num_columns
    fig, axs = plt.subplots(num_rows, num_columns, figsize=(15, 10))
    if num_rows == 1:
        axs = np.expand_dims(axs, axis=0)
    elif num_columns == 1:
        axs = np.expand_dims(axs, axis=1)
    for i, m in enumerate(metrics):
        row, col = i // num_columns, i % num_columns
        axs[row, col].plot(range(generations), history["metrics"][m.name]["mean"], label="Mean")
        axs[row, col].plot(range(generations), history["metrics"][m.name]["best"], label="Best")
        axs[row, col].set_title(m.name)
        axs[row, col].set_xlabel("Generation")
        axs[row, col].set_ylabel("Value")
        axs[row, col].grid(True)
        axs[row, col].legend()
    plt.tight_layout()
    plt.show()

    # Plot the fitness evolution separately.
    plt.figure(figsize=(8, 6))
    plt.plot(range(generations), history["fitness_mean"], label="Mean Fitness", color="blue", alpha=0.5)
    plt.plot(range(generations), history["fitness_best"], label="Best Fitness", color="red", alpha=0.5)
    plt.title("Fitness Evolution")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.grid(True)
    plt.legend()
    plt.show()


# =============================================================================
# Main Process: Running the EA on All Datasets
# =============================================================================

def process():
    folder_path = '../../../datasets'
    logging.info(f"Process started. Loading datasets from {folder_path}")
    data_files, headers = load_data_from_folder(folder_path)
    results = []

    # -----------------------------
    # Define the metrics (user-specified)
    # -----------------------------
    metrics = [
        Metric("Distinction", 5, Metric.compute_distinction, maximize=True),
        Metric("Separation", 0.5, Metric.compute_separation, maximize=True),
        Metric("k-Anonymity", -0.4, Metric.compute_k_anonymity, maximize=True),
        Metric("Delta Distinction", 0.2, Metric.compute_delta_distinction, maximize=True),
        Metric("Delta Separation", 0.2, Metric.compute_delta_separation, maximize=True),
        Metric("Attribute Length Penalty", -1, Metric.compute_attribute_length_penalty, maximize=True)
    ]

    for i, (file_name, df) in enumerate(data_files):
        logging.info(f"Processing dataset: {file_name}, shape: {df.shape}")
        header = headers[i][1]

        # Create and run the EA instance for this dataset.
        ea = EvolutionaryAlgorithm(df, metrics, alpha=5, population_size=50, generations=30,
                                   initial_crossover_rate=0.3, initial_mutation_rate=0.2,
                                   elite_size=1, tournament_size=5, interactive_plot=True)
        best_individual, best_fitness, history = ea.run()
        selected_indices = np.where(best_individual == 1)[0]
        logging.info(f"Dataset {file_name}: Best individual selected indices: {selected_indices}")

        # Compute final metric values for the best individual.
        best_metrics = {}
        for m in metrics:
            best_metrics[m.name] = m.compute(best_individual, None, df)
        logging.info(f"Dataset {file_name}: Final metrics for best individual: {best_metrics}")

        results.append({
            "file_name": file_name,
            "header": header,
            "best_attributes": df.columns[selected_indices].tolist(),
            "best_fitness": best_fitness,
            **best_metrics
        })

        # Optionally, plot the evolution of metrics for this dataset.
        plot_evolution(history, metrics, ea.generations)

    # Save aggregated results to CSV.
    df_results = pd.DataFrame(results)
    df_results.to_csv('aggregated_results.csv', index=False)
    logging.info("Results saved to 'aggregated_results.csv'")


def main():
    logging.info("Starting main process")
    process()
    logging.info("Main process completed")


if __name__ == '__main__':
    main()
