import numpy as np
import matplotlib.pyplot as plt
import logging
import math
from functools import partial

import pandas as pd

from QIDLearningLib.optimizer.metric import Metric
from QIDLearningLib.optimizer.util import load_data_from_folder

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

class SimulatedAnnealing:
    def __init__(self, data, metrics, alpha=1.0, max_iterations=100,
                 initial_temp=10.0, cooling_rate=0.99, aggregator=None,
                 interactive_plot=False, plot_update_interval=5):
        self.data = data
        self.metrics = metrics
        self.alpha = alpha
        self.max_iterations = max_iterations
        self.temperature = initial_temp
        self.cooling_rate = cooling_rate
        self.interactive_plot = interactive_plot
        self.plot_update_interval = plot_update_interval
        self.num_features = data.shape[1]
        logging.info(f"SimulatedAnnealing initialized: max_iterations={max_iterations}, alpha={alpha}")

        if aggregator is None:
            self.aggregator = partial(Metric.default_aggregator, metrics=self.metrics)
        else:
            self.aggregator = aggregator

    def evaluate_individual(self, individual):
        """Compute metric values and fitness for an individual."""
        metric_values = {m.name: m.compute(individual, None, self.data) for m in self.metrics}
        fitness = self.aggregator(metric_values, individual, self.data, self.alpha)
        return fitness, metric_values

    def get_neighbor(self, candidate):
        """
        Generate a neighbor by flipping one random bit.
        It allows both adding (0 → 1) and removing (1 → 0) attributes,
        ensuring a more balanced search space exploration.
        """
        neighbor = candidate.copy()

        # Choose a random index to flip
        idx = np.random.randint(0, len(candidate))
        neighbor[idx] = 1 - neighbor[idx]

        return neighbor

    def run(self):
        """Perform the Simulated Annealing search."""
        logging.info("Starting Simulated Annealing...")
        current_candidate = np.random.randint(0, 2, self.num_features)
        current_fitness, current_metrics = self.evaluate_individual(current_candidate)

        history = {
            "fitness": [current_fitness],
            "metrics": {m.name: [current_metrics[m.name]] for m in self.metrics}
        }

        if self.interactive_plot:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.set_title("Simulated Annealing Fitness Evolution")
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Fitness")
            ax.grid(True)
            plt.ion()
            plt.tight_layout()

        for iteration in range(self.max_iterations):
            neighbor = self.get_neighbor(current_candidate)
            neighbor_fitness, neighbor_metrics = self.evaluate_individual(neighbor)

            # Decide whether to accept the neighbor
            if neighbor_fitness > current_fitness:
                accept = True
            else:
                probability = math.exp((neighbor_fitness - current_fitness) / self.temperature)
                accept = np.random.rand() < probability

            if accept:
                current_candidate = neighbor
                current_fitness = neighbor_fitness
                current_metrics = neighbor_metrics
                logging.info(f"Iteration {iteration}: Accepted new solution with fitness {current_fitness}")

            history["fitness"].append(current_fitness)
            for m in self.metrics:
                history["metrics"][m.name].append(current_metrics[m.name])

            # Decrease temperature
            self.temperature *= self.cooling_rate

            # Update interactive plot
            if self.interactive_plot and (iteration % self.plot_update_interval == 0):
                ax.cla()
                ax.plot(range(len(history["fitness"])), history["fitness"], label="Fitness", color="blue")
                ax.set_title("Simulated Annealing Fitness Evolution")
                ax.set_xlabel("Iteration")
                ax.set_ylabel("Fitness")
                ax.grid(True)
                ax.legend()
                plt.pause(0.01)

        if self.interactive_plot:
            plt.ioff()
            plt.show()

        logging.info("Simulated Annealing completed.")
        return current_candidate, current_fitness, history

# =============================================================================
# Main Process: Running the Greedy Search on All Datasets
# =============================================================================

def process():
    """
    Process function for running Simulated Annealing on all datasets.
    """
    folder_path = '../../../../datasets'
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

        # Create and run the Simulated Annealing instance for this dataset.
        sa = SimulatedAnnealing(
            data=df,
            metrics=metrics,
            alpha=5,
            max_iterations=100,
            initial_temp=10.0,
            cooling_rate=0.95,
            interactive_plot=True
        )
        best_individual, best_fitness, history = sa.run()

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


    df_results = pd.DataFrame(results)
    df_results.to_csv('aggregated_results_sa.csv', index=False)
    logging.info("Results saved to 'aggregated_results_sa.csv'")


def main():
    logging.info("Starting main process")
    process()
    logging.info("Main process completed")


if __name__ == '__main__':
    main()