
import concurrent.futures
from functools import partial
import matplotlib
matplotlib.use('TkAgg')  # Set backend before importing pyplot
import matplotlib.pyplot as plt
import logging
import pandas as pd
import numpy as np
from QIDLearningLib.optimizer.metric import Metric
from QIDLearningLib.optimizer.util import load_data_from_folder
from QIDLearningLib.optimizer.graph import plot_evolution  # If you have a common plotting function

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

class TabuSearch:
    """
    A Tabu Search algorithm for selecting QIDs.

    Users supply:
      - data: a pandas DataFrame,
      - metrics: a list of Metric objects,
      - alpha: a scaling parameter,
      - max_iterations: maximum number of iterations,
      - tabu_size: maximum size of the tabu list,
      - aggregator: an optional function to combine metric values into a fitness.

    Optionally, interactive plotting can be enabled.
    """

    def __init__(self, data, metrics, alpha=1.0, max_iterations=100, tabu_size=10,
                 aggregator=None, interactive_plot=False, plot_update_interval=5):
        self.data = data
        self.metrics = metrics
        self.alpha = alpha
        self.max_iterations = max_iterations
        self.tabu_size = tabu_size
        self.interactive_plot = interactive_plot
        self.plot_update_interval = plot_update_interval
        self.num_features = data.shape[1]
        logging.info(f"TabuSearch initialized: max_iterations={max_iterations}, alpha={alpha}, tabu_size={tabu_size}")

        # Use a top-level aggregator via functools.partial (so it is picklable)
        if aggregator is None:
            self.aggregator = partial(Metric.default_aggregator, metrics=self.metrics)
        else:
            self.aggregator = aggregator

    def evaluate_individual(self, individual):
        """
        Compute all metric values for an individual and combine them into a fitness.
        """
        metric_values = {}
        for m in self.metrics:
            metric_values[m.name] = m.compute(individual, None, self.data)
        fitness = self.aggregator(metric_values, individual, self.data, self.alpha)
        return fitness, metric_values

    def get_neighbors(self, candidate):
        """
        Generate all neighbors by flipping each bit in the candidate solution.
        (Each neighbor is at a Hamming distance of 1.)
        """
        neighbors = []
        for i in range(len(candidate)):
            neighbor = candidate.copy()
            neighbor[i] = 1 - neighbor[i]
            neighbors.append(neighbor)
        return neighbors

    def run(self):
        """
        Run the Tabu Search and return the best solution, its fitness, and a history dictionary.
        """
        logging.info("Starting TabuSearch run...")
        # Initialize with a random candidate solution.
        current_candidate = np.random.randint(0, 2, self.num_features)
        current_fitness, current_metrics = self.evaluate_individual(current_candidate)
        logging.info(f"Initial candidate fitness: {current_fitness}")
        history = {
            "fitness": [current_fitness],
            "metrics": {m.name: [current_metrics[m.name]] for m in self.metrics}
        }
        # Initialize the tabu list with the current candidate (using tuple representation)
        tabu_list = [tuple(current_candidate)]

        # Set up interactive plotting if enabled.
        if self.interactive_plot:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.set_title("Tabu Search Fitness Evolution")
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Fitness")
            ax.grid(True)
            # Create a persistent line object.
            fitness_line, = ax.plot([], [], label="Fitness", color="blue")
            ax.legend()
            plt.ion()
            plt.show()

        for iteration in range(1, self.max_iterations + 1):
            neighbors = self.get_neighbors(current_candidate)
            # Filter out neighbors that are in the tabu list.
            valid_neighbors = [n for n in neighbors if tuple(n) not in tabu_list]
            if not valid_neighbors:
                logging.info(f"Iteration {iteration}: No valid neighbors found. Terminating search.")
                break

            # Evaluate valid neighbors concurrently.
            with concurrent.futures.ProcessPoolExecutor(max_workers=self.num_features) as executor:
                futures = {executor.submit(self.evaluate_individual, neighbor): neighbor for neighbor in
                           valid_neighbors}
                results = [future.result() for future in concurrent.futures.as_completed(futures)]
            neighbor_fitnesses = [res[0] for res in results]
            neighbor_metrics_list = [res[1] for res in results]
            best_idx = np.argmax(neighbor_fitnesses)
            best_neighbor_fitness = neighbor_fitnesses[best_idx]
            best_neighbor = valid_neighbors[best_idx]
            best_neighbor_metrics = neighbor_metrics_list[best_idx]

            logging.info(f"Iteration {iteration}: Best neighbor fitness: {best_neighbor_fitness}")
            # Accept the best neighbor (even if not an improvement) to keep search moving.
            current_candidate = best_neighbor.copy()
            current_fitness = best_neighbor_fitness
            current_metrics = best_neighbor_metrics
            logging.info(f"Iteration {iteration}: Updated candidate with fitness: {current_fitness}")

            # Update the tabu list.
            tabu_list.append(tuple(current_candidate))
            if len(tabu_list) > self.tabu_size:
                tabu_list.pop(0)

            history["fitness"].append(current_fitness)
            for m in self.metrics:
                history["metrics"][m.name].append(current_metrics[m.name])

            # Update interactive plot if enabled.
            if self.interactive_plot and (
                    iteration % self.plot_update_interval == 0 or iteration == self.max_iterations):
                fitness_line.set_data(range(len(history["fitness"])), history["fitness"])
                ax.relim()
                ax.autoscale_view()
                fig.canvas.draw_idle()
                fig.canvas.flush_events()

        if self.interactive_plot:
            plt.ioff()
            plt.show()
        logging.info("TabuSearch run completed.")
        return current_candidate, current_fitness, history


def process():
    """
    Process function for running Tabu Search on all datasets.
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

        # Create and run the TabuSearch instance for this dataset.
        ts = TabuSearch(
            data=df,
            metrics=metrics,
            alpha=5,
            max_iterations=100,
            tabu_size=10,
            interactive_plot=True,
            plot_update_interval=5
        )
        best_individual, best_fitness, history = ts.run()
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

        # Optionally, plot the evolution for this dataset.
        plot_evolution(history, metrics, len(history["fitness"]) - 1)

    df_results = pd.DataFrame(results)
    df_results.to_csv('aggregated_results_tabu.csv', index=False)
    logging.info("Results saved to 'aggregated_results_tabu.csv'")

def main():
    logging.info("Starting main process")
    process()
    logging.info("Main process completed")


if __name__ == '__main__':
    main()
