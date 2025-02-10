import concurrent.futures
import logging
import numpy as np
import pandas as pd
import matplotlib
from functools import partial

from optimizer.metric import Metric
from optimizer.util import load_data_from_folder

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
matplotlib.use('TkAgg')  # Set the backend before importing pyplot
import matplotlib.pyplot as plt

def default_greedy_aggregator(metric_values, individual, data, alpha, metrics):
    """
    A default aggregator function that combines the metric values.
    It multiplies each metric value by its coefficient, sums the results,
    and then divides by (alpha * penalty) where the penalty is taken from
    the 'Attribute Length Penalty' metric (or 1.0 if not present).
    """
    total = 0.0
    for m in metrics:
        total += m.coefficient * metric_values[m.name]
    penalty = metric_values.get("Attribute Length Penalty", 1.0)
    fitness = total / (alpha * penalty)
    return fitness

# =============================================================================
# Greedy Search (Hill-Climbing) Algorithm Class
# =============================================================================

class GreedySearch:
    """
    A greedy (hill-climbing) search algorithm for selecting QIDs.

    Users supply:
      - data: a pandas DataFrame,
      - metrics: a list of Metric objects,
      - alpha: a scaling parameter,
      - max_iterations: maximum number of iterations,
      - aggregator: an optional function to combine metric values into a fitness.

    Optionally, interactive plotting can be enabled.
    """

    def __init__(self, data, metrics, alpha=1.0, max_iterations=100,
                 aggregator=None, interactive_plot=False, plot_update_interval=5):
        self.data = data
        self.metrics = metrics
        self.alpha = alpha
        self.max_iterations = max_iterations
        self.interactive_plot = interactive_plot
        self.plot_update_interval = plot_update_interval
        self.num_features = data.shape[1]
        logging.info(f"GreedySearch initialized: max_iterations={max_iterations}, alpha={alpha}")

        # Use the top-level aggregator via functools.partial so it is picklable.
        if aggregator is None:
            self.aggregator = partial(default_greedy_aggregator, metrics=self.metrics)
        else:
            self.aggregator = aggregator

    def evaluate_individual(self, individual, best_individual):
        """
        Compute all metric values for an individual and combine them into a fitness.
        """
        metric_values = {}
        for m in self.metrics:
            metric_values[m.name] = m.compute(individual, best_individual, self.data)
        fitness = self.aggregator(metric_values, individual, self.data, self.alpha)
        return fitness, metric_values

    def get_neighbors(self, candidate):
        """
        Generate all neighbors (binary strings) with a Hamming distance of 1
        from the given candidate by flipping each bit.
        """
        return [np.where(np.arange(len(candidate)) == i, 1 - candidate[i], candidate)
                for i in range(len(candidate))]
    def run(self):
        """
        Run the greedy (hill-climbing) search and return the best solution,
        its fitness, and a history dictionary.
        """
        logging.info("Starting GreedySearch run...")
        # Initialize with a random candidate solution.
        current_candidate = np.random.randint(0, 2, self.num_features)
        current_fitness, current_metrics = self.evaluate_individual(current_candidate, None)
        logging.info(f"Initial candidate fitness: {current_fitness}")

        history = {
            "fitness": [current_fitness],
            "metrics": {m.name: [current_metrics[m.name]] for m in self.metrics}
        }

        # Set up interactive plotting if enabled.
        if self.interactive_plot:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.set_title("Greedy Search Fitness Evolution")
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Fitness")
            ax.grid(True)
            # Create a line object with empty data.
            fitness_line, = ax.plot([], [], label="Fitness", color="blue")
            ax.legend()
            plt.ion()  # Turn on interactive mode.
            plt.show()

        iteration = 0
        improvement = True

        # Use ProcessPoolExecutor for neighbor evaluations.
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.num_features) as executor:
            while improvement and iteration < self.max_iterations:
                iteration += 1
                logging.info(f"Iteration {iteration} starting...")
                neighbors = self.get_neighbors(current_candidate)

                # Evaluate all neighbors concurrently.
                futures = {executor.submit(self.evaluate_individual, neighbor, current_candidate): neighbor
                           for neighbor in neighbors}
                results = [future.result() for future in concurrent.futures.as_completed(futures)]
                neighbor_fitnesses = [res[0] for res in results]
                neighbor_metrics_list = [res[1] for res in results]

                best_neighbor_idx = np.argmax(neighbor_fitnesses)
                best_neighbor_fitness = neighbor_fitnesses[best_neighbor_idx]
                best_neighbor = neighbors[best_neighbor_idx]
                best_neighbor_metrics = neighbor_metrics_list[best_neighbor_idx]

                logging.info(f"Iteration {iteration}: Best neighbor fitness: {best_neighbor_fitness}")

                if best_neighbor_fitness > current_fitness:
                    current_candidate = best_neighbor.copy()
                    current_fitness = best_neighbor_fitness
                    current_metrics = best_neighbor_metrics
                    logging.info(f"Iteration {iteration}: Found improvement. New fitness: {current_fitness}")
                else:
                    logging.info(f"Iteration {iteration}: No improvement found. Terminating search.")
                    improvement = False

                history["fitness"].append(current_fitness)
                for m in self.metrics:
                    history["metrics"][m.name].append(current_metrics[m.name])

                # Update interactive plot if enabled.
                if self.interactive_plot and (iteration % self.plot_update_interval == 0 or iteration == self.max_iterations):
                    # Update the line data rather than clearing the axes.
                    fitness_line.set_data(range(len(history["fitness"])), history["fitness"])
                    ax.relim()            # Recompute the data limits.
                    ax.autoscale_view()   # Update the view.
                    fig.canvas.draw_idle()
                    fig.canvas.flush_events()

        if self.interactive_plot:
            plt.ioff()  # Turn interactive mode off.
            plt.show()  # Keep the window open.
        logging.info("GreedySearch run completed.")
        return current_candidate, current_fitness, history

# =============================================================================
# Post-run Plotting Function
# =============================================================================

def plot_evolution(history, metrics, iterations):
    """
    Plot the evolution of metrics and fitness over the iterations.
    """
    logging.info("Plotting evolution for GreedySearch.")
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
        axs[row, col].plot(range(iterations + 1), history["metrics"][m.name], label="Metric Value")
        axs[row, col].set_title(m.name)
        axs[row, col].set_xlabel("Iteration")
        axs[row, col].set_ylabel("Value")
        axs[row, col].grid(True)
        axs[row, col].legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.plot(range(iterations + 1), history["fitness"], label="Fitness", color="blue", alpha=0.8)
    plt.title("Fitness Evolution")
    plt.xlabel("Iteration")
    plt.ylabel("Fitness")
    plt.grid(True)
    plt.legend()
    plt.show()


# =============================================================================
# Main Process: Running the Greedy Search on All Datasets
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
        Metric("Distinction", 1, Metric.compute_distinction, maximize=True),
        Metric("Separation", 0.5, Metric.compute_separation, maximize=True),
        Metric("k-Anonymity", -0.4, Metric.compute_k_anonymity, maximize=True),
        Metric("Delta Distinction", 0.2, Metric.compute_delta_distinction, maximize=True),
        Metric("Delta Separation", 0.2, Metric.compute_delta_separation, maximize=True),
        Metric("Attribute Length Penalty", -2, Metric.compute_attribute_length_penalty, maximize=True)
    ]

    for i, (file_name, df) in enumerate(data_files):
        logging.info(f"Processing dataset: {file_name}, shape: {df.shape}")
        header = headers[i][1]

        # Create and run the GreedySearch instance for this dataset.
        gs = GreedySearch(df, metrics, alpha=5, max_iterations=100,
                          interactive_plot=True, plot_update_interval=5)
        best_individual, best_fitness, history = gs.run()
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

        # Optionally, plot evolution for this dataset.
        plot_evolution(history, metrics, len(history["fitness"]) - 1)

    df_results = pd.DataFrame(results)
    df_results.to_csv('aggregated_results_greedy.csv', index=False)
    logging.info("Results saved to 'aggregated_results_greedy.csv'")


def main():
    logging.info("Starting main process")
    process()
    logging.info("Main process completed")


if __name__ == '__main__':
    main()