import logging
import matplotlib.pyplot as plt
import numpy as np


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
