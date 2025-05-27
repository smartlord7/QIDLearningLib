import inspect
import logging
import numpy as np
from QIDLearningLib.metrics import qid_specific as qid
from QIDLearningLib.metrics import data_privacy as pr
from QIDLearningLib.metrics.causality import causal_importance
from QIDLearningLib.structure.grouped_metric import GroupedMetric

# =============================================================================
# Metric Classes and Functions
# =============================================================================

class Metric:
    """Container for a metric, with a function to compute its value."""

    def __init__(self, name, coefficient, compute_func, maximize=True):
        self.name = name
        self.coefficient = coefficient
        self.compute_func = compute_func
        self.maximize = maximize
        logging.info(f"Metric initialized: {self.name}, coefficient: {self.coefficient}, maximize: {self.maximize}")

    def compute(self, individual, best_individual, data, prev_value=None):
        return self._compute_metric_with_logging(individual, best_individual, data, prev_value)

    def _compute_metric_with_logging(self, individual, best_individual, data, prev_value):
        """Compute a metric and log the result."""
        value = self.compute_func(individual, best_individual, data, prev_value)
        logging.info(f"Metric [{self.name}] computed value: {value}")
        return value

    @staticmethod
    def _compute_metric_with_qid(individual, data, qid_function):
        """Generic method for computing QID-based metrics."""
        selected_columns = Metric._get_selected_columns(individual, data)
        if not selected_columns:
            return 0.0
        return qid_function(data, selected_columns) / 100.0

    @staticmethod
    def _compute_metric_with_pr(individual, data, pr_function):
        """Generic method for computing data privacy-based metrics."""
        selected_columns = Metric._get_selected_columns(individual, data)
        if not selected_columns:
            return 0.0

        if 'sensitive_arguments' in inspect.signature(pr_function).parameters:
            result = pr_function(data, selected_columns, data.sensitive_attributes)
        else:
            result = pr_function(data, selected_columns)
        return result.mean if isinstance(result, GroupedMetric) else result

    @staticmethod
    def _get_selected_columns(individual, data):
        """Retrieve selected columns based on a binary individual representation."""
        selected_indices = np.where(individual == 1)[0]
        return data.columns[selected_indices].tolist()

    # ==============================
    # Metric Computation Methods
    # ==============================

    @staticmethod
    def compute_distinction(individual, best_individual, data, prev_value=None):
        return Metric._compute_metric_with_qid(individual, data, qid.distinction)

    @staticmethod
    def compute_separation(individual, best_individual, data, prev_value=None):
        return Metric._compute_metric_with_qid(individual, data, qid.separation)

    @staticmethod
    def compute_k_anonymity(individual, best_individual, data, prev_value=None):
        return Metric._compute_metric_with_pr(individual, data, pr.k_anonymity)

    @staticmethod
    def compute_l_diverstity(individual, best_individual, data, prev_value=None):
        return Metric._compute_metric_with_pr(individual, data, pr.l_diversity)

    @staticmethod
    def compute_delta_metric(individual, best_individual, data, compute_func):
        """Compute the difference between the current and best individual for a given metric."""
        if best_individual is None:
            return 0.0
        return compute_func(individual, None, data) - compute_func(best_individual, None, data)

    @staticmethod
    def compute_causal_importance(individual, best_individual, data, prev_value=None):
        columns = Metric._get_selected_columns(individual, data)
        return causal_importance(data, columns, "GES", data.causal_graph)

    @staticmethod
    def compute_delta_distinction(individual, best_individual, data, prev_value=None):
        return Metric.compute_delta_metric(individual, best_individual, data, Metric.compute_distinction)

    @staticmethod
    def compute_delta_separation(individual, best_individual, data, prev_value=None):
        return Metric.compute_delta_metric(individual, best_individual, data, Metric.compute_separation)

    @staticmethod
    def compute_attribute_length_penalty(individual, best_individual, data, prev_value=None):
        num_attributes = np.sum(individual)
        proportion = num_attributes / data.shape[1]

        return (1 - proportion) ** 2 + proportion ** 2



    @staticmethod
    def default_aggregator(metric_values, individual, data, alpha, metrics):
        """Aggregate metric values, applying weights and penalties."""
        total = sum(m.coefficient * metric_values[m.name] for m in metrics)
        penalty = metric_values.get("Attribute Length Penalty", 1.0)
        return total / (alpha * penalty)
