import logging
import numpy as np
from QIDLearningLib.metrics import qid_specific as qid
from QIDLearningLib.metrics import data_privacy as pr
from QIDLearningLib.structure.grouped_metric import GroupedMetric

# =============================================================================
# Metric Classes and Functions
# =============================================================================

class Metric:
    """
    A simple container for a metric. The user supplies:
      - name: a string name,
      - coefficient: a numerical weight,
      - compute_func: a function with signature
            compute_func(individual, best_individual, data, prev_value=None)
        that returns a (float) metric value,
      - maximize: a Boolean indicating if a higher value is better.
    """

    def __init__(self, name, coefficient, compute_func, maximize=True):
        self.name = name
        self.coefficient = coefficient
        self.compute_func = compute_func
        self.maximize = maximize
        logging.info(f"Metric initialized: {self.name}, coefficient: {self.coefficient}, maximize: {self.maximize}")


    def compute(self, individual, best_individual, data, prev_value=None):
        value = self.compute_func(individual, best_individual, data, prev_value)
        logging.info(f"Metric [{self.name}] computed value: {value}")
        return value

    @staticmethod
    def compute_distinction(individual, best_individual, data, prev_value=None):
        selected_indices = np.where(individual == 1)[0]
        selected_columns = data.columns[selected_indices].tolist()
        if len(selected_columns) == 0:
            logging.info("compute_distinction: No columns selected, returning 0.0")
            return 0.0
        result = qid.distinction(data, selected_columns) / 100.0
        logging.info(f"compute_distinction: Selected columns: {selected_columns}, Result: {result}")
        return result

    @staticmethod
    def compute_separation(individual, best_individual, data, prev_value=None):
        selected_indices = np.where(individual == 1)[0]
        selected_columns = data.columns[selected_indices].tolist()
        if len(selected_columns) == 0:
            logging.info("compute_separation: No columns selected, returning 0.0")
            return 0.0
        result = qid.separation(data, selected_columns) / 100.0
        logging.info(f"compute_separation: Selected columns: {selected_columns}, Result: {result}")
        return result

    @staticmethod
    def compute_k_anonymity(individual, best_individual, data, prev_value=None):
        selected_indices = np.where(individual == 1)[0]
        selected_columns = data.columns[selected_indices].tolist()
        if len(selected_columns) == 0:
            logging.info("compute_k_anonymity: No columns selected, returning 0.0")
            return 0.0
        result = pr.k_anonymity(data, selected_columns)
        if isinstance(result, GroupedMetric):
            final_value = result.mean
        else:
            final_value = result
        logging.info(f"compute_k_anonymity: Selected columns: {selected_columns}, Result: {final_value}")
        return final_value

    @staticmethod
    def compute_delta_distinction(individual, best_individual, data, prev_value=None):
        if best_individual is None:
            logging.info("compute_delta_distinction: No best individual available, returning 0.0")
            return 0.0
        current = Metric.compute_distinction(individual, None, data)
        best_val = Metric.compute_distinction(best_individual, None, data)
        delta = current - best_val
        logging.info(f"compute_delta_distinction: Current: {current}, Best: {best_val}, Delta: {delta}")
        return delta

    @staticmethod
    def compute_delta_separation(individual, best_individual, data, prev_value=None):
        if best_individual is None:
            logging.info("compute_delta_separation: No best individual available, returning 0.0")
            return 0.0
        current = Metric.compute_separation(individual, None, data)
        best_val = Metric.compute_separation(best_individual, None, data)
        delta = current - best_val
        logging.info(f"compute_delta_separation: Current: {current}, Best: {best_val}, Delta: {delta}")
        return delta

    @staticmethod
    def compute_attribute_length_penalty(individual, best_individual, data, prev_value=None):
        num_attributes = np.sum(individual)
        proportion = num_attributes / data.shape[1]
        penalty = (1 - proportion) ** 2 + proportion ** 2
        logging.info(
            f"compute_attribute_length_penalty: Num attributes: {num_attributes}, Proportion: {proportion}, Penalty: {penalty}")
        return penalty