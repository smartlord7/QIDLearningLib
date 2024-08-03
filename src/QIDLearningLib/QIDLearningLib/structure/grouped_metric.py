"""
QIDLearningLib

Library Description (QIDLearningLib):
QIDLearningLib is a Python library designed to provide a comprehensive set of metrics for quasi-identifiers recognition processes.
The library encompasses metrics for assessing data privacy, data utility, and the performance of quasi-identifiers recognition algorithms.

Module Description:
This module contains the `GroupedMetric` class, which is designed to encapsulate and analyze grouped metric data. This class allows users to compute various descriptive statistics and visualize the distribution of values within different groups. The module provides methods for plotting histograms, Q-Q plots, empirical cumulative distribution functions (ECDFs), and a summary of descriptive statistics.

Year: 2023/2024
Institution: University of Coimbra
Department: Department of Informatics Engineering
Program: Master's in Informatics Engineering - Intelligent Systems
Author: Sancho Amaral Simões
Student No: 2019217590
Emails: sanchoamaralsimoes@gmail.com (Personal)| uc2019217590@student.uc.pt | sanchosimoes@student.dei.uc.pt
Version: v0.01

License:
This open-source software is released under the terms of the GNU General Public License, version 3 (GPL-3.0).
For more details, see https://www.gnu.org/licenses/gpl-3.0.html

"""

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import skew, kurtosis, mode, norm
from typing import Union


class GroupedMetric:
    def __init__(self, values: Union[list, np.ndarray], group_labels: Union[list, np.ndarray], name: str = None):
        """
        Create an instance of the GroupedMetric class.

        Synopse:
        This class represents a grouped metric with various descriptive statistics.

        Parameters:
        - values (Union[list, np.ndarray]): The values of the metric.
        - group_labels (Union[list, np.ndarray]): Labels for different groups.
        - name (str): The name of the metric.

        Return:
        None

        Example:
        >>> values = [1, 2, 3, 4, 5]
        >>> group_labels = ['A', 'B', 'A', 'B', 'A']
        >>> metric = GroupedMetric(values, group_labels, name='Example Metric')

        """
        self.values = np.array(values)
        self.group_labels = np.array(group_labels)
        self.name = name
        self.mean = np.mean(self.values)
        self.std = np.std(self.values)
        self.var = np.var(self.values)
        self.mn = np.min(self.values)
        self.mx = np.max(self.values)
        self.skewness = skew(self.values)
        self.kurtosis = kurtosis(self.values)
        self.median = np.median(self.values)
        self.mode = mode(self.values).mode
        self.mode_count =  mode(self.values).count
        self.histogram = None  # Will be populated when plot_histogram is called

    def plot_histogram(self) -> None:
        """
        Plot the histogram of the metric values.

        Synopse:
        This method plots the histogram of the metric values.

        Parameters:
        None

        Return:
        None

        Example:
        >>> metric.plot_histogram()

        """
        plt.figure(figsize=(10, 6))
        plt.hist(self.values, bins='auto', edgecolor='black', alpha=0.7)
        plt.title(f'Histogram - {self.name}')
        plt.xlabel('Values')
        plt.ylabel('Frequency')
        plt.show()

    def plot_qqplot(self) -> None:
        """
        Plot the Q-Q plot of the metric values.

        Synopse:
        This method plots the Q-Q plot of the metric values.

        Parameters:
        None

        Return:
        None

        Example:
        >>> metric.plot_qqplot()

        """
        plt.figure(figsize=(10, 6))
        from scipy.stats import probplot
        probplot(self.values, plot=plt)
        plt.title(f'Q-Q Plot - {self.name}')
        plt.show()

    def plot_ecdf(self) -> None:
        """
        Plot the empirical CDF and normal distribution's CDF.

        Synopse:
        This method plots the empirical CDF and normal distribution's CDF.

        Parameters:
        None

        Return:
        None

        Example:
        >>> metric.plot_ecdf()

        """
        plt.figure(figsize=(10, 6))

        # Plot empirical ECDF
        values_sorted = np.sort(self.values)
        ecdf = np.arange(1, len(values_sorted) + 1) / len(values_sorted)
        plt.step(values_sorted, ecdf, label='Empirical ECDF')

        # Plot normal distribution's CDF
        normal_cdf = norm.cdf(values_sorted, loc=self.mean, scale=self.std)
        plt.plot(values_sorted, normal_cdf, label='Normal CDF', linestyle='--')

        plt.title(f'Empirical CDF vs Normal CDF - {self.name}')
        plt.xlabel('Values')
        plt.ylabel('Cumulative Probability')
        plt.legend()
        plt.show()

    def plot_all(self) -> None:
        """
        Plot histogram, Q-Q plot, and empirical CDF.

        Synopse:
        This method plots the histogram, Q-Q plot, and empirical CDF of the metric values.

        Parameters:
        None

        Return:
        None

        Example:
        >>> metric.plot_all()

        """
        self.plot_histogram()
        self.plot_qqplot()
        self.plot_ecdf()

    def __repr__(self) -> str:
        """
        Provide a string representation of the GroupedMetric.

        Synopse:
        This method provides a string representation of the GroupedMetric.

        Parameters:
        None

        Return:
        str: String representation.

        Example:
        >>> repr(metric)

        """
        return (
            f"GroupedMetric(\n"
            f"  name={self.name},\n"
            f"  group_labels={self.group_labels},\n"
            f"  centrality={{\n"
            f"    mean={self.mean},\n"
            f"    median={self.median},\n"
            f"    mode={self.mode}\n"
            f"    mode_count={self.mode_count}\n"
            f"  }},\n"
            f"  dispersion={{\n"
            f"    std={self.std},\n"
            f"    var={self.var},\n"
            f"    min={self.mn},\n"
            f"    max={self.mx},\n"
            f"    skewness={self.skewness},\n"
            f"    kurtosis={self.kurtosis}\n"
            f"  }}\n"
            f")"
        )
