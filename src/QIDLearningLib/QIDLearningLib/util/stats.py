"""
QIDLearningLib

Library Description (QIDLearningLib):
QIDLearningLib is a Python library designed to provide a comprehensive set of metrics for quasi-identification recognition processes.
The library encompasses metrics for assessing data privacy, data utility, and the performance of quasi-identification recognition algorithms.

Module Description (util.stats):
This module in QIDLearningLib allows provides some auxiliar functions for the statistical tests necessary, specifically in the causality metrics module.

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

from scipy.stats import stats
from typing import List, Tuple


def ks_test(sample1: List[float], sample2: List[float]) -> Tuple[float, float]:
    """
    Perform the Kolmogorov-Smirnov (KS) test for two independent samples.

    Synopse:
    The KS test is a non-parametric test that determines if two samples are drawn from the same distribution.

    Parameters:
    - sample1 (List[float]): The first sample.
    - sample2 (List[float]): The second sample.

    Return:
    Tuple[float, float]: KS statistic and p-value.

    Example:
    >>> ks_stat, p_value = ks_test([1, 2, 3, 4], [1, 2, 3, 5])
    >>> print(ks_stat, p_value)

    """

    # Kolmogorov-Smirnov test
    ks_statistic, p_value = stats.ks_2samp(sample1, sample2)

    return ks_statistic, p_value


def t_test(sample1: List[float], sample2: List[float]) -> Tuple[float, float]:
    """
    Perform the Independent t-test for two independent samples.

    Synopse:
    The Independent t-test assesses whether the means of two independent samples are significantly different.

    Parameters:
    - sample1 (List[float]): The first sample.
    - sample2 (List[float]): The second sample.

    Return:
    Tuple[float, float]: t-statistic and p-value.

    Example:
    >>> t_stat, p_value = t_test([1, 2, 3, 4], [1, 2, 3, 5])
    >>> print(t_stat, p_value)

    """

    # Independent t-test
    t_statistic, p_value = stats.ttest_ind(sample1, sample2)

    return t_statistic, p_value
