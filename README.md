# QIDLearningLib

**QIDLearningLib** is an open-source Python library designed to support the automated detection and evaluation of *quasi-identifiers* (QIDs) within tabular datasets. Quasi-identifiers are attributes that, when combined with external information, can potentially re-identify individuals and pose privacy risks. Identifying these attributes is a critical first step in any privacy-preserving data publishing or anonymization workflow.

---

## Key Features

- **Comprehensive Metric Suite:**  
  Implements a broad set of metrics covering multiple domains including:
  - **Privacy risk** indicators (e.g., uniqueness, k-anonymity approximations)  
  - **Data utility** measures (e.g., attribute relevance, information loss estimates)  
  - **Performance metrics** to assess algorithm efficiency and scalability  
  - **Causality metrics** to understand attribute relationships

- **Metric Redundancy Analysis:**  
  To improve metric selection and reduce computational overhead, QIDLearningLib provides tools for analyzing correlations and redundancies among metrics. This helps users identify which metrics contribute unique information and which may be redundant, enabling a more concise and effective metric set for optimization.

- **Flexible Optimization Framework:**  
  Supports multiple optimization strategies for QID selection, including:  
  - Evolutionary Algorithms (EA)  
  - Simulated Annealing (SA)  
  - Greedy Search (GS)  

- **Extensible and Configurable:**  
  Easily extendable with custom metrics and optimization algorithms.  
  Configurable weighting schemes to tailor the privacy-utility trade-off.

- **Visualization Tools:**  
  Built-in plotting and analysis utilities to visualize metric distributions, optimization progress, and metric correlation matrices for redundancy assessment.

- **Interoperability:**  
  Export identified QID sets and their evaluation metrics in standard CSV format, enabling seamless integration with anonymization tools such as [ARX](https://arx.deidentifier.org/) and [Amnesia](https://amnesia.openaire.eu/).

---

## Why QIDLearningLib?

Manual identification of quasi-identifiers is often subjective, error-prone, and hard to reproduce. QIDLearningLib automates this process using rigorous data-driven methods, enabling:  
- Objective and reproducible QID detection  
- Transparent balancing of privacy and data utility  
- Reduction of metric redundancy for more efficient analysis  
- Scalability to large and complex datasets

---

## Installation

```bash
pip install qidlearninglib

