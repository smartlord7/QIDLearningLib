import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import skew, kurtosis, mode, norm


class GroupedMetric:
    def __init__(self, values, group_labels, name=None):
        self.values = values
        self.group_labels = group_labels
        self.name = name
        self.mean = np.mean(values)
        self.std = np.std(values)
        self.var = np.var(values)
        self.mn = np.min(values)
        self.mx = np.max(values)
        self.skewness = skew(values)
        self.kurtosis = kurtosis(values)
        self.median = np.median(values)
        self.mode = mode(values).mode
        self.mode_count =  mode(values).count
        self.histogram = None  # Will be populated when plot_histogram is called

    def plot_histogram(self):
        plt.figure(figsize=(10, 6))
        plt.hist(self.values, bins='auto', edgecolor='black', alpha=0.7)
        plt.title(f'Histogram - {self.name}')
        plt.xlabel('Values')
        plt.ylabel('Frequency')
        plt.show()

    def plot_qqplot(self):
        plt.figure(figsize=(10, 6))
        from scipy.stats import probplot
        probplot(self.values, plot=plt)
        plt.title(f'Q-Q Plot - {self.name}')
        plt.show()

    def plot_ecdf(self):
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

    def plot_all(self):
        self.plot_histogram()
        self.plot_qqplot()
        self.plot_ecdf()

    def __repr__(self):
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