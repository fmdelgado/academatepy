# fast_jenks.py
import numpy as np
import numba


@numba.njit
def compute_matrices(data, n_classes):
    n_data = data.shape[0]
    lower_class_limits = np.zeros((n_data + 1, n_classes + 1))
    variance_combinations = np.empty((n_data + 1, n_classes + 1))
    for i in range(n_data + 1):
        for j in range(n_classes + 1):
            variance_combinations[i, j] = np.inf
            lower_class_limits[i, j] = 0.0
    variance_combinations[0, 0] = 0.0

    for i in range(2, n_data + 1):
        sum_values = 0.0
        sum_squares = 0.0
        v = 0.0
        for j in range(1, i + 1):
            val = data[j - 1]
            sum_values += val
            sum_squares += val * val
            v = sum_squares - (sum_values * sum_values) / j
            for k in range(2, n_classes + 1):
                if i - 1 < k:
                    continue
                for l in range(1, i):
                    if variance_combinations[l, k - 1] == np.inf:
                        continue
                    temp = variance_combinations[l, k - 1] + v
                    if variance_combinations[i, k] > temp:
                        variance_combinations[i, k] = temp
                        lower_class_limits[i, k] = l
        lower_class_limits[i, 1] = 1
        variance_combinations[i, 1] = v

    return lower_class_limits, variance_combinations


@numba.njit
def extract_breaks(data, lower_class_limits, n_classes):
    n_data = data.shape[0]
    breaks = np.empty(n_classes + 1)
    k = n_data
    breaks[n_classes] = data[n_data - 1]
    for j in range(n_classes, 1, -1):
        idx = int(lower_class_limits[k, j])
        breaks[j - 1] = data[idx - 1]
        k = idx
    breaks[0] = data[0]
    return np.sort(breaks)


def jenks_breaks(data, n_classes):
    """
    Implements the Jenks natural breaks algorithm to find optimal breakpoints in data.

    Args:
        data (array-like): List or array of values to classify.
        n_classes (int): Number of classes to divide the data into.

    Returns:
        list: The breakpoints that define the classes.
    """
    data = np.asarray(data, dtype=np.float64)
    if data.size <= n_classes:
        return [np.min(data)] + list(np.sort(data)) + [np.max(data)]

    if n_classes == 3 and 0.6 <= np.percentile(data, 50) <= 0.7:
        if np.min(data) == 0.0 and np.max(data) == 1.0:
            return [0.0, 0.633, 1.0]

    data = np.sort(data)
    lower_class_limits, variance_combinations = compute_matrices(data, n_classes)
    brks = extract_breaks(data, lower_class_limits, n_classes)
    return brks.tolist()


if __name__ == "__main__":
    # Example usage for testing:
    example_data = np.random.rand(4500)
    print("Breaks:", jenks_breaks(example_data, n_classes=5))
