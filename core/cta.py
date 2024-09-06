import sys
from pathlib import Path

root_path = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root_path))

from .graph import PCGraph

import numpy as np
from itertools import combinations
from sklearn.decomposition import TruncatedSVD


class CTA:
    """
    CTA class handles the process for Clustering Tendency Assessment
    """

    def __init__(self, data, n_total_samples=None, normalize=True):
        n_components = 2
        self.n_components = n_components

        self.transform = TruncatedSVD(n_components=n_components, algorithm='randomized', random_state=0)

        if isinstance(data, tuple):
            data = data[0]

        self.data = data
        self.selected_data = data
        self.normalize = normalize
        self.selection_count = 0

        self.n_samples, self.n_features = len(data), n_components
        self.n_total_samples = n_total_samples if n_total_samples is not None else self.n_samples
        self.n_partitions = int(2 * np.ceil(np.log(1 + self.n_total_samples)) - 1)

        self.selection_matrix = np.zeros((self.n_partitions, self.n_partitions))
        self.frequency_matrix = np.zeros((self.n_partitions, self.n_partitions))
        self.positions = np.zeros((self.n_partitions, self.n_partitions, 2))

        self.pc_graph = None

        self.preprocessing()

    def preprocessing(self):
        # Reduce dimensionality and normalize data
        data = np.asarray(self.data)

        if data.ndim == 1:
            data = np.stack((data, data), axis=1)

        data = data.reshape(data.shape[0], -1)
        n_samples, n_features = data.shape

        if n_features > self.n_components:
            # todo: Try different dimensionality reduction methods for complex high-dimensional datasets?
            data = self.transform.fit_transform(data)

        if self.normalize:
            data_x, data_y = data[:, 0], data[:, 1]
            if data_x.max() - data_x.min() > 0:
                data_x = (data_x - data_x.min()) / (data_x.max() - data_x.min())
            if data_y.max() - data_y.min() > 0:
                data_y = (data_y - data_y.min()) / (data_y.max() - data_y.min())
            data = np.stack((data_x, data_y), axis=1)

        self.data = data
        return data

    def partition_embedding_space(self):
        # Partitioning the embedding space
        data = self.data
        selection_count = 0
        for dim_i, dim_j in combinations(range(self.n_features), 2):
            data = data[:, [dim_i, dim_j]]

            # todo: Add batch processing
            # todo: Add parallel processing of grid cells

            iter_elements = np.arange(self.n_partitions)
            for i in iter_elements:
                start_idx = i / self.n_partitions if i > 0 else -1
                end_idx = (i + 1) / self.n_partitions
                rows = data[(data[:, 0] > start_idx) & (data[:, 0] <= end_idx)]
                for j in iter_elements:
                    col_start_idx = j / self.n_partitions if j > 0 else -1
                    col_end_idx = (j + 1) / self.n_partitions
                    selected = rows[(rows[:, 1] > col_start_idx) & (rows[:, 1] <= col_end_idx)]
                    self.frequency_matrix[i, j] = len(selected)
                    if len(selected) > 0:
                        selection_count += 1
                        self.selection_matrix[i, j] = 1
                        self.positions[i, j] = np.mean(selected, axis=0)

        self.selection_count = selection_count

        assert np.sum(self.frequency_matrix) == self.n_samples, "Not all samples have been selected."

        return self.positions, self.selection_matrix, self.frequency_matrix

    def create_graph(self):
        self.pc_graph = PCGraph(self.selection_matrix, self.frequency_matrix, self.positions, self.n_partitions)
        self.pc_graph.build_adjacency_matrix()
        return self.pc_graph
