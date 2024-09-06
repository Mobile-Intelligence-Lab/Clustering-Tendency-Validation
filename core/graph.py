import os
import sys

ups = '/..' * 1
root_path = os.path.dirname(os.path.realpath(__file__)) + ups
sys.path.append(root_path)

import numpy as np
import networkx as nx
from sklearn.neighbors import KDTree


class PCGraph:
    """
    Implements Proximally-Connected graphs.
    """

    def __init__(self, selection_matrix, frequency_matrix, positions, n_partitions):
        self.selection_matrix = selection_matrix
        self.frequency_matrix = frequency_matrix
        self.positions = positions
        self.n_partitions = n_partitions
        self.graph = None

        self.adjacency_matrix = None

    def build_adjacency_matrix(self):
        """
        Creation of the adjacency matrix based constraints defined by PC graphs.
        """
        # Filtering positions based on the selection matrix
        positions = self.positions.reshape(-1, 2)

        xs, ys = np.where(self.selection_matrix > 0)
        adjacency_matrix = np.zeros((xs.shape[0], xs.shape[0]))

        if np.sum(self.selection_matrix) > 2:
            selected_positions = positions[self.selection_matrix.flatten() > 0]

            # Finding nearest neighbors
            tree = KDTree(selected_positions)
            distances, indices = tree.query(selected_positions, k=3, sort_results=True)

            # Populating the adjacency matrix based on distances
            threshold = np.sqrt(2) / self.n_partitions  # Threshold defined for PC graphs
            for i, (dist_row, idx_row) in enumerate(zip(distances, indices)):
                for dist, j in zip(dist_row, idx_row):
                    if 0 < dist < threshold:
                        adjacency_matrix[i, j] = dist
                        adjacency_matrix[j, i] = dist  # Ensuring symmetry

        adjacency_matrix = self.add_proximal_connectivity_constraint(adjacency_matrix)
        self.adjacency_matrix = adjacency_matrix

        # Re-creating graph with 'position' and 'weight' as attributes
        graph_with_attributes = nx.Graph()
        idx = 0
        for i in range(self.n_partitions):
            for j in range(self.n_partitions):
                if self.selection_matrix[i, j] == 1:
                    graph_with_attributes.add_node(idx, pos=self.positions[i, j])
                    graph_with_attributes.nodes[idx]['position'] = self.positions[i, j]
                    idx += 1

        for i in range(len(adjacency_matrix)):
            for j in range(i, len(adjacency_matrix)):
                if adjacency_matrix[i, j] > 0:
                    graph_with_attributes.add_edge(i, j, weight=adjacency_matrix[i, j])
                    graph_with_attributes.add_edge(j, i, weight=adjacency_matrix[j, i])

        self.graph = graph_with_attributes

    def add_proximal_connectivity_constraint(self, adjacency_matrix):
        graph = nx.from_numpy_array(adjacency_matrix)

        idx = 0
        mapping = {}
        for i in range(self.n_partitions):
            for j in range(self.n_partitions):
                if self.selection_matrix[i, j] == 1:
                    mapping[idx] = self.positions[i, j]
                    idx += 1

        for i, s1 in enumerate(nx.connected_components(graph)):
            for j, s2 in enumerate(nx.connected_components(graph)):
                if i == j:
                    continue
                subgraph1 = graph.subgraph(s1)
                subgraph2 = graph.subgraph(s2)

                edge_to_add = self.find_closest_edge_between_components(subgraph1, subgraph2, mapping)
                u, v, dist = edge_to_add
                adjacency_matrix[u, v] = adjacency_matrix[v, u] = dist
                del graph
                graph = nx.from_numpy_array(adjacency_matrix)
                break

        return adjacency_matrix

    def find_closest_edge_between_components(self, component_i, component_j, mapping):
        min_dist = np.inf
        edge_to_add = None

        for node_i in component_i:
            pos_i = mapping[node_i]
            for node_j in component_j:
                pos_j = mapping[node_j]
                dist = np.linalg.norm(pos_i - pos_j, ord=2)
                # Updating the closest edge if it is the shortest distance so far
                if dist < min_dist:
                    min_dist = dist
                    edge_to_add = (node_i, node_j, dist)

        return edge_to_add

    def count_nodes_and_edges(self):
        num_nodes = self.graph.number_of_nodes()
        num_edges = self.graph.number_of_edges()
        return num_nodes, num_edges
