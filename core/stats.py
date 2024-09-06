import numpy as np
from .cta import CTA


class CtaStats:
    """
    Implements statistical methods for analyzing Proximally-Connected graphs.
    """

    def __init__(self, data, n_total_samples=None, normalize=True):
        self.data = data

        self.cta = CTA(data, n_total_samples, normalize)
        self.cta.partition_embedding_space()

        self.edge_weights = []
        self.graph = None

    def compute_phi(self):
        """
        Computes the Proximal Homogeneity Index
        """

        # Ensuring the proximally-connected graph has been created
        nx_pcg_graph = self.cta.create_graph().graph
        self.graph = nx_pcg_graph
        self.edge_weights = [0]

        # Handling edge cases
        if self.cta.data[:, 0].min() == self.cta.data[:, 0].max() and \
                self.cta.data[:, 1].min() == self.cta.data[:, 1].max():
            return 1

        count = np.sum(self.cta.frequency_matrix)
        if count <= 1:
            return 1 / count if count > 0 else 0
        elif self.cta.selection_count <= 2:
            self.edge_weights = [np.sqrt(2) / self.cta.n_partitions]
            return 0.5

        # Collecting edge weights from the graph, ignoring any NaN values
        edge_weights = np.array(
            [data['weight'] for _, _, data in nx_pcg_graph.edges(data=True) if not np.isnan(data['weight'])])

        # Returning a default score if the graph has no edges
        if edge_weights.size == 0:
            return 0

        self.edge_weights = edge_weights

        # Calculating the normalized mean edge weight
        mean_edge_weight = np.mean(edge_weights)
        max_edge_weight = np.max(edge_weights)
        phi_score = mean_edge_weight / max_edge_weight if max_edge_weight > 0 else np.ones_like(edge_weights)

        return phi_score

    def compute_psi(self, adjust_for_misplaced=True):
        """
        Computes the Partitioning Sensitivity Index
        """
        xs, ys = self.data
        phi_data = self.cta.data
        ds_size = len(phi_data)

        unique_labels = list(set(ys))

        cluster_scores = []
        correct_count, total_misplaced_nodes = 0, 0

        # Computing global stats
        global_homogeneity = self.compute_phi()
        global_graph = self.graph

        if len(unique_labels) == 1:
            cluster_scores = [global_homogeneity]
            correct_count = 1
        else:
            # Processing each label as a separate cluster
            for i, label in enumerate(unique_labels):
                n_misplaced_nodes = 0
                x = phi_data[np.where(ys == label)[0]]

                x_cta_stats = CtaStats(x, n_total_samples=ds_size, normalize=False)
                x_phi = x_cta_stats.compute_phi()

                dists = []
                if adjust_for_misplaced:
                    x_bar = phi_data[np.where(ys != label)[0]]
                    x_bar_cta_stats = CtaStats(x_bar, n_total_samples=ds_size, normalize=False)
                    x_bar_cta_stats.compute_phi()

                    x_graph, x_bar_graph = x_cta_stats.graph, x_bar_cta_stats.graph
                    x_distances = x_cta_stats.edge_weights

                    if None in (x_graph, x_bar_graph, global_graph):
                        separation_distance = np.sqrt(2) / len(unique_labels)
                        n_misplaced_nodes += len(x_bar)
                    else:
                        # Analyzing distances between nodes of the current cluster and other clusters
                        counted_nodes = []
                        for ni, (_, node_s) in enumerate(x_graph.nodes(data=True)):
                            for nj, (_, node_d) in enumerate(x_bar_graph.nodes(data=True)):
                                if f"{ni}-{nj}" in counted_nodes or f"{nj}-{ni}" in counted_nodes:
                                    continue
                                dist = np.linalg.norm(node_s["position"] - node_d["position"], ord=2)
                                dists.append(dist)
                                counted_nodes.append(f"{ni}-{nj}")
                                if dist < np.max(x_distances):
                                    n_misplaced_nodes += 1
                        separation_distance = np.min(dists)

                    total_misplaced_nodes += n_misplaced_nodes

                    # Incrementing the correct count if the cluster is well-separated
                    correct_count += 1 if len(x) > 1 and np.max(x_distances) <= separation_distance else 0
                else:
                    correct_count += 1
                cluster_scores.append((1 - x_phi) * (1 if len(x) > 1 else 0))

        # Calculating the global sensitivity index
        cluster_scores = np.array(cluster_scores)
        if np.max(cluster_scores) > 0:
            cluster_scores /= np.max(cluster_scores)

        if adjust_for_misplaced:
            ratio = (correct_count + 1) / (len(unique_labels) + 1)
            ratio *= np.exp(-total_misplaced_nodes * (correct_count + 1) / (len(phi_data) ** 2))
            ratio = np.sqrt(np.log(1 + ratio) / np.log(2))
        else:
            ratio = 1

        psi_score = np.mean(cluster_scores) * ratio

        return psi_score
