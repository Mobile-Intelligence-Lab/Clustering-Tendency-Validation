import sys
from pathlib import Path

root_path = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root_path))

import numpy as np
from sklearn.cluster import KMeans, DBSCAN, SpectralClustering
from sklearn.mixture import GaussianMixture

from core.stats import CtaStats
from helpers.data import get_dataset
from helpers.params import EPS_DBSCAN as eps_dbscan
from helpers.stats import hopkins, purity_score, rand_index, adjusted_rand_index, nmi,\
    silhouette_coefficient, dunn_index, db_index, ch_index, accuracy

import warnings
warnings.filterwarnings('ignore')


if __name__ == "__main__":
    scores = {
        "purity": [], "rand": [], "adjusted_rand": [], "nmi": [],
        "sil": [], "dunn": [], "db": [], "ch": [],
        "psi": [],
        "acc": []
    }

    print(f"Validating clustering results...")

    for dataset_name in eps_dbscan.keys():
        dataset = get_dataset(dataset_name)
        samples, labels = dataset

        print(f"\n{'-' * 100}")
        print(f"Evaluating clustering qualities on dataset: {dataset_name} [{len(samples)} samples].")
        print('-' * 100)

        unique_labels = list(set(labels))
        n_classes = len(unique_labels)

        # Performing clustering
        print("\nPerforming clustering...")
        kmeans_labels = KMeans(n_clusters=n_classes).fit(samples).labels_

        dbscan_labels = DBSCAN(eps=eps_dbscan[dataset_name.lower()], min_samples=n_classes).fit(samples).labels_

        spectral_labels = SpectralClustering(n_clusters=n_classes, eigen_solver="arpack",
                                             random_state=0, n_neighbors=9,
                                             affinity="nearest_neighbors").fit(samples).labels_

        gm_labels = GaussianMixture(n_components=n_classes, random_state=0).fit_predict(samples)

        print(f"Dataset: {dataset_name} | Number of samples: {len(samples)} | Number of classes: {n_classes}")
        print(f"  Number of clusters: k-Means ({len(list(set(kmeans_labels)))}) "
              f"| DBSCAN ({len(list(set(dbscan_labels)))})"
              f"| Spectral Clustering ({len(list(set(spectral_labels)))})"
              f"| Gaussian Mixture ({len(list(set(gm_labels)))})")

        # external validation metrics
        print(f"\nComputing external cluster validation metrics...")
        purity_kmeans = purity_score(labels, kmeans_labels)
        purity_dbscan = purity_score(labels, dbscan_labels)
        purity_spectral = purity_score(labels, spectral_labels)
        purity_gm = purity_score(labels, gm_labels)
        print(f"  Purity Score | k-Means: {purity_kmeans:.2f}| DBSCAN: {purity_dbscan:.2f} "
              f"| Spectral: {purity_spectral:.2f}| Gaussian Mixture: {purity_gm:.2f}")

        ri_kmeans = rand_index(labels, kmeans_labels)
        ri_dbscan = rand_index(labels, dbscan_labels)
        ri_spectral = rand_index(labels, spectral_labels)
        ri_gm = rand_index(labels, gm_labels)
        print(f"  Rand Index | k-Means: {ri_kmeans:.2f}| DBSCAN: {ri_dbscan:.2f} "
              f"| Spectral: {ri_spectral:.2f}| Gaussian Mixture: {ri_gm:.2f}")

        ari_kmeans = adjusted_rand_index(labels, kmeans_labels)
        ari_dbscan = adjusted_rand_index(labels, dbscan_labels)
        ari_spectral = adjusted_rand_index(labels, spectral_labels)
        ari_gm = adjusted_rand_index(labels, gm_labels)
        print(f"  Adjusted RI | k-Means: {ari_kmeans:.2f}| DBSCAN: {ari_dbscan:.2f} "
              f"| Spectral: {ari_spectral:.2f}| Gaussian Mixture: {ari_gm:.2f}")

        nmi_kmeans = nmi(labels, kmeans_labels)
        nmi_dbscan = nmi(labels, dbscan_labels)
        nmi_spectral = nmi(labels, spectral_labels)
        nmi_gm = nmi(labels, gm_labels)
        print(f"  NMI | k-Means: {nmi_kmeans:.2f}| DBSCAN: {nmi_dbscan:.2f} "
              f"| Spectral: {nmi_spectral:.2f}| Gaussian Mixture: {nmi_gm:.2f}")

        # internal validation metrics
        print(f"\nEvaluating internal cluster validation metrics...")

        if n_classes > 1:
            sc_kmeans = silhouette_coefficient(samples, kmeans_labels)
            sc_dbscan = silhouette_coefficient(samples, dbscan_labels)
            sc_spectral = silhouette_coefficient(samples, spectral_labels)
            sc_gm = silhouette_coefficient(samples, gm_labels)
            print(f"  Silhouette Coefficient | k-Means: {sc_kmeans:.2f}| DBSCAN: {sc_dbscan:.2f} "
                  f"| Spectral: {sc_spectral:.2f}| Gaussian Mixture: {sc_gm:.2f}")
        else:
            print(f"  Silhouette Coefficient | k-Means: N/A | DBSCAN: N/A "
                  f"| Spectral: N/A | Gaussian Mixture: N/A")

        if n_classes > 1:
            di_kmeans = dunn_index(samples, kmeans_labels)
            di_dbscan = dunn_index(samples, dbscan_labels)
            di_spectral = dunn_index(samples, spectral_labels)
            di_gm = dunn_index(samples, gm_labels)
            print(f"  Dunn Index | k-Means: {di_kmeans:.2f}| DBSCAN: {di_dbscan:.2f} "
                  f"| Spectral: {di_spectral:.2f}| Gaussian Mixture: {di_gm:.2f}")
        else:
            print(f"  Dunn Index | k-Means: N/A | DBSCAN: N/A "
                  f"| Spectral: N/A | Gaussian Mixture: N/A")

        if n_classes > 1:
            db_kmeans = db_index(samples, kmeans_labels)
            db_dbscan = db_index(samples, dbscan_labels)
            db_spectral = db_index(samples, spectral_labels)
            db_gm = db_index(samples, gm_labels)
            print(f"  DB index | k-Means: {db_kmeans:.2f}| DBSCAN: {db_dbscan:.2f} "
                  f"| Spectral: {db_spectral:.2f}| Gaussian Mixture: {db_gm:.2f}")
        else:
            print(f"  DB index | k-Means: N/A | DBSCAN: N/A "
                  f"| Spectral: N/A | Gaussian Mixture: N/A")

        if n_classes > 1:
            ch_kmeans = ch_index(samples, kmeans_labels)
            ch_dbscan = ch_index(samples, dbscan_labels)
            ch_spectral = ch_index(samples, spectral_labels)
            ch_gm = ch_index(samples, gm_labels)
            print(f"  CH Index | k-Means: {ch_kmeans:.2f}| DBSCAN: {ch_dbscan:.2f} "
                  f"| Spectral: {ch_spectral:.2f}| Gaussian Mixture: {ch_gm:.2f}")
        else:
            print(f"  CH Index | k-Means: N/A | DBSCAN: N/A "
                  f"| Spectral: N/A | Gaussian Mixture: N/A")

        # Partitioning Sensitivity Indices
        cta_stats_kmeans = CtaStats(data=(samples, kmeans_labels))
        psi_score_kmeans = cta_stats_kmeans.compute_psi()

        cta_stats_dbscan = CtaStats(data=(samples, dbscan_labels))
        psi_score_dbscan = cta_stats_dbscan.compute_psi()

        cta_stats_spectral = CtaStats(data=(samples, spectral_labels))
        psi_score_spectral = cta_stats_spectral.compute_psi()

        cta_stats_gm = CtaStats(data=(samples, gm_labels))
        psi_score_gm = cta_stats_gm.compute_psi()

        print(f"  PSI | k-Means: {psi_score_kmeans:.2f} | DBSCAN: {psi_score_dbscan:.2f} "
              f"| Spectral: {psi_score_spectral:.2f} | Gaussian Mixture: {psi_score_gm:.2f}")

        # Collecting results
        accuracies = np.round([
            accuracy(labels, kmeans_labels),
            accuracy(labels, dbscan_labels),
            accuracy(labels, spectral_labels),
            accuracy(labels, gm_labels)
        ], 2)

        all_max_acc_indices = np.where(accuracies == accuracies.max())[0]

        values = np.round([purity_kmeans, purity_dbscan, purity_spectral, purity_gm], 2)
        scores["purity"].append(all_max_acc_indices[0] if values.max() in values[all_max_acc_indices]
                                else np.argmax(values))

        values = np.round([ri_kmeans, ri_dbscan, ri_spectral, ri_gm], 2)
        scores["rand"].append(all_max_acc_indices[0] if values.max() in values[all_max_acc_indices]
                              else np.argmax(values))

        values = np.round([ari_kmeans, ari_dbscan, ari_spectral, ari_gm], 2)
        scores["adjusted_rand"].append(all_max_acc_indices[0] if values.max() in values[all_max_acc_indices]
                                       else np.argmax(values))

        values = np.round([nmi_kmeans, nmi_dbscan, nmi_spectral, nmi_gm], 2)
        scores["nmi"].append(all_max_acc_indices[0] if values.max() in values[all_max_acc_indices]
                             else np.argmax(values))

        values = np.round([sc_kmeans, sc_dbscan, sc_spectral, sc_gm] if n_classes > 1 else [1, 1, 1, 1], 2)
        scores["sil"].append(all_max_acc_indices[0] if values.max() in values[all_max_acc_indices]
                             else np.argmax(values))

        values = np.round([di_kmeans, di_dbscan, di_spectral, di_gm] if n_classes > 1 else [1, 1, 1, 1], 2)
        scores["dunn"].append(all_max_acc_indices[0] if values.max() in values[all_max_acc_indices]
                              else np.argmax(values))

        values = np.round([db_kmeans, db_dbscan, db_spectral, db_gm] if n_classes > 1 else [1, 1, 1, 1], 2)
        scores["db"].append(all_max_acc_indices[0] if values.min() in values[all_max_acc_indices]
                            else np.argmin(values))

        values = np.round([ch_kmeans, ch_dbscan, ch_spectral, ch_gm] if n_classes > 1 else [1, 1, 1, 1], 2)
        scores["ch"].append(all_max_acc_indices[0] if values.max() in values[all_max_acc_indices]
                            else np.argmax(values))

        values = np.round([psi_score_kmeans, psi_score_dbscan, psi_score_spectral, psi_score_gm], 2)
        scores["psi"].append(all_max_acc_indices[0] if values.max() in values[all_max_acc_indices]
                             else np.argmax(values))

        scores["acc"].append(all_max_acc_indices[0])

        print(f"\nTrue Accuracies | k-Means: {accuracies[0]:.2f} | DBSCAN: {accuracies[1]:.2f} "
              f"| Spectral: Acc: {accuracies[2]:.2f} | Gaussian Mixture: Acc: {accuracies[3]:.2f} "
              f"\n   Best Accuracy: {np.max(accuracies):.2f}\n")

    print(f"\n{'-' * 50}")
    print(f"Computing correct selection ratios...")
    print('-' * 50)

    s = "\n  "
    for m in ["purity", "rand", "adjusted_rand", "nmi", "sil", "dunn", "db", "ch", "psi"]:
        acc = np.array(scores["acc"])
        vals = np.array(scores[m])
        s += f"{m} {np.sum(np.where(vals == acc, 1, 0)) / len(vals):.2f} | "
    print(s)
