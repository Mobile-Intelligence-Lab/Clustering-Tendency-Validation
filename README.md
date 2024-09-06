# Proximally-Connected Graphs for Cluster Analysis


A python implementation of two novel statistics introduced in the paper:
**"[Deciphering Clusters with a Deterministic Measure of Clustering Tendency](https://ieeexplore.ieee.org/abstract/document/10227568)."**


The **Proximal Homogeneity Index (PHI)** and the **Partitioning Sensitivity Index (PSI)**, derived from **Proximally-Connected (PC) Graphs**, are designed to assess the clustering tendency of datasets and validate the quality of clustering results, respectively.

### Key Concepts:
- **Proximally-Connected Graphs (PC Graphs)**: These graphs are constructed from the relationships between dataset samples, preserving their local and global structures. They are central to both PHI and PSI, allowing efficient and deterministic computation of clustering tendencies and validation metrics.
  
  - PC Graphs ensure that all components of the dataset are connected in a way that reflects their true structural properties, without introducing biases towards dense regions.
  - They are highly scalable, making the approach suitable for very large datasets.

### Statistics:

- **PHI (Proximal Homogeneity Index)**:
  - Provides a deterministic measure of the clustering tendency of a dataset based on the connected structure of PC Graphs.
  - Computes the homogeneity of samples, indicating whether a dataset contains distinct groups (clusters) or is more homogeneous.
  - A PHI score closer to 0 indicates a dataset that is highly clusterable, while a score closer to 1 reflects high homogeneity (less likely to contain clusters).

- **PSI (Partitioning Sensitivity Index)**:
  - Evaluates the quality of clusters by using PHI to assess the homogeneity of each individual cluster.
  - Helps to identify the most suitable clustering algorithm, the optimal number of clusters, and the best dimensionality reduction method to use.
  - A higher PSI score indicates better separation between clusters and a more accurate clustering.

## Getting Started
Setting up a local environment to run and use the statistics.

### Cloning the repository
Clone the repository to your local machine by running:

```shell
git clone https://github.com/Mobile-Intelligence-Lab/Clustering-Tendency-Validation.git [LOCAL_DIRECTORY_PATH]
```
Replace `[LOCAL_DIRECTORY_PATH]` with your desired directory path or leave it blank to clone in the current directory.

### Requirements

- Python 3.x
- Required Libraries:
  - `numpy`
  - `scikit-learn`
  - `networkx`

To install the required dependencies, run:

```bash
pip install -r requirements.txt
```

## Usage

### Clustering Tendency Assessment (PHI)

To assess the clustering tendency of a dataset using the Proximal Homogeneity Index (PHI) and the Hopkins statistics, run the `phi.py` script. Example:

```bash
python experiments/phi.py --dataset_name="grid"
```

This will construct the PC Graph for your dataset and compute the PHI score, giving insights into whether the dataset contains inherent clusters.

### Cluster Validation (PSI)

To evaluate the quality of clusters produced by various clustering algorithms using the Partitioning Sensitivity Index (PSI), run the `psi.py` script. Example:

```bash
python experiments/psi.py
```

This will compute the PSI score for the clustering results, as well as different internal and external clustering validation metrics, helping assess the quality of the clusters discovered by various clustering algorithms.

## Citation

<pre><code>
@article{diallo2024deciphering,
  title={Deciphering Clusters With a Deterministic Measure of Clustering Tendency},
  author={Diallo, Alec F. and Patras, Paul},
  journal={IEEE Transactions on Knowledge and Data Engineering},
  year={2024},
  volume={36},
  number={4},
  pages={1489-1501}
}
</code></pre>
