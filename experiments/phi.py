import sys
from pathlib import Path

root_path = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root_path))

import argparse
from time import time

from core.stats import CtaStats
from helpers.data import get_dataset
from helpers.stats import hopkins


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute PHI and hopkins statistics for a dataset.")

    dataset_choices = [
        "grid", "random", "iris", "digits", "blobs", "sine/cosine",
        "2 moons", "circle", "2 circles", "5 circles"
    ]

    parser.add_argument(
        '--dataset_name',
        type=str,
        choices=dataset_choices,
        help='Name of the dataset to process',
        required=True
    )

    args = parser.parse_args()

    dataset_name = args.dataset_name
    dataset = get_dataset(dataset_name)
    samples, labels = dataset

    print(f"Evaluating clusterability statistics on dataset: {dataset_name} [{len(samples)} samples].")

    print(f"Computing PHI...")
    start_time = time()
    cta_stats = CtaStats(data=dataset)
    phi_score = cta_stats.compute_phi()
    end_time = time()

    cta_phi_time = end_time - start_time

    print(f"Computing Hopkins...")
    start_time = time()
    hopkins_score = hopkins(samples)
    end_time = time()

    hopkins_time = end_time - start_time

    print(f"\nResults: PHI Score={phi_score:.2f} | Hopkins={hopkins_score:.2f}")

    print(f"Runtime in seconds: PHI Score: {cta_phi_time:.4f} | Hopkins: {hopkins_time:.4f}")
