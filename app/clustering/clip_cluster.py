"""HDBSCAN over CLIP embeddings."""

from __future__ import annotations

import hdbscan
import numpy as np

from app.embedding.clip import bytes_to_vec


def cluster_embeddings(
    hashes: list[str], vecs: list[bytes], min_cluster_size: int = 2
) -> dict[str, int]:
    if not hashes:
        return {}
    matrix = np.stack([bytes_to_vec(v).astype(np.float32) for v in vecs])
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size, metric="euclidean", cluster_selection_method="eom"
    )
    # CLIP features are L2-normalized; euclidean distance ~ angular for unit vectors.
    labels = clusterer.fit_predict(matrix)
    return {h: int(label) for h, label in zip(hashes, labels, strict=True)}
