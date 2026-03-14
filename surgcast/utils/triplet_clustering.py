from __future__ import annotations

from typing import Dict, List

import numpy as np


def compute_cooccurrence_matrix(
    triplet_labels: np.ndarray,
) -> np.ndarray:
    """Compute triplet co-occurrence matrix from training data.

    Args:
        triplet_labels: [N, T, num_triplets] binary triplet presence across videos/frames

    Returns:
        S_cooc: [num_triplets, num_triplets] normalized co-occurrence matrix.
    """
    # labels shape: [N, T, V] where V = num_triplets
    # Flatten to [N*T, V], compute co-occurrence: S = (X^T @ X) / N_frames
    N, T, V = triplet_labels.shape
    X = triplet_labels.reshape(-1, V)  # [N*T, V]
    cooc = X.T @ X  # [V, V]
    # Normalize by number of frames where at least one triplet is active
    active_frames = (X.sum(axis=1) > 0).sum()
    if active_frames > 0:
        cooc = cooc / active_frames
    # Normalize to [0, 1] range
    diag = np.sqrt(np.diag(cooc)).clip(1e-8)
    S_cooc = cooc / np.outer(diag, diag)
    np.fill_diagonal(S_cooc, 1.0)
    return S_cooc


def compute_semantic_embeddings(
    triplet_names: List[str],
    model_name: str = "all-MiniLM-L6-v2",
) -> np.ndarray:
    """Compute semantic embeddings for triplet names using a sentence model.

    Args:
        triplet_names: list of triplet name strings
        model_name: sentence transformer model name

    Returns:
        S_sem: [num_triplets, embed_dim] semantic embeddings.
    """
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    embeddings = model.encode(triplet_names, show_progress_bar=False)
    return np.array(embeddings)


def hybrid_clustering(
    S_cooc: np.ndarray,
    S_sem: np.ndarray,
    alpha: float = 0.7,
    n_clusters: int = 18,
) -> np.ndarray:
    """Hybrid clustering combining co-occurrence and semantic similarity.

    Similarity = alpha * S_cooc + (1 - alpha) * cosine_sim(S_sem).

    Args:
        S_cooc: [V, V] co-occurrence similarity matrix
        S_sem: [V, D] semantic embeddings
        alpha: weight for co-occurrence vs semantic
        n_clusters: number of output clusters

    Returns:
        assignments: [V] cluster assignment for each triplet.
    """
    from sklearn.cluster import SpectralClustering
    # Compute cosine similarity from semantic embeddings
    norms = np.linalg.norm(S_sem, axis=1, keepdims=True).clip(1e-8)
    S_sem_normed = S_sem / norms
    S_cos = S_sem_normed @ S_sem_normed.T
    # Hybrid similarity
    S_hybrid = alpha * S_cooc + (1 - alpha) * S_cos
    # Ensure symmetric and non-negative
    S_hybrid = (S_hybrid + S_hybrid.T) / 2
    S_hybrid = np.clip(S_hybrid, 0, None)
    # Spectral clustering on affinity matrix
    clustering = SpectralClustering(
        n_clusters=n_clusters, affinity="precomputed", random_state=42,
    )
    assignments = clustering.fit_predict(S_hybrid)
    return assignments


def validate_groups(
    assignments: np.ndarray,
    triplet_names: List[str],
) -> Dict[str, list]:
    """Validate triplet group assignments.

    Args:
        assignments: [V] cluster assignment for each triplet
        triplet_names: list of triplet name strings

    Returns:
        Dict with 'groups' (list of lists of triplet names),
        'sizes' (list of group sizes), 'singletons' (list of solo triplets).
    """
    groups: Dict[str, list] = {}
    for idx, cluster_id in enumerate(assignments):
        key = f"G{cluster_id}"
        groups.setdefault(key, []).append(triplet_names[idx])
    sizes = [len(g) for g in groups.values()]
    singletons = [names[0] for names in groups.values() if len(names) == 1]
    return {
        "groups": list(groups.values()),
        "sizes": sizes,
        "singletons": singletons,
    }
