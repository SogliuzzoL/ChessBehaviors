"""
Policy pruning algorithms module providing probability mass truncation functions,
including nucleus (top-p) pruning and information-theoretic divergence-bounded pruning.
"""

import math

import numpy as np


def nucleus_prune(moves: dict[str, float], top_p: float = 0.95) -> list[str]:
    """Perform nucleus (top-p) action space pruning on a move probability distribution.

    Selects the minimal set of candidate moves whose cumulative probability mass meets or exceeds
    the target threshold parameter `top_p`.

    Args:
        moves (Dict[str, float]): Dictionary mapping candidate move UCI strings to posterior probabilities.
        top_p (float, optional): Cumulative probability mass threshold limit. Defaults to 0.95.

    Returns:
        List[str]: List of candidate move UCI strings forming the pruned action space nucleus.
    """
    sorted_moves: list[tuple[str, float]] = sorted(
        moves.items(), key=lambda x: x[1], reverse=True
    )
    p_probs: np.ndarray = np.array([prob for _, prob in sorted_moves], dtype=np.float64)

    total_mass = np.sum(p_probs)
    if total_mass > 0:
        p_probs /= total_mass

    for k in range(1, len(sorted_moves) + 1):
        mass_k = np.sum(p_probs[:k])

        if mass_k >= top_p:
            return [move for move, _ in sorted_moves[:k]]

    return [move for move, _ in sorted_moves]


def information_prune(moves: dict[str, float], epsilon: float = 0.05) -> list[str]:
    """Perform information-theoretic policy space pruning bounded by KL divergence.

    Truncates candidate move distributions when the Kullback-Leibler divergence penalty
    attributable to probability mass exclusion falls below a specified tolerance threshold `epsilon`.

    Args:
        moves (Dict[str, float]): Dictionary mapping candidate move UCI strings to posterior probabilities.
        epsilon (float, optional): Maximum allowable Kullback-Leibler divergence bound. Defaults to 0.05.

    Returns:
        List[str]: List of candidate move UCI strings retained within the bounded action space.
    """
    sorted_moves: list[tuple[str, float]] = sorted(
        moves.items(), key=lambda x: x[1], reverse=True
    )
    p_probs: np.ndarray = np.array([prob for _, prob in sorted_moves], dtype=np.float64)

    total_mass = np.sum(p_probs)
    if total_mass > 0:
        p_probs /= total_mass

    for k in range(1, len(sorted_moves) + 1):
        mass_k = float(np.sum(p_probs[:k]))

        if mass_k >= 1.0 - 1e-9:
            return [move for move, _ in sorted_moves[:k]]

        kl_divergence = -math.log(mass_k)
        if kl_divergence <= epsilon:
            return [move for move, _ in sorted_moves[:k]]

    return [move for move, _ in sorted_moves]
