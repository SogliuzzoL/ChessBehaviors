import math

import numpy as np


def nucleus_prune(moves: dict[str, float], top_p: float = 0.95) -> list[str]:
    sorted_moves = sorted(moves.items(), key=lambda x: x[1], reverse=True)
    p_probs = np.array([prob for _, prob in sorted_moves], dtype=np.float64)
    p_probs /= np.sum(p_probs)

    for k in range(1, len(sorted_moves) + 1):
        mass_k = np.sum(p_probs[:k])

        if mass_k >= top_p:
            return [move for move, _ in sorted_moves[:k]]

    return [move for move, _ in sorted_moves]


def information_prune(moves: dict[str, float], epsilon: float = 0.05) -> list[str]:
    sorted_moves = sorted(moves.items(), key=lambda x: x[1], reverse=True)
    p_probs = np.array([prob for _, prob in sorted_moves], dtype=np.float64)
    p_probs /= np.sum(p_probs)

    for k in range(1, len(sorted_moves) + 1):
        mass_k = np.sum(p_probs[:k])

        if mass_k >= 1.0 - 1e-9:
            return [move for move, _ in sorted_moves[:k]]

        kl_divergence = -math.log(mass_k)
        if kl_divergence <= epsilon:
            return [move for move, _ in sorted_moves[:k]]

    return [move for move, _ in sorted_moves]
