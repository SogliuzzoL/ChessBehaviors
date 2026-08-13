"""
Utility module for converting chess board state transitions into high-dimensional feature representations.
"""

import chess
import numpy as np
from maia2.utils import board_to_tensor

# Global cache for transition representations mapping (FEN, UCI move) to 2304-dimensional vectors
TRANSITION_CACHE: dict[tuple[str, str], np.ndarray] = {}


def get_transition_vector(fen: str, move_uci: str) -> np.ndarray | None:
    """Compute and cache a 2304-dimensional vector representing a board state transition.

    Concatenates the pre-move and post-move spatial tensor representations into a unified
    feature vector for downstream stylometric and representation analysis.

    Args:
        fen (str): Initial board position in Forsyth-Edwards Notation.
        move_uci (str): Candidate action in Universal Chess Interface (UCI) string format.

    Returns:
        Optional[np.ndarray]: Flattened 2304-dimensional float32 vector concatenation
            of pre-move and post-move board states, or None if the action is illegal or unparseable.
    """
    cache_key: tuple[str, str] = (fen, move_uci)
    if cache_key in TRANSITION_CACHE:
        return TRANSITION_CACHE[cache_key]

    board_before = chess.Board(fen)
    move = chess.Move.from_uci(move_uci)
    if move not in board_before.legal_moves:
        return None

    board_after = board_before.copy()
    board_after.push(move)

    # Normalize board perspective symmetry when Black is the active player
    if board_before.turn == chess.BLACK:
        board_before = board_before.mirror()
        board_after = board_after.mirror()

    # Extract 1152-dimensional spatial representations via Maia-2 preprocessing
    t_before: np.ndarray = board_to_tensor(board_before).cpu().numpy().flatten()
    t_after: np.ndarray = board_to_tensor(board_after).cpu().numpy().flatten()

    # Concatenate pre-move and post-move representations into a 2304-dimensional transition vector
    vec: np.ndarray = np.concatenate([t_before, t_after]).astype(np.float32)
    TRANSITION_CACHE[cache_key] = vec
    return vec
