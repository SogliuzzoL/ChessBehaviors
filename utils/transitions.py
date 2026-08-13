"""
Utility module for processing board state transitions into high-dimensional vector representations.
"""

import chess
import numpy as np
from maia2.utils import board_to_tensor

# Global lookup cache for transition vectors: (fen, move_uci) -> 2304-D vector
TRANSITION_CACHE: dict[tuple[str, str], np.ndarray] = {}


def get_transition_vector(fen: str, move_uci: str) -> np.ndarray | None:
    """
    Computes and caches the 2304-dimensional board state transition vector
    for a given board position (FEN) and played move (UCI).

    Args:
        fen (str): Board state in Forsyth-Edwards Notation.
        move_uci (str): Played move in Universal Chess Interface format.

    Returns:
        Optional[np.ndarray]: Flat 2304-dimensional float32 vector concatenating
                              the pre-move and post-move board representations,
                              or None if the move is illegal or unparseable.
    """
    cache_key = (fen, move_uci)
    if cache_key in TRANSITION_CACHE:
        return TRANSITION_CACHE[cache_key]

    board_before = chess.Board(fen)
    move = chess.Move.from_uci(move_uci)
    if move not in board_before.legal_moves:
        return None

    board_after = board_before.copy()
    board_after.push(move)

    # Enforce board orientation symmetry for Black to move
    if board_before.turn == chess.BLACK:
        board_before = board_before.mirror()
        board_after = board_after.mirror()

    # Extract 1152-D spatial representations using Maia-2 pre-processing
    t_before = board_to_tensor(board_before).cpu().numpy().flatten()
    t_after = board_to_tensor(board_after).cpu().numpy().flatten()

    # Concatenate pre- and post-move representations into a 2304-D transition vector
    vec = np.concatenate([t_before, t_after]).astype(np.float32)
    TRANSITION_CACHE[cache_key] = vec
    return vec
