import chess
import numpy as np
from maia2.utils import board_to_tensor

# Cache global des transitions (fen, move) -> vecteur 2304-D
transition_cache: dict[tuple[str, str], np.ndarray] = {}


def get_transition_vector(fen: str, move_uci: str) -> np.ndarray | None:
    """Calcule et met en cache le vecteur de transition 2304-D pour un couple (FEN, coup)."""
    cache_key = (fen, move_uci)
    if cache_key in transition_cache:
        return transition_cache[cache_key]

    try:
        board_before = chess.Board(fen)
        move = chess.Move.from_uci(move_uci)
        if move not in board_before.legal_moves:
            return None

        board_after = board_before.copy()
        board_after.push(move)

        if board_before.turn == chess.BLACK:
            board_before = board_before.mirror()
            board_after = board_after.mirror()

        t_before = board_to_tensor(board_before).cpu().numpy().flatten()
        t_after = board_to_tensor(board_after).cpu().numpy().flatten()

        vec = np.concatenate([t_before, t_after]).astype(np.float32)
        transition_cache[cache_key] = vec
        return vec
    except Exception:
        return None
