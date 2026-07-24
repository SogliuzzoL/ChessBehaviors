import time
from dataclasses import dataclass, field

import chess
import numpy as np

from models import ChessModel


@dataclass
class TranspositionNode:
    v: float = 0.0
    c: float = 0.0
    r: int = 0
    n: dict[str, int] = field(default_factory=dict)


class DescentSearch:
    def __init__(self, model: ChessModel):
        self.model = model
        self.T: dict[str, TranspositionNode] = {}

    def _get_terminal_gain(self, board: chess.Board) -> float:
        outcome = board.outcome()
        if outcome is None or outcome.winner is None:
            return 0.0
        return 1.0 if outcome.winner == chess.WHITE else -1.0

    def _get_child_node(
        self, board: chess.Board, move_uci: str
    ) -> tuple[str, TranspositionNode]:
        board.push_uci(move_uci)
        child_s = board.fen()
        board.pop()
        return child_s, self.T[child_s]

    def _select_best_action(
        self, board: chess.Board, moves: list[str], dual: bool = False
    ) -> str:
        s = board.fen()
        is_white = board.turn == chess.WHITE

        def key_fn(move: str) -> tuple[float, float, float]:
            _, child_node = self._get_child_node(board, move)
            n_val = float(self.T[s].n.get(move, 0))
            if dual:
                n_val = -n_val
            return (child_node.c, child_node.v, n_val if is_white else -n_val)

        return max(moves, key=key_fn) if is_white else min(moves, key=key_fn)

    def _backup_resolution(self, board: chess.Board, moves: list[str]) -> int:
        s = board.fen()
        if abs(self.T[s].c) == 1.0:
            return 1

        children_r = [self._get_child_node(board, m)[1].r for m in moves]
        return min(children_r) if children_r else 0

    def _update_node_values(self, board: chess.Board, moves: list[str]) -> None:
        s = board.fen()
        best_move = self._select_best_action(board, moves, dual=False)
        _, best_child = self._get_child_node(board, best_move)

        self.T[s].c = best_child.c
        self.T[s].v = best_child.v
        self.T[s].r = self._backup_resolution(board, moves)

    def iteration(self, board: chess.Board) -> None:
        s = board.fen()

        if board.is_game_over():
            gain = self._get_terminal_gain(board)
            self.T[s] = TranspositionNode(v=gain, c=gain, r=1)
            return

        if s not in self.T:
            moves_probs, value = self.model.predict(board)
            self.T[s] = TranspositionNode(
                v=value, c=0.0, r=0, n={m: 0 for m in moves_probs}
            )

            for move_uci, prob in moves_probs.items():
                board.push_uci(move_uci)
                child_s = board.fen()

                if board.is_game_over():
                    gain = self._get_terminal_gain(board)
                    self.T[child_s] = TranspositionNode(v=gain, c=gain, r=1)
                elif child_s not in self.T:
                    self.T[child_s] = TranspositionNode(v=prob, c=0.0, r=0)

                board.pop()

            legal_moves = list(moves_probs.keys())
            if legal_moves:
                self._update_node_values(board, legal_moves)
            return

        if self.T[s].r == 0:
            legal_moves = list(self.T[s].n.keys())
            unresolved = [
                m for m in legal_moves if self._get_child_node(board, m)[1].r == 0
            ]

            if unresolved:
                selected_move = self._select_best_action(board, unresolved, dual=True)
                self.T[s].n[selected_move] += 1

                board.push_uci(selected_move)
                self.iteration(board)
                board.pop()

                self._update_node_values(board, legal_moves)

    def get_policy(
        self, board: chess.Board, temperature: float = 1.0
    ) -> dict[str, float]:
        s = board.fen()
        if s not in self.T or not self.T[s].n:
            return {}

        moves = list(self.T[s].n.keys())
        values = [self._get_child_node(board, m)[1].v for m in moves]

        values_arr = np.array(values, dtype=np.float64) / max(temperature, 1e-6)
        exp_values = np.exp(values_arr - np.max(values_arr))
        probs = exp_values / np.sum(exp_values)

        return {move: float(p) for move, p in zip(moves, probs)}

    def search(
        self, board: chess.Board, max_iterations: int = 100, timeout: float = 2.0
    ) -> tuple[dict[str, float], float]:
        start_time = time.time()
        for _ in range(max_iterations):
            if (
                time.time() - start_time >= timeout
                or self.T.get(board.fen(), TranspositionNode()).r == 1
            ):
                break
            self.iteration(board)

        policy = self.get_policy(board)
        root_value = self.T.get(board.fen(), TranspositionNode()).v
        return policy, root_value
