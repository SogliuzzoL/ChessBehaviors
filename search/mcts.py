import time
from dataclasses import dataclass, field

import chess
import numpy as np

from models import ChessModel


@dataclass
class MCTSNode:
    v: float = 0.0
    n_sum: int = 0
    n: dict[str, int] = field(default_factory=dict)
    w: dict[str, float] = field(default_factory=dict)
    p: dict[str, float] = field(default_factory=dict)
    is_terminal: bool = False
    terminal_value: float = 0.0


class MCTSSearch:
    def __init__(self, model: ChessModel, cpuct: float = 1.5):
        self.model = model
        self.cpuct = cpuct
        self.T: dict[str, MCTSNode] = {}

    def _get_terminal_gain(self, board: chess.Board) -> float:
        outcome = board.outcome()
        if outcome is None or outcome.winner is None:
            return 0.0
        return 1.0 if outcome.winner == chess.WHITE else -1.0

    def _select_puct_action(self, board: chess.Board, node: MCTSNode) -> str:
        is_white = board.turn == chess.WHITE
        best_action = None
        best_puct = -float("inf") if is_white else float("inf")
        sqrt_total = np.sqrt(max(1, node.n_sum))

        for a, prob in node.p.items():
            visit_count = node.n[a]
            q_val = node.w[a] / visit_count if visit_count > 0 else node.v
            u_val = self.cpuct * prob * sqrt_total / (1 + visit_count)
            score = q_val + u_val if is_white else q_val - u_val

            if is_white:
                if score > best_puct:
                    best_puct = score
                    best_action = a
            else:
                if score < best_puct:
                    best_puct = score
                    best_action = a

        return best_action or list(node.p.keys())[0]

    def _simulation(self, board: chess.Board) -> None:
        path = []
        curr_board = board.copy()
        s = curr_board.fen()

        while True:
            if curr_board.is_game_over():
                if s not in self.T:
                    gain = self._get_terminal_gain(curr_board)
                    self.T[s] = MCTSNode(is_terminal=True, terminal_value=gain)
                break

            if s not in self.T:
                moves_probs, value = self.model.predict(curr_board)
                self.T[s] = MCTSNode(
                    v=value,
                    n={m: 0 for m in moves_probs},
                    w={m: 0.0 for m in moves_probs},
                    p=moves_probs,
                )
                break

            node = self.T[s]
            if not node.p:
                break

            best_action = self._select_puct_action(curr_board, node)
            path.append((s, best_action))
            curr_board.push_uci(best_action)
            s = curr_board.fen()

        if s in self.T and self.T[s].is_terminal:
            v = self.T[s].terminal_value
        elif s in self.T:
            v = self.T[s].v
        else:
            v = 0.0

        for state_fen, action in reversed(path):
            node = self.T[state_fen]
            node.n_sum += 1
            node.n[action] += 1
            node.w[action] += v

    def get_policy(self, board: chess.Board) -> dict[str, float]:
        s = board.fen()
        if s not in self.T or not self.T[s].n:
            return {}
        node = self.T[s]
        total_visits = sum(node.n.values())
        if total_visits == 0:
            return {m: prob for m, prob in node.p.items()}
        return {m: visits / total_visits for m, visits in node.n.items()}

    def search(
        self, board: chess.Board, max_iterations: int = 50, timeout: float = 1.0
    ) -> tuple[dict[str, float], float]:
        start_time = time.time()
        for _ in range(max_iterations):
            if time.time() - start_time >= timeout:
                break
            self._simulation(board)

        policy = self.get_policy(board)
        root_value = self.T.get(board.fen(), MCTSNode()).v
        return policy, root_value
