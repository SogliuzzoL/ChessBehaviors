from abc import ABC, abstractmethod

import chess
from maia2 import inference


class ChessModel(ABC):
    @abstractmethod
    def predict(self, board: chess.Board) -> tuple[dict[str, float], float]:
        pass


class Maia2(ChessModel):
    def __init__(self, model, pruning_fn=None):
        self.model = model
        self.prepared = inference.prepare()
        self.pruning_fn = pruning_fn

    def predict(self, board: chess.Board) -> tuple[dict[str, float], float]:
        raw_moves, value = inference.inference_each(
            self.model, self.prepared, board.fen(), 2500, 2500
        )

        legal_uci_moves = {m.uci() for m in board.legal_moves}
        legal_moves_dict = {
            move: score for move, score in raw_moves.items() if move in legal_uci_moves
        }

        if self.pruning_fn and legal_moves_dict:
            moves_pruned = self.pruning_fn(legal_moves_dict)
            legal_moves_dict = {
                move: legal_moves_dict[move]
                for move in moves_pruned
                if move in legal_moves_dict
            }

        total = sum(legal_moves_dict.values())
        if total > 0:
            legal_moves_dict = {
                move: score / total for move, score in legal_moves_dict.items()
            }

        return legal_moves_dict, value


class DescentModelWrapper(ChessModel):
    def __init__(
        self, base_model: ChessModel, max_iterations: int = 100, timeout: float = 2.0
    ):
        self.base_model = base_model
        self.max_iterations = max_iterations
        self.timeout = timeout

    def predict(self, board: chess.Board) -> tuple[dict[str, float], float]:
        from search.descent import DescentSearch

        search_engine = DescentSearch(model=self.base_model)
        policy_probs, root_value = search_engine.search(
            board, max_iterations=self.max_iterations, timeout=self.timeout
        )
        return policy_probs, root_value
