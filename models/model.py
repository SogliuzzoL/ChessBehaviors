from abc import ABC, abstractmethod
from typing import Tuple

import chess
from maia2 import inference


class ChessModel(ABC):
    @abstractmethod
    def predict(self, board: chess.Board) -> Tuple[dict[str, float], float]:
        pass

class Maia2(ChessModel):
    def __init__(self, model, pruning_fn):
        self.model = model
        self.prepared = inference.prepare()
        self.pruning_fn = pruning_fn

    def predict(self, board: chess.Board) -> Tuple[dict[str, float], float]:
        moves, value = inference.inference_each(self.model, self.prepared, board.fen(), 2500, 2500)

        if self.pruning_fn:
            moves_pruned = self.pruning_fn(moves)
            moves = {move: moves[move] for move in moves_pruned}

        total = sum(moves.values())
        moves = {move: score / total for move, score in moves.items()}

        return moves, value
