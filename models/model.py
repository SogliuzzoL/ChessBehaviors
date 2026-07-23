from abc import ABC, abstractmethod
from typing import Tuple

import chess
import torch
import torch.nn as nn
import torch.optim as optim
from maia2 import inference


class ChessModel(ABC):
    @abstractmethod
    def predict(self, board: chess.Board) -> Tuple[dict[str, float], float]:
        pass


class Maia2(ChessModel):
    def __init__(self, model, pruning_fn=None):
        self.model = model
        self.prepared = inference.prepare()
        self.pruning_fn = pruning_fn

    def predict(self, board: chess.Board) -> Tuple[dict[str, float], float]:
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

    def predict(self, board: chess.Board) -> Tuple[dict[str, float], float]:
        from search.descent import DescentSearch

        search_engine = DescentSearch(model=self.base_model)
        policy_probs, root_value = search_engine.search(
            board, max_iterations=self.max_iterations, timeout=self.timeout
        )
        return policy_probs, root_value


class Maia2FT(ChessModel):
    def __init__(self, model, n_players: int, pruning_fn=None):
        self.model = model
        self.prepared = inference.prepare()
        self.pruning_fn = pruning_fn

        original_emb = getattr(
            self.model,
            "elo_embedding",
            getattr(getattr(self.model, "net", None), "elo_embedding", None),
        )
        self.max_maia_idx = original_emb.num_embeddings - 1

        self.custom_emb = PlayerStyleEmbedding(original_emb, n_players)
        if hasattr(self.model, "elo_embedding"):
            self.model.elo_embedding = self.custom_emb
        else:
            self.model.net.elo_embedding = self.custom_emb

    def train_player_embedding(
        self,
        player_index: int,
        dataloader: torch.utils.data.DataLoader,
        epochs: int = 5,
        lr: float = 1e-3,
    ):
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        self.custom_emb.players_embeddings.weight.requires_grad = True

        optimizer = optim.Adam([self.custom_emb.players_embeddings.weight], lr=lr)
        criterion = nn.CrossEntropyLoss()

        virtual_elo_idx = self.max_maia_idx + 1 + player_index

        for epoch in range(epochs):
            for batch_boards, batch_move_targets in dataloader:
                optimizer.zero_grad()
                logits = self.model.forward_board(batch_boards, virtual_elo_idx)
                loss = criterion(logits, batch_move_targets)
                loss.backward()
                optimizer.step()

    def predict(
        self, board: chess.Board, player_index: int = 0
    ) -> Tuple[dict[str, float], float]:
        virtual_elo_idx = self.max_maia_idx + 1 + player_index

        raw_moves, value = inference.inference_each(
            self.model, self.prepared, board.fen(), virtual_elo_idx, virtual_elo_idx
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
