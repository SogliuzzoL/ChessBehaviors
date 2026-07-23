from abc import ABC, abstractmethod
from typing import Any, Tuple

import chess
import pandas as pd
import polars as pl
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


class PlayerStyleEmbedding(nn.Embedding):
    def __init__(self, elo_embeddings: nn.Embedding, n_players: int) -> None:
        total_embeddings = elo_embeddings.num_embeddings + n_players
        super().__init__(
            num_embeddings=total_embeddings, embedding_dim=elo_embeddings.embedding_dim
        )
        self.weight = nn.Parameter(torch.empty(0))
        self.elo_embeddings: nn.Embedding = elo_embeddings
        self.elo_embeddings.requires_grad_(False)
        self.max_maia_idx: int = elo_embeddings.num_embeddings - 1
        self.dim: int = elo_embeddings.embedding_dim
        self.players_embeddings = nn.Embedding(n_players, self.dim)

        with torch.no_grad():
            best_weights: Any = (
                self.elo_embeddings.weight[self.max_maia_idx].detach().clone()
            )
            self.players_embeddings.weight.data = best_weights.repeat(n_players, 1)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        is_player = input > self.max_maia_idx
        out = torch.zeros(*input.shape, self.dim, device=input.device)
        if (~is_player).any():
            out[~is_player] = self.elo_embeddings(input[~is_player])
        if is_player.any():
            shifted_indices = input[is_player] - (self.max_maia_idx + 1)
            out[is_player] = self.players_embeddings(shifted_indices)
        return out


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

    def reset_player_embedding(self, player_index: int):
        with torch.no_grad():
            init_weights = (
                self.custom_emb.elo_embeddings.weight[self.max_maia_idx]
                .detach()
                .clone()
            )
            self.custom_emb.players_embeddings.weight.data[player_index] = init_weights

    def fit_player(
        self,
        player_index: int,
        train_pos: pl.DataFrame,
        epochs: int = 3,
        lr: float = 1e-3,
    ):
        if len(train_pos) == 0:
            return

        virtual_elo_idx = self.max_maia_idx + 1 + player_index

        train_df = (
            train_pos.rename({"fen": "board"})
            .with_columns(
                [
                    pl.lit(virtual_elo_idx).alias("active_elo"),
                    pl.lit(virtual_elo_idx).alias("opponent_elo"),
                ]
            )
            .select(["board", "move", "active_elo", "opponent_elo"])
            .to_pandas()
        )

        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        self.custom_emb.players_embeddings.weight.requires_grad = True

        optimizer = optim.Adam([self.custom_emb.players_embeddings.weight], lr=lr)
        criterion = nn.CrossEntropyLoss()

        prepared_batch = self.prepared.prepare_batch(train_df)

        for epoch in range(epochs):
            optimizer.zero_grad()
            output = self.model(prepared_batch)
            logits = output.move_logits
            targets = prepared_batch.move_targets

            loss = criterion(logits, targets)
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
