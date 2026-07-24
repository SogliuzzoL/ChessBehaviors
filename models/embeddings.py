from typing import Any

import torch
from torch import nn


class DynamicPlayerEmbedding(nn.Embedding):
    def __init__(self, base_embeddings: nn.Embedding, n_players: int) -> None:
        total_embeddings = base_embeddings.num_embeddings + n_players
        super().__init__(
            num_embeddings=total_embeddings, embedding_dim=base_embeddings.embedding_dim
        )
        self.weight = nn.Parameter(torch.empty(0))
        self.base_embeddings: nn.Embedding = base_embeddings
        self.base_embeddings.requires_grad_(False)
        self.max_base_idx: int = base_embeddings.num_embeddings - 1
        self.dim: int = base_embeddings.embedding_dim
        self.players_embeddings = nn.Embedding(n_players, self.dim)

        with torch.no_grad():
            best_weights: Any = (
                self.base_embeddings.weight[self.max_base_idx].detach().clone()
            )
            self.players_embeddings.weight.data = best_weights.repeat(n_players, 1)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        is_player = input > self.max_base_idx
        out = torch.zeros(*input.shape, self.dim, device=input.device)
        if (~is_player).any():
            out[~is_player] = self.base_embeddings(input[~is_player])
        if is_player.any():
            shifted_indices = input[is_player] - (self.max_base_idx + 1)
            out[is_player] = self.players_embeddings(shifted_indices)
        return out
