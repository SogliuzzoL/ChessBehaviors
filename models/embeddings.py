"""
Dynamic player embedding layer for subject-conditioned neural network representations.

This module provides a unified embedding interface that seamlessly integrates
frozen pre-trained base embeddings with dynamically initialized, learnable subject-specific
embedding vectors.
"""

from typing import Any

import torch
from torch import nn


class DynamicPlayerEmbedding(nn.Embedding):
    """Dynamic embedding container combining static base embeddings with trainable subject embeddings.

    Extends `nn.Embedding` to route forward pass queries between a frozen dictionary of static
    pre-trained base embeddings and a trainable matrix of subject-conditioned vectors.
    Player embeddings are initialized dynamically using the terminal weight vector
    of the pre-trained embedding table.

    Attributes:
        base_embeddings (nn.Embedding): Frozen pre-trained embedding table.
        max_base_idx (int): Maximum index threshold distinguishing base tokens from subjects.
        dim (int): Embedding vector space dimensionality.
        players_embeddings (nn.Embedding): Trainable subject-specific embedding table.
    """

    def __init__(self, base_embeddings: nn.Embedding, n_players: int) -> None:
        """Initialize the DynamicPlayerEmbedding layer.

        Args:
            base_embeddings (nn.Embedding): Pre-trained base embedding layer instance.
            n_players (int): Total count of trainable subject profiles to instantiate.
        """
        total_embeddings: int = base_embeddings.num_embeddings + n_players
        super().__init__(
            num_embeddings=total_embeddings, embedding_dim=base_embeddings.embedding_dim
        )
        self.weight = nn.Parameter(torch.empty(0))
        self.base_embeddings: nn.Embedding = base_embeddings
        self.base_embeddings.requires_grad_(False)
        self.max_base_idx: int = base_embeddings.num_embeddings - 1
        self.dim: int = base_embeddings.embedding_dim
        self.players_embeddings = nn.Embedding(n_players, self.dim)

        # Initialize subject vectors from the terminal weight vector of the base embedding table
        with torch.no_grad():
            best_weights: Any = (
                self.base_embeddings.weight[self.max_base_idx].detach().clone()
            )
            self.players_embeddings.weight.data = best_weights.repeat(n_players, 1)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Perform forward embedding lookup by conditionally routing index tensors.

        Args:
            input (torch.Tensor): Tensor containing numerical token/subject indices.

        Returns:
            torch.Tensor: Combined output tensor containing retrieved embedding vectors.
        """
        is_player: torch.Tensor = input > self.max_base_idx
        out: torch.Tensor = torch.zeros(*input.shape, self.dim, device=input.device)

        # Retrieve vectors from frozen base embedding matrix
        if (~is_player).any():
            out[~is_player] = self.base_embeddings(input[~is_player])

        # Retrieve vectors from trainable subject embedding matrix
        if is_player.any():
            shifted_indices: torch.Tensor = input[is_player] - (self.max_base_idx + 1)
            out[is_player] = self.players_embeddings(shifted_indices)

        return out
