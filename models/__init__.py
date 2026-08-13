"""
Models module initialization for the chess style and behavior evaluation framework.

This module exposes core interface abstractions, neural model wrappers (Maia-2 baselines
and fine-tuned variants), parameter-efficient adapters (MoE-LoRA), search-based inference
wrappers (Descent, MCTS), and dataset primitives utilized across behavioral evaluation pipelines.
"""

from models.adapters.moe_lora import Maia2MoELoRA, PlayerMoEAdapter
from models.base import ChessModel, DescentModelWrapper, Maia2, MCTSModelWrapper
from models.dataset import PlayerTrainDataset
from models.embeddings import DynamicPlayerEmbedding
from models.maia_ft import Maia2FineTuned

__all__: list[str] = [
    "ChessModel",
    "DescentModelWrapper",
    "DynamicPlayerEmbedding",
    "MCTSModelWrapper",
    "Maia2",
    "Maia2FineTuned",
    "Maia2MoELoRA",
    "PlayerMoEAdapter",
    "PlayerTrainDataset",
]
