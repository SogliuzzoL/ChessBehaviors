"""
Subject-level fine-tuning module for adapting the Maia-2 architecture using dynamic player embeddings.
"""

from collections.abc import Callable
from typing import Any

import chess
import polars as pl
import torch
import tqdm
from maia2 import inference
from maia2.utils import (
    board_to_tensor,
    create_elo_dict,
    get_all_possible_moves,
    mirror_move,
)
from torch import nn, optim
from torch.utils.data import DataLoader

from models.base import ChessModel
from models.dataset import PlayerTrainDataset
from models.embeddings import DynamicPlayerEmbedding


class Maia2FineTuned(ChessModel):
    """Fine-tuned variant of the Maia-2 chess architecture incorporating subject-specific embeddings.

    Adapts the base model to individual decision-making styles by optimizing dynamic player
    embedding vectors while keeping the backbone network parameters frozen. Supports joint
    loss minimization incorporating cross-entropy policy loss and L2 anchor regularization.

    Attributes:
        model: Underlying pre-trained Maia-2 neural execution model.
        prepared: Pre-allocated inference context structures.
        pruning_fn (Optional[Callable[[Dict[str, float]], List[str]]]): Optional policy pruning transformation.
        all_moves_dict (Dict[str, int]): Vocabulary mapping UCI move strings to integer indices.
        elo_dict (Dict[Any, Any]): Subject rating metadata dictionary.
        max_maia_idx (int): Upper boundary index threshold for base Maia-2 rating embeddings.
        custom_emb (DynamicPlayerEmbedding): Dynamic embedding layer managing subject vectors.
    """

    def __init__(
        self,
        model: object,
        n_players: int,
        pruning_fn: Callable[[dict[str, float]], list[str]] | None = None,
    ) -> None:
        """Initialize the fine-tuned Maia-2 model instance.

        Args:
            model (object): Pre-trained Maia-2 model architecture instance.
            n_players (int): Total number of distinct subject profiles to accommodate.
            pruning_fn (Optional[Callable[[Dict[str, float]], List[str]]], optional): Function
                executing policy space pruning. Defaults to None.
        """
        self.model = model
        self.prepared = inference.prepare()
        self.pruning_fn = pruning_fn

        all_moves = get_all_possible_moves()
        self.all_moves_dict: dict[str, int] = {
            move: i for i, move in enumerate(all_moves)
        }
        self.elo_dict: dict[Any, Any] = create_elo_dict()

        original_emb = getattr(
            self.model,
            "elo_embedding",
            getattr(getattr(self.model, "net", None), "elo_embedding", None),
        )
        self.max_maia_idx: int = original_emb.num_embeddings - 1

        self.custom_emb = DynamicPlayerEmbedding(original_emb, n_players)
        if hasattr(self.model, "elo_embedding"):
            self.model.elo_embedding = self.custom_emb
        else:
            self.model.net.elo_embedding = self.custom_emb

    def reset_player_embedding(self, player_index: int) -> None:
        """Reset specified subject embedding vector to default baseline initialization.

        Args:
            player_index (int): Integer identifier for target subject cohort.
        """
        with torch.no_grad():
            init_weights = (
                self.custom_emb.base_embeddings.weight[self.max_maia_idx]
                .detach()
                .clone()
            )
            self.custom_emb.players_embeddings.weight.data[player_index] = init_weights

    def fit_player(
        self,
        player_index: int,
        train_pos: pl.DataFrame,
        test_pos: pl.DataFrame | None = None,
        epochs: int = 5,
        batch_size: int = 64,
        lr: float = 1e-2,
        l2_anchor_weight: float = 1e-5,
    ) -> list[dict[str, Any]]:
        """Optimize subject embedding vector on observational training positions.

        Executes iterative parameter updating for the target subject embedding vector using
        AdamW optimization and cosine annealing learning rate schedules. Evaluates intermediate
        convergence against validation sets if provided.

        Args:
            player_index (int): Target subject profile identifier.
            train_pos (pl.DataFrame): Dataframe containing training board positions.
            test_pos (Optional[pl.DataFrame], optional): Optional validation board positions. Defaults to None.
            epochs (int, optional): Optimization epoch count. Defaults to 5.
            batch_size (int, optional): Training mini-batch size. Defaults to 64.
            lr (float, optional): Initial peak learning rate. Defaults to 1e-2.
            l2_anchor_weight (float, optional): Penalty weight for L2 distance regularization to base embedding.
                Defaults to 1e-5.

        Returns:
            List[Dict[str, Any]]: History logs containing batch-level and epoch-level loss metrics.
        """
        if len(train_pos) == 0:
            return []

        virtual_elo_idx = self.max_maia_idx + 1 + player_index

        train_df = (
            train_pos.rename({"fen": "board"}).select(["board", "move"]).to_pandas()
        )

        dataset = PlayerTrainDataset(train_df, self.all_moves_dict, self.elo_dict)
        if len(dataset) == 0:
            return []

        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.model.train()
        for param in self.model.parameters():
            param.requires_grad = False
        self.custom_emb.players_embeddings.weight.requires_grad = True

        optimizer = optim.AdamW(
            [self.custom_emb.players_embeddings.weight], lr=lr, weight_decay=1e-4
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs * len(dataloader), eta_min=1e-4
        )
        criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

        device = next(self.model.parameters()).device

        with torch.no_grad():
            ref_elo_emb = (
                self.custom_emb.base_embeddings.weight[self.max_maia_idx]
                .detach()
                .clone()
            )

        history: list[dict[str, Any]] = []
        global_step = 0

        total_batches = epochs * len(dataloader)
        train_pbar = tqdm.tqdm(
            total=total_batches,
            desc=f"Optimizing Subject {player_index}",
            leave=False,
            unit="batch",
        )

        for epoch in range(epochs):
            running_loss = 0.0
            running_ce_loss = 0.0
            running_anchor_loss = 0.0
            steps = 0

            for boards, targets in dataloader:
                boards = boards.to(device)
                targets = targets.to(device)

                elos_self = torch.tensor([virtual_elo_idx] * len(boards), device=device)
                elos_oppo = torch.tensor([virtual_elo_idx] * len(boards), device=device)

                optimizer.zero_grad()

                logits_maia, _, _ = self.model(boards, elos_self, elos_oppo)

                ce_loss = criterion(logits_maia, targets)

                current_player_emb = self.custom_emb.players_embeddings.weight[
                    player_index
                ]
                anchor_loss = torch.mean((current_player_emb - ref_elo_emb) ** 2)

                total_loss = ce_loss + l2_anchor_weight * anchor_loss

                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [self.custom_emb.players_embeddings.weight], max_norm=1.0
                )
                optimizer.step()
                scheduler.step()

                loss_val = total_loss.item()
                ce_val = ce_loss.item()
                anchor_val = anchor_loss.item()

                running_loss += loss_val
                running_ce_loss += ce_val
                running_anchor_loss += anchor_val

                steps += 1
                global_step += 1

                history.append(
                    {
                        "type": "batch",
                        "player_index": player_index,
                        "epoch": epoch + 1,
                        "batch_step": steps,
                        "global_step": global_step,
                        "train_loss": loss_val,
                        "ce_loss": ce_val,
                        "anchor_loss": anchor_val,
                        "lr": scheduler.get_last_lr()[0],
                    }
                )

                train_pbar.set_postfix(
                    {
                        "epoch": f"{epoch + 1}/{epochs}",
                        "loss": f"{loss_val:.4f}",
                        "ce_loss": f"{ce_val:.4f}",
                        "avg": f"{running_loss / steps:.4f}",
                    }
                )
                train_pbar.update(1)

            epoch_log: dict[str, Any] = {
                "type": "epoch",
                "player_index": player_index,
                "epoch": epoch + 1,
                "avg_train_loss": running_loss / steps,
                "avg_ce_loss": running_ce_loss / steps,
                "avg_anchor_loss": running_anchor_loss / steps,
            }

            # Evaluate performance on validation dataset partition
            if test_pos is not None and len(test_pos) > 0:
                self.model.eval()
                test_df = (
                    test_pos.rename({"fen": "board"})
                    .select(["board", "move"])
                    .to_pandas()
                )
                val_dataset = PlayerTrainDataset(
                    test_df, self.all_moves_dict, self.elo_dict
                )

                if len(val_dataset) > 0:
                    val_loader = DataLoader(
                        val_dataset, batch_size=batch_size, shuffle=False
                    )
                    val_loss, correct, total = 0.0, 0, 0

                    with torch.no_grad():
                        for v_boards, v_targets in val_loader:
                            v_boards = v_boards.to(device)
                            v_targets = v_targets.to(device)

                            v_elos_self = torch.tensor(
                                [virtual_elo_idx] * len(v_boards), device=device
                            )
                            v_elos_oppo = torch.tensor(
                                [virtual_elo_idx] * len(v_boards), device=device
                            )

                            v_logits, _, _ = self.model(
                                v_boards, v_elos_self, v_elos_oppo
                            )

                            v_ce = criterion(v_logits, v_targets)
                            val_loss += v_ce.item()

                            preds = v_logits.argmax(dim=-1)
                            correct += (preds == v_targets).sum().item()
                            total += len(v_targets)

                    epoch_log["val_loss"] = val_loss / len(val_loader)
                    epoch_log["val_acc"] = correct / total if total > 0 else 0.0

                self.model.train()
                self.custom_emb.players_embeddings.weight.requires_grad = True

            history.append(epoch_log)

        train_pbar.close()
        self.model.eval()
        return history

    def predict(
        self, board: chess.Board, player_index: int = 0
    ) -> tuple[dict[str, float], float]:
        """Perform subject-conditioned forward inference over a board position.

        Args:
            board (chess.Board): Target chess board state instance.
            player_index (int, optional): Subject profile identifier for conditional embedding lookup.
                Defaults to 0.

        Returns:
            Tuple[Dict[str, float], float]: Tuple containing normalized move policy distribution
                and scalar position evaluation score.
        """
        device = next(self.model.parameters()).device
        virtual_elo_idx = self.max_maia_idx + 1 + player_index

        fen = board.fen()
        black_flag = False

        # Apply spatial perspective normalization when Black is active player
        if fen.split(" ")[1] == "b":
            proc_board = chess.Board(fen).mirror()
            black_flag = True
        else:
            proc_board = chess.Board(fen)

        board_input = board_to_tensor(proc_board).unsqueeze(0).to(device)
        elos_self = torch.tensor([virtual_elo_idx], device=device)
        elos_oppo = torch.tensor([virtual_elo_idx], device=device)

        # Build legal move mask in tensor representation
        legal_moves = torch.zeros(len(self.all_moves_dict), device=device)
        legal_indices = [
            self.all_moves_dict[m.uci()]
            for m in proc_board.legal_moves
            if m.uci() in self.all_moves_dict
        ]
        if legal_indices:
            legal_moves[torch.tensor(legal_indices, device=device)] = 1.0

        self.model.eval()
        with torch.no_grad():
            logits_maia, _, logits_value = self.model(board_input, elos_self, elos_oppo)
            logits_maia_legal = logits_maia * legal_moves
            probs = logits_maia_legal.softmax(dim=-1).squeeze(0).cpu().tolist()

            val = (logits_value / 2 + 0.5).clamp(0, 1).item()
            if black_flag:
                val = 1 - val

        # Map predictions back to original board orientation and UCI move strings
        all_moves_reversed = {v: k for k, v in self.all_moves_dict.items()}
        raw_moves: dict[str, float] = {}
        for idx in legal_indices:
            move_uci = all_moves_reversed[idx]
            if black_flag:
                move_uci = mirror_move(move_uci)
            raw_moves[move_uci] = round(probs[idx], 4)

        raw_moves = dict(sorted(raw_moves.items(), key=lambda x: x[1], reverse=True))

        legal_uci_moves = {m.uci() for m in board.legal_moves}
        legal_moves_dict: dict[str, float] = {
            move: score for move, score in raw_moves.items() if move in legal_uci_moves
        }

        # Apply optional policy action space pruning
        if self.pruning_fn and legal_moves_dict:
            moves_pruned = self.pruning_fn(legal_moves_dict)
            legal_moves_dict = {
                move: legal_moves_dict[move]
                for move in moves_pruned
                if move in legal_moves_dict
            }

        # Renormalize posterior legal move probability distribution
        total = sum(legal_moves_dict.values())
        if total > 0:
            legal_moves_dict = {
                move: score / total for move, score in legal_moves_dict.items()
            }

        return legal_moves_dict, val
