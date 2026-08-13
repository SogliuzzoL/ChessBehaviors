"""
Mixture-of-Experts Low-Rank Adaptation (MoE-LoRA) module for subject-conditioned
policy adaptation on top of the pre-trained Maia-2 architecture.
"""

from collections.abc import Callable
from typing import Any

import chess
import polars as pl
import torch
import torch.nn.functional as F
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


class LoRAExpert(nn.Module):
    """Low-Rank Adaptation (LoRA) expert block for parameter-efficient adapter layers.

    Attributes:
        down (nn.Linear): Down-projection linear bottleneck mapping hidden representation to rank dimension.
        up (nn.Linear): Up-projection linear mapping rank dimension to output logit space.
        scale (float): Constant scaling factor derived from rank hyperparameter.
    """

    def __init__(
        self, v_dim: int, out_dim: int, rank: int = 8, alpha: float = 1.0
    ) -> None:
        """Initialize LoRA expert parameters.

        Args:
            v_dim (int): Input feature representation dimension.
            out_dim (int): Output logit dimension matching policy action space.
            rank (int, optional): Bottleneck rank dimension. Defaults to 8.
            alpha (float, optional): Scaling hyperparameter constant. Defaults to 1.0.
        """
        super().__init__()
        self.down = nn.Linear(v_dim, rank, bias=False)
        self.up = nn.Linear(rank, out_dim, bias=False)
        self.scale: float = alpha / max(1.0, float(rank))

        # Kaiming-style zero-initialization for up-projection weights
        nn.init.zeros_(self.up.weight)
        nn.init.normal_(self.down.weight, std=0.02)

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        """Perform forward low-rank transformation.

        Args:
            v (torch.Tensor): Input hidden activation state tensor.

        Returns:
            torch.Tensor: Scaled output delta logit tensor.
        """
        return self.up(self.down(v)) * self.scale


class PlayerMoEAdapter(nn.Module):
    """Subject-conditioned Mixture-of-Experts (MoE) routing adapter layer.

    Routes subject embedding representations alongside backbone hidden states through a gated router
    network to dynamically combine outputs from multiple parallel LoRA expert modules.

    Attributes:
        v_dim (int): Backbone hidden state dimension.
        out_dim (int): Policy logit output dimension.
        player_emb (nn.Embedding): Trainable subject profile embedding table.
        router (nn.Sequential): Multi-Layer Perceptron gating network outputting softmax expert weights.
        experts (nn.ModuleList): Collection of independent LoRA expert modules.
    """

    def __init__(
        self,
        v_dim: int,
        out_dim: int,
        n_players: int,
        n_experts: int = 8,
        lora_rank: int = 8,
    ) -> None:
        """Initialize PlayerMoEAdapter.

        Args:
            v_dim (int): Backbone representation dimension.
            out_dim (int): Target output logit dimension.
            n_players (int): Total subject profiles in population catalog.
            n_experts (int, optional): Total parallel LoRA expert modules. Defaults to 8.
            lora_rank (int, optional): Inner rank of individual LoRA experts. Defaults to 8.
        """
        super().__init__()
        self.v_dim: int = v_dim
        self.out_dim: int = out_dim
        self.player_emb = nn.Embedding(n_players, 32)

        self.router = nn.Sequential(
            nn.Linear(v_dim + 32, 128),
            nn.ReLU(),
            nn.Linear(128, n_experts),
        )
        self.experts = nn.ModuleList(
            [LoRAExpert(v_dim, out_dim, rank=lora_rank) for _ in range(n_experts)]
        )

    def forward(self, v: torch.Tensor, player_ids: torch.Tensor) -> torch.Tensor:
        """Compute gated linear combination of expert outputs conditioned on subject identities.

        Args:
            v (torch.Tensor): Backbone hidden representation tensor.
            player_ids (torch.Tensor): Subject identifier index tensor.

        Returns:
            torch.Tensor: Aggregated policy logit delta tensor.
        """
        p_emb = self.player_emb(player_ids)
        router_in = torch.cat([v, p_emb], dim=-1)
        g = F.softmax(self.router(router_in), dim=-1)

        expert_outs = [expert(v) for expert in self.experts]
        deltas = torch.stack(expert_outs, dim=1)

        return (g.unsqueeze(-1) * deltas).sum(dim=1)


class Maia2MoELoRA(ChessModel):
    """Maia-2 model architecture augmented with subject-conditioned MoE-LoRA adapters.

    Integrates parameter-efficient adapters hooked into intermediate layer normalization states
    to specialize policy predictions per human subject cohort without altering base weights.

    Attributes:
        model: Underlying pre-trained Maia-2 neural execution model.
        prepared: Pre-allocated inference context structures.
        pruning_fn (Optional[Callable[[Dict[str, float]], List[str]]]): Optional policy space pruning function.
        all_moves_dict (Dict[str, int]): Vocabulary mapping UCI move strings to integer indices.
        elo_dict (Dict[Any, Any]): Subject rating metadata dictionary.
        max_maia_idx (int): Baseline reference rating embedding index.
        extracted_hidden_v (Optional[torch.Tensor]): Intermediate activation tensor captured via forward hook.
        adapter (PlayerMoEAdapter): Instantiated trainable MoE-LoRA adapter layer.
    """

    def __init__(
        self,
        model: object,
        n_players: int,
        pruning_fn: Callable[[dict[str, float]], list[str]] | None = None,
        n_experts: int = 8,
        lora_rank: int = 8,
    ) -> None:
        """Initialize the Maia2MoELoRA architecture wrapper.

        Args:
            model (object): Pre-trained Maia-2 model architecture instance.
            n_players (int): Total subject profiles in population catalog.
            pruning_fn (Optional[Callable[[Dict[str, float]], List[str]]], optional): Action space pruning
                transformation. Defaults to None.
            n_experts (int, optional): Parallel LoRA expert count. Defaults to 8.
            lora_rank (int, optional): Inner rank parameter for LoRA blocks. Defaults to 8.

        Raises:
            AttributeError: If target hook layer ('last_ln') is missing from the base architecture.
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

        net_model = getattr(self.model, "net", self.model)
        self.extracted_hidden_v: torch.Tensor | None = None

        def hook_fn(
            module: nn.Module, input_tensor: Any, output_tensor: torch.Tensor
        ) -> None:
            self.extracted_hidden_v = output_tensor

        if hasattr(net_model, "last_ln"):
            net_model.last_ln.register_forward_hook(hook_fn)
        else:
            raise AttributeError(
                "Unable to locate target normalization layer 'last_ln' in Maia-2 network topology."
            )

        device = next(self.model.parameters()).device

        # Infer intermediate hidden dimension using a zero-gradient dry run
        with torch.no_grad():
            dummy_b = board_to_tensor(chess.Board()).unsqueeze(0).to(device)
            ref_elo = torch.tensor([self.max_maia_idx], device=device)
            logits, _, _ = self.model(dummy_b, ref_elo, ref_elo)
            hidden_v = self.extracted_hidden_v

        self.adapter = PlayerMoEAdapter(
            v_dim=hidden_v.size(-1),
            out_dim=logits.size(-1),
            n_players=n_players,
            n_experts=n_experts,
            lora_rank=lora_rank,
        ).to(device)

    def reset_adapter(self) -> None:
        """Reset adapter weights to default normal distributions."""
        for m in self.adapter.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def fit_player(
        self,
        player_index: int,
        train_pos: pl.DataFrame,
        test_pos: pl.DataFrame | None = None,
        epochs: int = 20,
        batch_size: int = 64,
        lr: float = 1e-2,
    ) -> list[dict[str, Any]]:
        """Optimize MoE-LoRA adapter parameters on subject observational positions.

        Args:
            player_index (int): Target subject profile identifier.
            train_pos (pl.DataFrame): Dataframe containing training board positions.
            test_pos (Optional[pl.DataFrame], optional): Optional validation board positions. Defaults to None.
            epochs (int, optional): Optimization epoch count. Defaults to 20.
            batch_size (int, optional): Mini-batch processing size. Defaults to 64.
            lr (float, optional): Initial peak learning rate. Defaults to 1e-2.

        Returns:
            List[Dict[str, Any]]: Metrics history log containing batch and epoch-level evaluation records.
        """
        if len(train_pos) == 0:
            return []

        train_df = (
            train_pos.rename({"fen": "board"}).select(["board", "move"]).to_pandas()
        )
        dataset = PlayerTrainDataset(train_df, self.all_moves_dict, self.elo_dict)
        if len(dataset) == 0:
            return []

        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        self.adapter.train()
        for param in self.adapter.parameters():
            param.requires_grad = True

        optimizer = optim.AdamW(self.adapter.parameters(), lr=lr, weight_decay=1e-2)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs * len(dataloader), eta_min=1e-5
        )
        criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
        device = next(self.model.parameters()).device

        history: list[dict[str, Any]] = []
        global_step = 0

        total_batches = epochs * len(dataloader)
        train_pbar = tqdm.tqdm(
            total=total_batches,
            desc=f"Optimizing MoE Adapter [Subject {player_index}]",
            leave=False,
            unit="batch",
        )

        for epoch in range(epochs):
            running_loss = 0.0
            steps = 0

            for boards, targets in dataloader:
                boards, targets = boards.to(device), targets.to(device)
                ref_elos = torch.tensor(
                    [self.max_maia_idx] * len(boards), device=device
                )
                player_ids = torch.tensor([player_index] * len(boards), device=device)

                optimizer.zero_grad()

                with torch.no_grad():
                    logits_maia, _, _ = self.model(boards, ref_elos, ref_elos)
                    hidden_v = self.extracted_hidden_v

                delta_logits = self.adapter(hidden_v, player_ids)
                final_logits = logits_maia + delta_logits

                loss = criterion(final_logits, targets)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.adapter.parameters(), max_norm=1.0)
                optimizer.step()

                current_lr = scheduler.get_last_lr()[0]
                scheduler.step()

                loss_val = loss.item()
                running_loss += loss_val
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
                        "lr": current_lr,
                    }
                )

                train_pbar.set_postfix(
                    {
                        "epoch": f"{epoch + 1}/{epochs}",
                        "loss": f"{loss_val:.4f}",
                        "avg": f"{running_loss / steps:.4f}",
                    }
                )
                train_pbar.update(1)

            epoch_log: dict[str, Any] = {
                "type": "epoch",
                "player_index": player_index,
                "epoch": epoch + 1,
                "avg_train_loss": running_loss / steps,
                "lr": current_lr,
            }

            # Evaluate intermediate validation performance
            if test_pos is not None and len(test_pos) > 0:
                self.adapter.eval()
                val_loss, correct, total = 0.0, 0, 0
                test_df = (
                    test_pos.rename({"fen": "board"})
                    .select(["board", "move"])
                    .to_pandas()
                )
                val_dataset = PlayerTrainDataset(
                    test_df, self.all_moves_dict, self.elo_dict
                )
                val_loader = DataLoader(
                    val_dataset, batch_size=batch_size, shuffle=False
                )

                with torch.no_grad():
                    for v_boards, v_targets in val_loader:
                        v_boards, v_targets = v_boards.to(device), v_targets.to(device)
                        v_elos = torch.tensor(
                            [self.max_maia_idx] * len(v_boards), device=device
                        )
                        v_pids = torch.tensor(
                            [player_index] * len(v_boards), device=device
                        )

                        v_maia, _, _ = self.model(v_boards, v_elos, v_elos)
                        v_hidden = self.extracted_hidden_v
                        v_final = v_maia + self.adapter(v_hidden, v_pids)

                        val_loss += criterion(v_final, v_targets).item()
                        preds = v_final.argmax(dim=-1)
                        correct += (preds == v_targets).sum().item()
                        total += len(v_targets)

                epoch_log["val_loss"] = val_loss / len(val_loader)
                epoch_log["val_acc"] = correct / total if total > 0 else 0.0
                self.adapter.train()

            history.append(epoch_log)

        train_pbar.close()
        self.adapter.eval()
        return history

    def predict(
        self, board: chess.Board, player_index: int = 0
    ) -> tuple[dict[str, float], float]:
        """Perform forward inference with MoE-LoRA adapter policy corrections.

        Args:
            board (chess.Board): Target chess board position.
            player_index (int, optional): Subject profile identifier. Defaults to 0.

        Returns:
            Tuple[Dict[str, float], float]: Tuple containing normalized move policy distribution
                and scalar board value evaluation.
        """
        device = next(self.model.parameters()).device

        fen = board.fen()
        black_flag = False

        # Apply perspective transformation if active side is Black
        if fen.split(" ")[1] == "b":
            proc_board = chess.Board(fen).mirror()
            black_flag = True
        else:
            proc_board = chess.Board(fen)

        board_input = board_to_tensor(proc_board).unsqueeze(0).to(device)
        ref_elo = torch.tensor([self.max_maia_idx], device=device)
        player_ids = torch.tensor([player_index], device=device)

        # Build legal action mask over full vocabulary
        legal_moves = torch.zeros(len(self.all_moves_dict), device=device)
        legal_indices = [
            self.all_moves_dict[m.uci()]
            for m in proc_board.legal_moves
            if m.uci() in self.all_moves_dict
        ]
        if legal_indices:
            legal_moves[torch.tensor(legal_indices, device=device)] = 1.0

        self.model.eval()
        self.adapter.eval()
        with torch.no_grad():
            logits_maia, _, logits_value = self.model(board_input, ref_elo, ref_elo)
            hidden_v = self.extracted_hidden_v
            delta_logits = self.adapter(hidden_v, player_ids)
            final_logits = logits_maia + delta_logits

            logits_legal = final_logits * legal_moves
            probs = logits_legal.softmax(dim=-1).squeeze(0).cpu().tolist()

            val = (logits_value / 2 + 0.5).clamp(0, 1).item()
            if black_flag:
                val = 1 - val

        # Revert legal moves to original perspective and string format
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

        # Apply optional action space pruning
        if self.pruning_fn and legal_moves_dict:
            moves_pruned = self.pruning_fn(legal_moves_dict)
            legal_moves_dict = {
                move: legal_moves_dict[move]
                for move in moves_pruned
                if move in legal_moves_dict
            }

        # Renormalize posterior legal move distribution
        total = sum(legal_moves_dict.values())
        if total > 0:
            legal_moves_dict = {
                move: score / total for move, score in legal_moves_dict.items()
            }

        return legal_moves_dict, val
