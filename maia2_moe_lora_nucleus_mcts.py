"""
Cross-validation evaluation pipeline for assessing the Maia-2 architecture enhanced
with Mixture-of-Experts Low-Rank Adaptation (MoE-LoRA), nucleus pruning,
and Monte Carlo Tree Search (MCTS) search optimization.
"""

import logging
from typing import Any

import chess
import pandas as pd
import polars as pl
import tqdm
from maia2 import model

from models import Maia2MoELoRA, MCTSModelWrapper
from utils.data import createPlayerDict, getPlayers, set_seed, split_games_into_folds
from utils.pruning import nucleus_prune

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

SEARCH_ITERATIONS: int = 50


def evaluate_maia2_moe_lora_nucleus_mcts(
    metadata_path: str = "data/metadata.csv",
    positions_path: str = "data/positions.csv",
    max_iterations: int = SEARCH_ITERATIONS,
    timeout: float = 1.0,
    top_p: float = 0.95,
    seed: int = 42,
    device: str = "gpu",
) -> None:
    """Evaluate candidate move prediction accuracy and posterior policy distributions
    of the Maia-2 model augmented with Mixture-of-Experts Low-Rank Adaptation (MoE-LoRA) adapters,
    nucleus pruning, and MCTS search optimization within a k-fold cross-validation framework.

    Args:
        metadata_path (str, optional): File path leading to subject metadata catalog.
            Defaults to "data/metadata.csv".
        positions_path (str, optional): File path leading to input positional observations CSV.
            Defaults to "data/positions.csv".
        max_iterations (int, optional): Maximum simulation budget per search tree node expansion.
            Defaults to SEARCH_ITERATIONS.
        timeout (float, optional): Maximum search execution time budget in seconds per board state.
            Defaults to 1.0.
        top_p (float, optional): Cumulative probability threshold limit for nucleus action pruning.
            Defaults to 0.95.
        seed (int, optional): Global seed value establishing experimental determinism.
            Defaults to 42.
        device (str, optional): Computation target hardware device (e.g., "gpu", "cpu").
            Defaults to "gpu".
    """
    set_seed(seed)

    logger.info("Initializing pre-trained Maia-2 rapid base model architecture...")
    raw_maia_model = model.from_pretrained(type="rapid", device=device)

    logger.info(
        "Loading subject metadata catalog and positional observation dataset..."
    )
    players = getPlayers(metadata_path)
    players_dict: dict[str, int] = createPlayerDict(players)
    positions = pl.read_csv(positions_path)

    # Establish nucleus pruning criteria over policy output distribution
    prune_fn = lambda moves: nucleus_prune(moves, top_p=top_p)

    logger.info(
        "Instantiating Maia-2 MoE-LoRA model with nucleus pruning (top_p=%.2f)...",
        top_p,
    )
    style_moe_model = Maia2MoELoRA(
        model=raw_maia_model, n_players=len(players_dict), pruning_fn=prune_fn
    )

    logger.info(
        "Instantiating MCTS search agent (max_iterations=%d, timeout=%.1fs)...",
        max_iterations,
        timeout,
    )
    mcts_agent = MCTSModelWrapper(
        base_model=style_moe_model, max_iterations=max_iterations, timeout=timeout
    )

    predictions: list[pd.DataFrame] = []
    accuracies: list[dict[str, Any]] = []

    player_pbar = tqdm.tqdm(
        players_dict.items(),
        desc="Evaluating Subject Cohorts",
        unit="subject",
    )

    for player_name, player_index in player_pbar:
        player_positions = positions.filter(pl.col("player_index") == player_index)
        unique_games: list[str] = (
            player_positions.select("game_id").unique()["game_id"].to_list()
        )

        if not unique_games:
            logger.warning(
                "Zero unique games recorded for subject %s (index %d). Skipping evaluation.",
                player_name,
                player_index,
            )
            continue

        n_splits = min(5, len(unique_games))
        game_folds = split_games_into_folds(unique_games, n_splits=n_splits, seed=seed)

        player_preds: list[dict[str, Any]] = []
        correct_count = 0
        total_count = len(player_positions)

        fold_pbar = tqdm.tqdm(
            enumerate(game_folds),
            total=len(game_folds),
            desc=f"  -> {n_splits}-Fold MoE CV [{player_name}]",
            leave=False,
            unit="fold",
        )

        for fold_idx, test_games in fold_pbar:
            train_pos = player_positions.filter(~pl.col("game_id").is_in(test_games))
            test_pos = player_positions.filter(pl.col("game_id").is_in(test_games))

            set_seed(seed + fold_idx)

            # Re-initialize MoE adapter parameters and optimize on training partition
            style_moe_model.reset_adapter()
            style_moe_model.fit_player(player_index, train_pos)

            position_pbar = tqdm.tqdm(
                test_pos.iter_rows(named=True),
                desc="  -> Evaluating test partition",
                leave=False,
                total=len(test_pos),
                unit="pos",
            )

            for row in position_pbar:
                fen: str = row["fen"]
                target_move: str = row["move"]
                board = chess.Board(fen)

                moves_probs, _ = mcts_agent.predict(board)

                # Select mode candidate move corresponding to maximum posterior probability
                predicted_move = (
                    max(moves_probs.items(), key=lambda x: x[1])[0]
                    if moves_probs
                    else ""
                )

                if predicted_move == target_move:
                    correct_count += 1

                player_preds.append(
                    {
                        "player_index": player_index,
                        "fen": fen,
                        "move": target_move,
                        "predicted_move": predicted_move,
                        "moves_probs": str(moves_probs),
                    }
                )

        acc = correct_count / total_count if total_count > 0 else 0.0
        predictions.append(pd.DataFrame(player_preds))
        accuracies.append(
            {
                "player_index": player_index,
                "player_name": player_name,
                "accuracy": acc,
            }
        )

        logger.info(
            "Subject %s (index %d) %d-Fold MoE CV prediction accuracy: %.4f",
            player_name,
            player_index,
            n_splits,
            acc,
        )

    output_predictions_path = (
        f"data/maia2_moe_lora_nucleus_mcts_{max_iterations}_predictions.csv"
    )
    output_accuracies_path = (
        f"data/maia2_moe_lora_nucleus_mcts_{max_iterations}_accuracies.csv"
    )
    general_accuracies_path = "data/maia2_moe_lora_nucleus_mcts_accuracies.csv"

    if predictions:
        predictions_df = pd.concat(predictions, ignore_index=True)
        predictions_df.to_csv(output_predictions_path, index=False)
        logger.info(
            "Model predictions successfully exported to: %s", output_predictions_path
        )

    if accuracies:
        accuracies_df = pl.DataFrame(accuracies)
        accuracies_df.write_csv(output_accuracies_path)
        accuracies_df.write_csv(general_accuracies_path)
        logger.info(
            "Subject accuracy metrics successfully exported to: %s and %s",
            output_accuracies_path,
            general_accuracies_path,
        )


if __name__ == "__main__":
    evaluate_maia2_moe_lora_nucleus_mcts()
