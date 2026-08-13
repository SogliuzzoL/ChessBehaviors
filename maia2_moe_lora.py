"""
Cross-validation evaluation pipeline for subject-level conditioning of the Maia-2 architecture
using Mixture-of-Experts Low-Rank Adaptation (MoE-LoRA) adapters, evaluating move prediction
accuracy and policy distributions across stratified game partitions.
"""

import logging
from typing import Any

import chess
import pandas as pd
import polars as pl
import tqdm
from maia2 import model

from models import Maia2MoELoRA
from utils.data import createPlayerDict, getPlayers, set_seed, split_games_into_folds

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def evaluate_maia2_moe_lora(
    metadata_path: str = "data/metadata.csv",
    positions_path: str = "data/positions.csv",
    output_predictions_path: str = "data/maia2_moe_lora_predictions.csv",
    output_accuracies_path: str = "data/maia2_moe_lora_accuracies.csv",
    seed: int = 42,
    device: str = "gpu",
) -> None:
    """Evaluate candidate move prediction accuracy and posterior policy distributions
    of the Maia-2 model augmented with Mixture-of-Experts Low-Rank Adaptation (MoE-LoRA)
    adapters within a k-fold cross-validation framework.

    Args:
        metadata_path (str, optional): File path leading to subject metadata catalog.
            Defaults to "data/metadata.csv".
        positions_path (str, optional): File path leading to input positional observations CSV.
            Defaults to "data/positions.csv".
        output_predictions_path (str, optional): Destination path for exported prediction records.
            Defaults to "data/maia2_moe_lora_predictions.csv".
        output_accuracies_path (str, optional): Destination path for aggregated subject accuracy metrics.
            Defaults to "data/maia2_moe_lora_accuracies.csv".
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

    logger.info(
        "Instantiating Maia-2 MoE-LoRA model wrapper for %d subject cohorts...",
        len(players_dict),
    )
    style_moe_model = Maia2MoELoRA(model=raw_maia_model, n_players=len(players_dict))

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
        all_train_logs: list[dict[str, Any]] = []

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
            logs = style_moe_model.fit_player(
                player_index, train_pos, test_pos=test_pos
            )

            for entry in logs:
                entry["fold"] = fold_idx
            all_train_logs.extend(logs)

            # Persist intermediate convergence logs per subject cohort
            logs_df = pd.DataFrame(all_train_logs)
            logs_df.to_csv(f"data/training_logs_player_{player_index}.csv", index=False)

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

                moves_probs, _ = style_moe_model.predict(
                    board, player_index=player_index
                )

                # Determine mode candidate move corresponding to highest posterior probability
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

    if predictions:
        predictions_df = pd.concat(predictions, ignore_index=True)
        predictions_df.to_csv(output_predictions_path, index=False)
        logger.info(
            "Model predictions successfully exported to: %s", output_predictions_path
        )

    if accuracies:
        accuracies_df = pl.DataFrame(accuracies)
        accuracies_df.write_csv(output_accuracies_path)
        logger.info(
            "Subject accuracy metrics successfully exported to: %s",
            output_accuracies_path,
        )


if __name__ == "__main__":
    evaluate_maia2_moe_lora()
