"""
Baseline evaluation pipeline for predicting subject decision-making accuracy
and move probability distributions using the pre-trained Maia-2 neural architecture.
"""

import logging
from typing import Any

import chess
import pandas as pd
import polars as pl
import tqdm
from maia2 import model

from models import Maia2
from utils.data import createPlayerDict, getPlayers

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def evaluate_maia2_baseline(
    metadata_path: str = "data/metadata.csv",
    positions_path: str = "data/positions.csv",
    output_predictions_path: str = "data/maia2_predictions.csv",
    output_accuracies_path: str = "data/maia2_accuracies.csv",
    device: str = "gpu",
) -> None:
    """Evaluate candidate move prediction accuracy and output probability distributions
    using the un-tuned Maia-2 baseline neural model across human subject cohorts.

    Args:
        metadata_path (str, optional): File path leading to subject metadata catalog.
            Defaults to "data/metadata.csv".
        positions_path (str, optional): File path leading to input positional observations CSV.
            Defaults to "data/positions.csv".
        output_predictions_path (str, optional): Destination path for exported prediction records.
            Defaults to "data/maia2_predictions.csv".
        output_accuracies_path (str, optional): Destination path for aggregated subject accuracy metrics.
            Defaults to "data/maia2_accuracies.csv".
        device (str, optional): Computation target hardware device (e.g., "gpu", "cpu").
            Defaults to "gpu".
    """
    logger.info("Initializing pre-trained Maia-2 rapid baseline architecture...")
    raw_maia_model = model.from_pretrained(type="rapid", device=device)
    maia2_model = Maia2(model=raw_maia_model, pruning_fn=None)

    logger.info("Loading evaluation dataset and subject cohort metadata...")
    players = getPlayers(metadata_path)
    players_dict: dict[str, int] = createPlayerDict(players)
    positions = pl.read_csv(positions_path)

    predictions: list[pd.DataFrame] = []
    accuracies: list[dict[str, Any]] = []

    player_pbar = tqdm.tqdm(
        players_dict.items(),
        desc="Evaluating Subject Cohorts",
        unit="subject",
    )

    for player_name, player_index in player_pbar:
        player_positions = positions.filter(pl.col("player_index") == player_index)
        total_count = len(player_positions)

        if total_count == 0:
            logger.warning(
                "Zero positional observations recorded for subject %s (index %d).",
                player_name,
                player_index,
            )
            accuracies.append(
                {
                    "player_index": player_index,
                    "player_name": player_name,
                    "accuracy": 0.0,
                }
            )
            continue

        correct_count = 0
        player_preds: list[dict[str, Any]] = []

        position_pbar = tqdm.tqdm(
            player_positions.iter_rows(named=True),
            total=total_count,
            desc=f"  -> Processing positions [{player_name}]",
            leave=False,
            unit="pos",
        )

        for row in position_pbar:
            fen: str = row["fen"]
            target_move: str = row["move"]
            board = chess.Board(fen)

            moves_probs, _ = maia2_model.predict(board)

            # Determine mode candidate move corresponding to highest predicted probability
            predicted_move = (
                max(moves_probs.items(), key=lambda x: x[1])[0] if moves_probs else ""
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
            "Subject %s (index %d) prediction accuracy: %.4f",
            player_name,
            player_index,
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
    evaluate_maia2_baseline()
