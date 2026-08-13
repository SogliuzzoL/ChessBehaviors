"""
Evaluation pipeline for evaluating move prediction accuracy and policy distributions
using the Maia-2 architecture enhanced with gradient descent optimization (DescentModelWrapper).
"""

import logging
from typing import Any

import chess
import pandas as pd
import polars as pl
import tqdm
from maia2 import model

from models import DescentModelWrapper, Maia2
from utils.data import createPlayerDict, getPlayers

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

ITERATION_LIMIT: int = 50


def evaluate_maia2_descent(
    metadata_path: str = "data/metadata.csv",
    positions_path: str = "data/positions.csv",
    max_iterations: int = ITERATION_LIMIT,
    timeout: float = 1.0,
    device: str = "gpu",
) -> None:
    """Evaluate decision-making accuracy and policy distributions of the Maia-2 model
    augmented with iterative gradient descent optimization across human subject cohorts.

    Args:
        metadata_path (str, optional): File path leading to subject metadata catalog.
            Defaults to "data/metadata.csv".
        positions_path (str, optional): File path leading to input positional observations CSV.
            Defaults to "data/positions.csv".
        max_iterations (int, optional): Maximum optimization iterations for the descent wrapper.
            Defaults to ITERATION_LIMIT.
        timeout (float, optional): Maximum search execution time in seconds per board state.
            Defaults to 1.0.
        device (str, optional): Computation target hardware device (e.g., "gpu", "cpu").
            Defaults to "gpu".
    """
    logger.info("Initializing pre-trained Maia-2 rapid base architecture...")
    raw_maia_model = model.from_pretrained(type="rapid", device=device)
    maia2_model = Maia2(model=raw_maia_model, pruning_fn=None)

    logger.info(
        "Instantiating descent optimization agent (max_iterations=%d, timeout=%.1fs)...",
        max_iterations,
        timeout,
    )
    descent_agent = DescentModelWrapper(
        base_model=maia2_model, max_iterations=max_iterations, timeout=timeout
    )

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

            moves_probs, _ = descent_agent.predict(board)

            # Extract mode candidate move corresponding to maximum posterior probability
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

    output_predictions_path = f"data/maia2_descent_{max_iterations}_predictions.csv"
    output_accuracies_path = f"data/maia2_descent_{max_iterations}_accuracies.csv"

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
    evaluate_maia2_descent()
