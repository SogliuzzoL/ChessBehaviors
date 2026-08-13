"""
Baseline empirical prediction pipeline for calculating subject move accuracy
and extracting marginal move distributions across board states (FENs).
"""

import logging
from collections import Counter
from typing import Any

import pandas as pd
import polars as pl
import tqdm

from utils.data import createPlayerDict, getPlayers

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def evaluate_ground_truth_baseline(
    metadata_path: str = "data/metadata.csv",
    positions_path: str = "data/positions.csv",
    output_predictions_path: str = "data/ground_truth_predictions.csv",
    output_accuracies_path: str = "data/ground_truth_accuracies.csv",
) -> None:
    """Compute empirical mode move accuracy and empirical marginal move distributions per subject profile.

    Evaluates subject decision accuracy using a deterministic mode-move predictor derived
    from the empirical majority move per board state (FEN). Generates empirical target
    action distributions and exports prediction records alongside aggregated accuracy metrics.

    Args:
        metadata_path (str, optional): File path to subject metadata catalog. Defaults to "data/metadata.csv".
        positions_path (str, optional): File path to input position dataset CSV. Defaults to "data/positions.csv".
        output_predictions_path (str, optional): Destination path for exported ground-truth prediction records.
            Defaults to "data/ground_truth_predictions.csv".
        output_accuracies_path (str, optional): Destination path for exported subject accuracy metrics.
            Defaults to "data/ground_truth_accuracies.csv".
    """
    logger.info("Initiating empirical baseline evaluation pipeline.")

    players = getPlayers(metadata_path)
    players_dict: dict[str, int] = createPlayerDict(players)

    positions_pl = pl.read_csv(positions_path)

    predictions: list[pd.DataFrame] = []
    accuracies: list[dict[str, Any]] = []

    player_pbar = tqdm.tqdm(
        players_dict.items(),
        desc="Evaluating Subject Cohorts",
        unit="subject",
    )

    for player_name, player_index in player_pbar:
        # Filter positional observations pertaining to target subject cohort
        player_df = positions_pl.filter(
            pl.col("player_index") == player_index
        ).to_pandas()

        total_count = len(player_df)
        if total_count == 0:
            logger.warning(
                "Zero observations recorded for subject %s (index %d).",
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

        # Group observed actions by distinct board positions (FENs)
        fen_groups: dict[str, list[str]] = (
            player_df.groupby("fen")["move"].apply(list).to_dict()
        )

        fen_marginal_probs: dict[str, str] = {}
        fen_top_move: dict[str, str] = {}

        # Construct empirical probability distributions and extract majority moves per position
        for fen, moves in fen_groups.items():
            counts = Counter(moves)
            total_fen_moves = len(moves)

            probs = {
                move: round(cnt / total_fen_moves, 6) for move, cnt in counts.items()
            }
            fen_marginal_probs[fen] = str(probs)
            fen_top_move[fen] = counts.most_common(1)[0][0]

        player_preds: list[dict[str, Any]] = []
        correct_count = 0

        # Evaluate target subject observations against empirical majority move predictions
        for row in player_df.itertuples(index=False):
            fen = row.fen
            target_move = row.move

            predicted_move = fen_top_move[fen]
            moves_probs = fen_marginal_probs[fen]

            if predicted_move == target_move:
                correct_count += 1

            player_preds.append(
                {
                    "player_index": player_index,
                    "fen": fen,
                    "move": target_move,
                    "predicted_move": predicted_move,
                    "moves_probs": moves_probs,
                }
            )

        acc = correct_count / total_count
        predictions.append(pd.DataFrame(player_preds))
        accuracies.append(
            {
                "player_index": player_index,
                "player_name": player_name,
                "accuracy": acc,
            }
        )

        logger.info(
            "Subject %s (index %d) empirical accuracy: %.4f",
            player_name,
            player_index,
            acc,
        )

    if predictions:
        predictions_df = pd.concat(predictions, ignore_index=True)
        predictions_df.to_csv(output_predictions_path, index=False)
        logger.info(
            "Empirical prediction records successfully saved to: %s",
            output_predictions_path,
        )

    if accuracies:
        accuracies_df = pl.DataFrame(accuracies)
        accuracies_df.write_csv(output_accuracies_path)
        logger.info(
            "Subject cohort accuracy metrics successfully saved to: %s",
            output_accuracies_path,
        )


if __name__ == "__main__":
    evaluate_ground_truth_baseline()
