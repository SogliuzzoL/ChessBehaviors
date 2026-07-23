import logging

import pandas as pd
import polars as pl

from utils.data import createPlayerDict, getPlayers

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    players = getPlayers("data/metadata.csv")
    players_dict = createPlayerDict(players)
    positions = pl.read_csv("data/positions.csv")

    logger.info("Computing marginal distributions...")

    move_counts = positions.group_by(["player_index", "fen", "move"]).agg(
        pl.len().alias("count")
    )

    total_counts = move_counts.group_by(["player_index", "fen"]).agg(
        pl.sum("count").alias("total")
    )

    distributions = move_counts.join(
        total_counts, on=["player_index", "fen"]
    ).with_columns((pl.col("count") / pl.col("total")).alias("prob"))

    top_moves = (
        distributions.sort("prob", descending=True)
        .group_by(["player_index", "fen"])
        .first()
        .select(["player_index", "fen", pl.col("move").alias("predicted_move")])
    )

    dist_dict = distributions.group_by(["player_index", "fen"]).agg(
        pl.struct(["move", "prob"])
        .map_elements(
            lambda x: str({item["move"]: round(item["prob"], 6) for item in x}),
            return_dtype=pl.String,
        )
        .alias("moves_probs")
    )

    logger.info("Aligning distributions over all positions...")
    final_positions = (
        positions.join(top_moves, on=["player_index", "fen"], how="left")
        .join(dist_dict, on=["player_index", "fen"], how="left")
        .select(["player_index", "fen", "move", "predicted_move", "moves_probs"])
    )

    logger.info("Computing marginal accuracies by player...")
    predictions_df = final_positions.to_pandas()
    accuracies = []

    for player_name, player_index in players_dict.items():
        player_preds = predictions_df[predictions_df["player_index"] == player_index]
        total_count = len(player_preds)

        if total_count > 0:
            correct_count = (
                player_preds["predicted_move"] == player_preds["move"]
            ).sum()
            acc = correct_count / total_count
        else:
            acc = 0.0

        accuracies.append(
            {
                "player_index": player_index,
                "player_name": player_name,
                "accuracy": acc,
            }
        )
        logger.info(f"Player {player_name} (index {player_index}) accuracy: {acc:.4f}")

    predictions_df.to_csv("data/ground_truth_predictions.csv", index=False)
    accuracies_df = pl.DataFrame(accuracies)
    accuracies_df.write_csv("data/ground_truth_accuracies.csv")
