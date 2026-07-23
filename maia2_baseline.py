import logging

import chess
import pandas as pd
import polars as pl
import tqdm
from maia2 import model

from models.model import Maia2
from utils.data import createPlayerDict, getPlayers

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    raw_maia_model = model.from_pretrained(type="rapid", device="gpu")
    maia2_model = Maia2(model=raw_maia_model, pruning_fn=None)

    players = getPlayers("data/metadata.csv")
    players_dict = createPlayerDict(players)

    positions = pl.read_csv("data/positions.csv")

    predictions = []
    accuracies = []

    for player_name, player_index in tqdm.tqdm(players_dict.items()):
        player_positions = positions.filter(pl.col("player_index") == player_index)

        correct_count = 0
        total_count = len(player_positions)
        player_preds = []

        for row in tqdm.tqdm(player_positions.iter_rows(named=True), total=total_count):
            fen = row["fen"]
            target_move = row["move"]
            board = chess.Board(fen)

            moves_probs, _ = maia2_model.predict(board)

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

        logger.info(f"Player {player_name} (index {player_index}) accuracy: {acc}")

    predictions_df = pd.concat(predictions, ignore_index=True)
    predictions_df.to_csv("data/maia2_predictions.csv", index=False)

    accuracies_df = pl.DataFrame(accuracies)
    accuracies_df.write_csv("data/maia2_accuracies.csv")
