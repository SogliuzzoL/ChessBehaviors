import logging
from collections import Counter

import pandas as pd
import polars as pl
import tqdm

from utils.data import createPlayerDict, getPlayers

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    players = getPlayers("data/metadata.csv")
    players_dict = createPlayerDict(players)

    positions_pl = pl.read_csv("data/positions.csv")

    predictions = []
    accuracies = []

    for player_name, player_index in tqdm.tqdm(players_dict.items()):
        player_df = (
            positions_pl.filter(pl.col("player_index") == player_index)
            .to_pandas()
        )

        total_count = len(player_df)
        if total_count == 0:
            accuracies.append(
                {
                    "player_index": player_index,
                    "player_name": player_name,
                    "accuracy": 0.0,
                }
            )
            continue

        fen_groups = player_df.groupby("fen")["move"].apply(list).to_dict()

        fen_marginal_probs = {}
        fen_top_move = {}

        for fen, moves in fen_groups.items():
            counts = Counter(moves)
            total_fen_moves = len(moves)

            probs = {
                move: round(cnt / total_fen_moves, 6)
                for move, cnt in counts.items()
            }
            fen_marginal_probs[fen] = str(probs)

            fen_top_move[fen] = counts.most_common(1)[0][0]

        player_preds = []
        correct_count = 0

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

        logger.info(f"Player {player_name} (index {player_index}) accuracy: {acc:.4f}")

    predictions_df = pd.concat(predictions, ignore_index=True)
    predictions_df.to_csv("data/ground_truth_predictions.csv", index=False)

    accuracies_df = pl.DataFrame(accuracies)
    accuracies_df.write_csv("data/ground_truth_accuracies.csv")
