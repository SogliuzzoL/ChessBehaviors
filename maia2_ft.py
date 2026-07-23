import logging
import random

import chess
import pandas as pd
import polars as pl
import tqdm
from maia2 import model

from models.model import Maia2FT
from utils.data import createPlayerDict, getPlayers

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def split_games_into_folds(game_ids: list, n_splits: int = 5) -> list[list]:
    shuffled_games = game_ids.copy()
    random.seed(42)
    random.shuffle(shuffled_games)

    folds = [[] for _ in range(n_splits)]
    for idx, game_id in enumerate(shuffled_games):
        folds[idx % n_splits].append(game_id)
    return folds


if __name__ == "__main__":
    raw_maia_model = model.from_pretrained(type="rapid", device="gpu")

    players = getPlayers("data/metadata.csv")
    players_dict = createPlayerDict(players)
    positions = pl.read_csv("data/positions.csv")

    maia2_ft_model = Maia2FT(model=raw_maia_model, n_players=len(players_dict))

    predictions = []
    accuracies = []

    for player_name, player_index in tqdm.tqdm(players_dict.items(), desc="Joueurs"):
        player_positions = positions.filter(pl.col("player_index") == player_index)

        unique_games = player_positions.select("game_id").unique()["game_id"].to_list()

        if not unique_games:
            continue

        n_splits = min(5, len(unique_games))
        game_folds = split_games_into_folds(unique_games, n_splits=n_splits)

        player_preds = []
        correct_count = 0
        total_count = len(player_positions)

        for fold_idx, test_games in enumerate(
            tqdm.tqdm(game_folds, desc=f"5-Fold CV {player_name}", leave=False)
        ):
            train_pos = player_positions.filter(~pl.col("game_id").is_in(test_games))
            test_pos = player_positions.filter(pl.col("game_id").is_in(test_games))

            maia2_ft_model.reset_player_embedding(player_index)
            maia2_ft_model.fit_player(player_index, train_pos, epochs=3, lr=1e-3)

            for row in test_pos.iter_rows(named=True):
                fen = row["fen"]
                target_move = row["move"]
                board = chess.Board(fen)

                moves_probs, _ = maia2_ft_model.predict(
                    board, player_index=player_index
                )

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
            f"Player {player_name} (index {player_index}) 5-Fold CV Accuracy: {acc:.4f}"
        )

    predictions_df = pd.concat(predictions, ignore_index=True)
    predictions_df.to_csv("data/maia2_ft_predictions.csv", index=False)

    accuracies_df = pl.DataFrame(accuracies)
    accuracies_df.write_csv("data/maia2_ft_accuracies.csv")
