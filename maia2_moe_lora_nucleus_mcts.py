import logging

import chess
import pandas as pd
import polars as pl
import tqdm
from maia2 import model

from models import Maia2MoELoRA, MCTSModelWrapper
from utils.data import createPlayerDict, getPlayers, set_seed, split_games_into_folds
from utils.pruning import nucleus_prune

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

N = 50

if __name__ == "__main__":
    SEED = 42
    set_seed(SEED)

    raw_maia_model = model.from_pretrained(type="rapid", device="gpu")
    players = getPlayers("data/metadata.csv")
    players_dict = createPlayerDict(players)
    positions = pl.read_csv("data/positions.csv")

    prune_fn = lambda moves: nucleus_prune(moves, top_p=0.95)
    style_moe_model = Maia2MoELoRA(
        model=raw_maia_model, n_players=len(players_dict), pruning_fn=prune_fn
    )
    mcts_agent = MCTSModelWrapper(
        base_model=style_moe_model, max_iterations=N, timeout=1.0
    )

    predictions = []
    accuracies = []

    for player_name, player_index in tqdm.tqdm(players_dict.items(), desc="Players"):
        player_positions = positions.filter(pl.col("player_index") == player_index)
        unique_games = player_positions.select("game_id").unique()["game_id"].to_list()
        if not unique_games:
            continue

        n_splits = min(5, len(unique_games))
        game_folds = split_games_into_folds(unique_games, n_splits=n_splits, seed=SEED)

        player_preds = []
        correct_count = 0
        total_count = len(player_positions)

        for fold_idx, test_games in enumerate(
            tqdm.tqdm(game_folds, desc=f"5-Fold MoE {player_name}", leave=False)
        ):
            train_pos = player_positions.filter(~pl.col("game_id").is_in(test_games))
            test_pos = player_positions.filter(pl.col("game_id").is_in(test_games))

            set_seed(SEED + fold_idx)
            style_moe_model.reset_adapter()
            style_moe_model.fit_player(player_index, train_pos)

            for row in tqdm.tqdm(
                test_pos.iter_rows(named=True),
                desc="Testing",
                leave=False,
                total=len(test_pos),
            ):
                fen = row["fen"]
                target_move = row["move"]
                board = chess.Board(fen)

                moves_probs, _ = mcts_agent.predict(board)
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
            f"Player {player_name} (index {player_index}) 5-Fold MoE Accuracy: {acc:.4f}"
        )

    predictions_df = pd.concat(predictions, ignore_index=True)
    predictions_df.to_csv(
        f"data/maia2_moe_lora_nucleus_mcts_{N}_predictions.csv", index=False
    )

    accuracies_df = pl.DataFrame(accuracies)
    accuracies_df.write_csv(f"data/maia2_moe_lora_nucleus_mcts_{N}_accuracies.csv")
    accuracies_df.write_csv("data/maia2_moe_lora_nucleus_mcts_accuracies.csv")
