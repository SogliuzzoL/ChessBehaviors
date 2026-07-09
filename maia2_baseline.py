import logging

import polars as pl
import pandas as pd
import tqdm
from maia2 import inference, model

from utils.data import createPlayerDict, getPlayers

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    maia2_model = model.from_pretrained(type="rapid", device="gpu")

    players = getPlayers("data/metadata.csv")
    players_dict = createPlayerDict(players)

    positions = pl.read_csv("data/positions.csv")

    predictions = []
    accuracies = []

    for player_name, player_index in tqdm.tqdm(players_dict.items()):
        player_positions = positions.filter(pl.col("player_index") == player_index)

        maia2_input = (
            player_positions.rename({"fen": "board"})
            .with_columns(
                [pl.lit(2500).alias("active_elo"), pl.lit(2500).alias("opponent_elo")]
            )
            .select(["board", "move", "active_elo", "opponent_elo"])
        )

        maia2_input = maia2_input.to_pandas()

        maia2_output, acc = inference.inference_batch(
            maia2_input, maia2_model, verbose=1, batch_size=1024, num_workers=4
        )

        maia2_output.insert(0, "player_index", player_index)
        predictions.append(maia2_output)


        accuracies.append(
            {"player_index": player_index, "player_name": player_name, "accuracy": acc}
        )

        logger.info(f"Player {player_name} (index {player_index}) accuracy: {acc}")

    predictions = pd.concat(predictions)
    predictions.to_csv("data/maia2_predictions.csv", index=False)

    accuracies = pl.DataFrame(accuracies)
    accuracies.write_csv("data/maia2_accuracies.csv")
