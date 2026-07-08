import logging
import os

import polars as pl
import tqdm

from utils.data import (
    createPlayerDict,
    filterMetadataByPlayer,
    flattenData,
    getPGNmetadata,
    getPlayers,
)

logger = logging.getLogger(__name__)


def buildMetadataFile():
    """
    Builds a metadata file from PGN files in the data/raw directory.
    """
    logger.info("Building metadata file")

    metadata = []
    for folder in tqdm.tqdm(os.listdir("data/raw")):
        for file in tqdm.tqdm(os.listdir(f"data/raw/{folder}")):
            result = getPGNmetadata(
                file_path=f"data/raw/{folder}/{file}",
                user_id=folder,
                game_id=file.replace(".pgn", ""),
            )
            if result:
                metadata.append(result)
            else:
                logger.info(f"Failed to parse {folder}/{file}")

    df = pl.DataFrame(metadata)
    print(df)
    df.write_csv("data/metadata.csv")

    logger.info("Metadata file built successfully")


def buildPositionFile():
    logger.info("Building position file")

    metadata = pl.read_csv("data/metadata.csv")

    player_df = getPlayers("data/metadata.csv")
    player_dict = createPlayerDict(player_df)

    filtered_df = filterMetadataByPlayer(metadata, player_dict)

    flattened_df = flattenData(filtered_df, "data/raw", player_dict)
    print(flattened_df)
    flattened_df.write_csv("data/positions.csv")

    logger.info("Position file built successfully")


if __name__ == "__main__":
    buildMetadataFile()
    buildPositionFile()
