import logging
import os
from typing import Any

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


def buildMetadataFile() -> None:
    """Extract and compile metadata across raw Portable Game Notation (PGN) files.

    Iterates through structured directories containing raw PGN files, extracts
    game-level metadata for each record, and persists the aggregated output
    to a tabular CSV format for downstream analysis.
    """
    logger.info("Initiating metadata extraction pipeline.")

    metadata: list[dict[str, Any]] = []
    raw_data_dir: str = "data/raw"

    for folder in tqdm.tqdm(
        os.listdir(raw_data_dir), desc="Processing user directories"
    ):
        folder_path: str = os.path.join(raw_data_dir, folder)
        if not os.path.isdir(folder_path):
            continue

        for file in tqdm.tqdm(
            os.listdir(folder_path),
            desc=f"Parsing games for user {folder}",
            leave=False,
        ):
            if not file.endswith(".pgn"):
                continue

            file_path: str = os.path.join(folder_path, file)
            game_id: str = file.replace(".pgn", "")

            result: dict[str, Any] | None = getPGNmetadata(
                file_path=file_path,
                user_id=folder,
                game_id=game_id,
            )

            if result:
                metadata.append(result)
            else:
                logger.warning(
                    "Failed to extract metadata from PGN record: %s/%s", folder, file
                )

    df: pl.DataFrame = pl.DataFrame(metadata)
    print(df)

    output_path: str = "data/metadata.csv"
    df.write_csv(output_path)

    logger.info(
        "Metadata extraction completed successfully. Output saved to %s", output_path
    )


def buildPositionFile() -> None:
    """Construct a board position dataset from filtered game metadata.

    Loads compiled metadata, filters records against targeted cohort parameters,
    and flattens game histories into ply-by-ply position records.
    The resulting dataset is exported as a CSV file.
    """
    logger.info("Initiating position dataset construction pipeline.")

    metadata_path: str = "data/metadata.csv"
    metadata: pl.DataFrame = pl.read_csv(metadata_path)

    player_df: pl.DataFrame = getPlayers(metadata_path)
    player_dict: dict[str, Any] = createPlayerDict(player_df)

    filtered_df: pl.DataFrame = filterMetadataByPlayer(metadata, player_dict)

    flattened_df: pl.DataFrame = flattenData(filtered_df, "data/raw", player_dict)
    print(flattened_df)

    output_path: str = "data/positions.csv"
    flattened_df.write_csv(output_path)

    logger.info(
        "Position dataset constructed successfully. Output saved to %s", output_path
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    buildMetadataFile()
    buildPositionFile()
