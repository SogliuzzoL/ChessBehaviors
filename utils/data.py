"""
Data preprocessing and splitting utilities for parsing PGN games, extracting positional
observations, mapping subject cohorts, and preparing deterministic cross-validation folds.
"""

import os
import random

import chess
import numpy as np
import polars as pl
import torch
import tqdm
from chess import pgn


def getPGNmetadata(file_path: str, user_id: str, game_id: str) -> dict[str, str] | None:
    """Extract game metadata headers from a target Portable Game Notation (PGN) record.

    Args:
        file_path (str): File system path leading to target PGN file.
        user_id (str): User identifier corresponding to player cohort directory.
        game_id (str): Unique game instance identifier.

    Returns:
        Optional[Dict[str, str]]: Dictionary containing extracted header fields,
            or None if record parsing fails.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        game = pgn.read_game(f)
        if game is None:
            return None
        return {
            "game_id": game_id,
            "user_id": user_id,
            "White": game.headers.get("White", ""),
            "Black": game.headers.get("Black", ""),
            "Date": game.headers.get("Date", ""),
            "ECO": game.headers.get("ECO", ""),
            "Result": game.headers.get("Result", ""),
        }


def getPlayers(file_path: str, game_count_threshold: int = 2000) -> pl.DataFrame:
    """Identify player subjects exceeding specified observational game activity thresholds.

    Args:
        file_path (str): File path to metadata catalog containing player match histories.
        game_count_threshold (int, optional): Minimum required observations per subject cohort.
            Defaults to 2000.

    Returns:
        pl.DataFrame: Filtered DataFrame containing player identifiers and game counts
            sorted descending.
    """
    metadata = pl.read_csv(file_path, columns=["White", "Black"])

    df = (
        pl.concat([metadata["White"], metadata["Black"]])
        .value_counts()
        .rename({"White": "player", "count": "count"})
    )

    df = df.filter(pl.col("count") > game_count_threshold)
    df = df.sort("count", descending=True)

    return df


def createPlayerDict(df: pl.DataFrame) -> dict[str, int]:
    """Construct ordinal subject index mapping from filtered player cohorts.

    Args:
        df (pl.DataFrame): DataFrame containing unique subject cohort names.

    Returns:
        Dict[str, int]: Dictionary mapping player name strings to categorical integer indices.
    """
    return dict(zip(df["player"], list(range(len(df)))))


def filterMetadataByPlayer(
    metadata: pl.DataFrame, player_dict: dict[str, int]
) -> pl.DataFrame:
    """Filter metadata catalog to retain matches involving targeted player cohorts.

    Args:
        metadata (pl.DataFrame): Master metadata DataFrame.
        player_dict (Dict[str, int]): Map of target subject profiles to categorical indices.

    Returns:
        pl.DataFrame: Subsampled metadata DataFrame.
    """
    return metadata.filter(
        pl.col("White").is_in(player_dict) | pl.col("Black").is_in(player_dict)
    )


def flattenData(
    metadata: pl.DataFrame, data_dir: str, player_dict: dict[str, int]
) -> pl.DataFrame:
    """Flatten game PGN records into individual ply-level positional observations.

    Extracts legal position state-action pairs (FEN, move) and maps participating
    subjects to their respective categorical indices.

    Args:
        metadata (pl.DataFrame): Catalog of candidate game records.
        data_dir (str): Base directory storing user PGN files.
        player_dict (Dict[str, int]): Mapping of target subject names to categorical indices.

    Returns:
        pl.DataFrame: Polars DataFrame where each row represents a single decision point.
    """
    positions: list[dict[str, str | int]] = []
    processed_games: list[str] = []

    for row in tqdm.tqdm(
        metadata.iter_rows(named=True), total=len(metadata), desc="Flattening Positions"
    ):
        game_id = row["game_id"]
        user_id = row["user_id"]
        white = row["White"]
        black = row["Black"]
        result = row["Result"]

        if game_id in processed_games:
            continue
        processed_games.append(game_id)

        white_index = player_dict.get(white, -1)
        black_index = player_dict.get(black, -1)

        if white_index == -1 and black_index == -1:
            continue

        file_path = os.path.join(data_dir, str(user_id), f"{game_id}.pgn")
        if not os.path.exists(file_path):
            continue

        with open(file_path, "r", encoding="utf-8") as f:
            game = pgn.read_game(f)
            if game is None:
                continue

            initial_fen = game.headers.get("FEN", None)

            board = chess.Board()
            if initial_fen is not None:
                board.set_fen(initial_fen)

            for move in game.mainline_moves():
                turn = board.turn
                fen = board.fen()
                board.push(move)

                position = {
                    "game_id": game_id,
                    "turn": "white" if turn == chess.WHITE else "black",
                    "fen": fen,
                    "move": move.uci(),
                    "result": result,
                }
                if turn == chess.WHITE and white_index != -1:
                    position["player_index"] = white_index
                    positions.append(position)
                elif turn == chess.BLACK and black_index != -1:
                    position["player_index"] = black_index
                    positions.append(position)

    return pl.DataFrame(positions)


def set_seed(seed: int = 42) -> None:
    """Enforce strict experimental reproducibility across execution environments.

    Configures global random seeds for Python native built-ins, NumPy, PyTorch CPU,
    and PyTorch CUDA backends while forcing deterministic cuDNN algorithms.

    Args:
        seed (int, optional): Global seed integer. Defaults to 42.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def split_games_into_folds(
    game_ids: list[str], n_splits: int = 5, seed: int = 42
) -> list[list[str]]:
    """Partition unique game identifiers into balanced cross-validation folds.

    Ensures zero data leakage between training and evaluation splits by partitioning
    at the game entity level rather than individual position observations.

    Args:
        game_ids (List[str]): Unique game identifier strings.
        n_splits (int, optional): K-fold partition split count. Defaults to 5.
        seed (int, optional): Seed establishing deterministic shuffling order. Defaults to 42.

    Returns:
        List[List[str]]: List of lists containing partitioned game IDs for each fold.
    """
    shuffled_games = game_ids.copy()
    random.seed(seed)
    random.shuffle(shuffled_games)

    folds: list[list[str]] = [[] for _ in range(n_splits)]
    for idx, game_id in enumerate(shuffled_games):
        folds[idx % n_splits].append(game_id)
    return folds
