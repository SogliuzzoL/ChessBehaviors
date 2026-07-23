import os
import random

import chess
import chess.pgn as pgn
import numpy as np
import polars as pl
import torch
import tqdm


def getPGNmetadata(file_path: str, user_id: str, game_id: str) -> dict | None:
    """
    Extracts metadata from a PGN file.
    """
    with open(file_path, "r") as f:
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
    """
    Returns a DataFrame of players with a game count above the threshold.
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
    """
    Returns a dictionary mapping player names to their index in the DataFrame.
    """
    return dict(zip(df["player"], [i for i in range(len(df))]))


def filterMetadataByPlayer(
    metadata: pl.DataFrame, player_dict: dict[str, int]
) -> pl.DataFrame:
    """
    Returns a DataFrame filtered to include only games played by players in the player_dict.
    """
    return metadata.filter(
        pl.col("White").is_in(player_dict) | pl.col("Black").is_in(player_dict)
    )


def flattenData(
    metadata: pl.DataFrame, data_dir: str, player_dict: dict[str, int]
) -> pl.DataFrame:
    """
    Flattens the metadata into a DataFrame of positions, with player names replaced by their index in the player_dict.
    """
    positions = []
    games_id = []
    for row in tqdm(metadata.iter_rows(named=True), total=len(metadata)):
        game_id = row["game_id"]
        user_id = row["user_id"]
        white = row["White"]
        black = row["Black"]
        result = row["Result"]

        if game_id in games_id:
            continue
        games_id.append(game_id)

        white_index = player_dict.get(white, -1)
        black_index = player_dict.get(black, -1)

        if white_index == -1 and black_index == -1:
            continue

        with open(f"{data_dir}/{user_id}/{game_id}.pgn", "r") as f:
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

    df = pl.DataFrame(positions)
    return df


def set_seed(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def split_games_into_folds(
    game_ids: list, n_splits: int = 5, seed: int = 42
) -> list[list]:
    shuffled_games = game_ids.copy()
    random.seed(seed)
    random.shuffle(shuffled_games)

    folds = [[] for _ in range(n_splits)]
    for idx, game_id in enumerate(shuffled_games):
        folds[idx % n_splits].append(game_id)
    return folds
