import gc
import logging
from pathlib import Path

import chess
import chess.engine
import polars as pl
import tqdm

from utils.data import createPlayerDict, getPlayers

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Engine path specification and search evaluation parameters
STOCKFISH_PATH: str = "stockfish"
SEARCH_DEPTH: int = 10
MATE_SCORE_CAP: int = 10000

PREDICTION_FILES: dict[str, str] = {
    "Maia-2 Baseline": "data/maia2_predictions.csv",
    "Maia-2 FT": "data/maia2_ft_predictions.csv",
    "Maia-2 Nucleus": "data/maia2_nucleus_predictions.csv",
    "Maia-2 Descent": "data/maia2_descent_50_predictions.csv",
    "Maia-2 N. + Descent": "data/maia2_nucleus_descent_50_predictions.csv",
    "Maia-2 MCTS": "data/maia2_mcts_50_predictions.csv",
    "Maia-2 N. + MCTS": "data/maia2_nucleus_mcts_50_predictions.csv",
    "Maia-2 FT + N. + Descent": "data/maia2_ft_nucleus_descent_50_predictions.csv",
    "Maia-2 FT + N. + MCTS": "data/maia2_ft_nucleus_mcts_50_predictions.csv",
    "Maia-2 MoE-LoRA": "data/maia2_moe_lora_predictions.csv",
    "Maia-2 MoE-LoRA N. + Descent": (
        "data/maia2_moe_lora_nucleus_descent_50_predictions.csv"
    ),
    "Maia-2 MoE-LoRA N. + MCTS": (
        "data/maia2_moe_lora_nucleus_mcts_50_predictions.csv"
    ),
}

# In-memory memoization table for computed Centipawn Loss values
cpl_cache: dict[tuple[str, str], float] = {}


def get_position_and_move_cpl(
    engine: chess.engine.SimpleEngine,
    fen: str,
    move_uci: str,
    depth: int = SEARCH_DEPTH,
) -> float:
    r"""Compute the Centipawn Loss (CPL) for a specified move given a board state (FEN).

    Evaluates the optimal engine evaluation versus the evaluation following the candidate
    move execution. Results are memoized in an in-memory dictionary cache to accelerate
    subsequent evaluations across redundant state-action pairs.

    Args:
        engine (chess.engine.SimpleEngine): Initialized UCI chess engine instance.
        fen (str): Board state formatted as a Forsyth-Edwards Notation string.
        move_uci (str): Candidate move represented in Universal Chess Interface (UCI) format.
        depth (int, optional): Fixed engine search depth. Defaults to SEARCH_DEPTH.

    Returns:
        float: Computed Centipawn Loss $\ge 0.0$. Returns a penalty score of 1000.0 for
            illegal or null candidate moves.
    """
    cache_key = (fen, move_uci)
    if cache_key in cpl_cache:
        return cpl_cache[cache_key]

    board = chess.Board(fen)
    if not move_uci or chess.Move.from_uci(move_uci) not in board.legal_moves:
        return 1000.0

    info_best = engine.analyse(board, chess.engine.Limit(depth=depth))
    score_best = info_best["score"].pov(board.turn).score(mate_score=MATE_SCORE_CAP)

    move = chess.Move.from_uci(move_uci)
    board.push(move)
    info_after = engine.analyse(board, chess.engine.Limit(depth=depth))
    score_after = (
        info_after["score"].pov(not board.turn).score(mate_score=MATE_SCORE_CAP)
    )

    cpl = max(0.0, float(score_best - score_after))
    cpl_cache[cache_key] = cpl
    return cpl


def evaluate_model_cp_error(
    model_name: str,
    csv_path: str,
    players_dict: dict[str, int],
    engine: chess.engine.SimpleEngine,
) -> None:
    """Compute per-subject Centipawn Error (CP Error) for a designated model architecture.

    Quantifies prediction accuracy by evaluating the absolute difference between
    the mean human subject Centipawn Loss (CPL) and the mean model candidate move CPL.

    Args:
        model_name (str): Human-readable identifier for the candidate evaluation model.
        csv_path (str): File system path leading to the model target prediction CSV file.
        players_dict (Dict[str, int]): Mapping from subject identifiers to cohort indices.
        engine (chess.engine.SimpleEngine): UCI-compliant chess evaluation engine.
    """
    path = Path(csv_path)
    if not path.exists():
        logger.warning(
            f"Prediction record file not found: {csv_path}. Skipping model evaluation for: {model_name}."
        )
        return

    logger.info(
        f"Initiating optimized Centipawn Error evaluation pipeline for candidate model: {model_name}"
    )
    lazy_df = pl.scan_csv(csv_path)
    results = []

    for player_name, player_index in tqdm.tqdm(players_dict.items(), desc=model_name):
        player_data = (
            lazy_df.filter(pl.col("player_index") == player_index)
            .select(["fen", "move", "predicted_move"])
            .collect()
        )
        total_positions = len(player_data)

        if total_positions == 0:
            del player_data
            gc.collect()
            continue

        # Vectorized column extraction to eliminate iteration overhead over DataFrame rows
        fens: list[str] = player_data["fen"].to_list()
        player_moves: list[str] = player_data["move"].to_list()
        model_moves: list[str] = player_data["predicted_move"].to_list()

        player_cpl_sum = 0.0
        model_cpl_sum = 0.0

        for fen, p_move, m_move in zip(fens, player_moves, model_moves):
            player_cpl_sum += get_position_and_move_cpl(engine, fen, p_move)
            model_cpl_sum += get_position_and_move_cpl(engine, fen, m_move)

        mean_cpl_player = player_cpl_sum / total_positions
        mean_cpl_model = model_cpl_sum / total_positions
        cp_error = abs(mean_cpl_player - mean_cpl_model)

        results.append(
            {
                "player_index": player_index,
                "player_name": player_name,
                "mean_cpl_player": round(mean_cpl_player, 4),
                "mean_cpl_model": round(mean_cpl_model, 4),
                "cp_error": round(cp_error, 4),
            }
        )

        del player_data, fens, player_moves, model_moves
        gc.collect()

    if results:
        output_df = pl.DataFrame(results)
        clean_name = (
            model_name.lower()
            .replace(" ", "_")
            .replace("-", "_")
            .replace(".", "")
            .replace("+", "")
        )
        out_file = f"data/{clean_name}_cp_error.csv"
        output_df.write_csv(out_file)
        logger.info(f"Evaluation metrics successfully exported to: {out_file}")


if __name__ == "__main__":
    players = getPlayers("data/metadata.csv")
    players_dict = createPlayerDict(players)

    logger.info("Initializing Stockfish evaluation engine binary...")
    try:
        engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
    except Exception as e:
        logger.error(
            f"Failed to instantiate Stockfish process from binary path '{STOCKFISH_PATH}'. "
            "Ensure the executable is correctly installed and configured within system PATH."
        )
        raise e

    try:
        for model_name, path_str in PREDICTION_FILES.items():
            evaluate_model_cp_error(model_name, path_str, players_dict, engine)
    finally:
        engine.quit()
        logger.info("Stockfish engine process terminated successfully.")
