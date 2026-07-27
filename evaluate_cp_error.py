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

# Configuration des chemins et paramètres Stockfish
STOCKFISH_PATH = "stockfish"
SEARCH_DEPTH = 10
MATE_SCORE_CAP = 10000

PREDICTION_FILES = {
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

cpl_cache: dict[tuple[str, str], float] = {}


def get_position_and_move_cpl(
    engine: chess.engine.SimpleEngine,
    fen: str,
    move_uci: str,
    depth: int = SEARCH_DEPTH,
) -> float:
    """Calcule le Centipawn Loss d'un coup donné à partir d'une FEN."""
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
    board.pop()

    cpl = max(0.0, float(score_best - score_after))
    cpl_cache[cache_key] = cpl
    return cpl


def evaluate_model_cp_error(
    model_name: str,
    csv_path: str,
    players_dict: dict[str, int],
    engine: chess.engine.SimpleEngine,
):
    """Calcule la CP Error par joueur en mode Lazy Polars."""
    path = Path(csv_path)
    if not path.exists():
        logger.warning(
            f"Fichier de prédictions introuvable : {csv_path}, passage au suivant."
        )
        return

    logger.info(f"Évaluation du modèle (Lazy Mode) : {model_name}")
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

        player_cpl_sum = 0.0
        model_cpl_sum = 0.0

        for row in player_data.iter_rows(named=True):
            fen = row["fen"]
            player_move = row["move"]
            model_move = row["predicted_move"]

            player_cpl = get_position_and_move_cpl(engine, fen, player_move)
            model_cpl = get_position_and_move_cpl(engine, fen, model_move)

            player_cpl_sum += player_cpl
            model_cpl_sum += model_cpl

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

        del player_data
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
        logger.info(f"Résultats enregistrés dans : {out_file}")


if __name__ == "__main__":
    players = getPlayers("data/metadata.csv")
    players_dict = createPlayerDict(players)

    logger.info("Initialisation du moteur Stockfish...")
    try:
        engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
    except Exception as e:
        logger.error(
            f"Impossible d'ouvrir Stockfish depuis '{STOCKFISH_PATH}'. "
            "Vérifie qu'il est bien installé et accessible dans le PATH."
        )
        raise e

    try:
        for model_name, path_str in PREDICTION_FILES.items():
            evaluate_model_cp_error(model_name, path_str, players_dict, engine)
    finally:
        engine.quit()
        logger.info("Moteur Stockfish fermé avec succès.")
