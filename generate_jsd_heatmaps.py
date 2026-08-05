import ast
import gc
import logging
import math
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns

from utils.data import createPlayerDict, getPlayers

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("figures")
DATA_DIR = Path("data")

PREDICTION_FILES = {
    "Maia-2 Baseline": DATA_DIR / "maia2_predictions.csv",
    "Maia-2 FT": DATA_DIR / "maia2_ft_predictions.csv",
    "Maia-2 Nucleus": DATA_DIR / "maia2_nucleus_predictions.csv",
    "Maia-2 Descent": DATA_DIR / "maia2_descent_50_predictions.csv",
    "Maia-2 N. + Descent": DATA_DIR / "maia2_nucleus_descent_50_predictions.csv",
    "Maia-2 MCTS": DATA_DIR / "maia2_mcts_50_predictions.csv",
    "Maia-2 N. + MCTS": DATA_DIR / "maia2_nucleus_mcts_50_predictions.csv",
    "Maia-2 FT + N. + Descent": DATA_DIR
    / "maia2_ft_nucleus_descent_50_predictions.csv",
    "Maia-2 FT + N. + MCTS": DATA_DIR / "maia2_ft_nucleus_mcts_50_predictions.csv",
    "Maia-2 MoE-LoRA": DATA_DIR / "maia2_moe_lora_predictions.csv",
    "Maia-2 MoE-LoRA N. + Descent": (
        DATA_DIR / "maia2_moe_lora_nucleus_descent_50_predictions.csv"
    ),
    "Maia-2 MoE-LoRA N. + MCTS": (
        DATA_DIR / "maia2_moe_lora_nucleus_mcts_50_predictions.csv"
    ),
}


def compute_jensen_shannon_divergence_fast(
    p_dict: dict[str, float], q_dict: dict[str, float], eps: float = 1e-12
) -> float:
    """Calcul optimisé de la JSD entre deux distributions par dictionnaires."""
    all_moves = list(set(p_dict.keys()).union(set(q_dict.keys())))
    if not all_moves:
        return 0.0

    p_vals = np.array([p_dict.get(m, 0.0) for m in all_moves], dtype=np.float64)
    q_vals = np.array([q_dict.get(m, 0.0) for m in all_moves], dtype=np.float64)

    sum_p, sum_q = np.sum(p_vals), np.sum(q_vals)
    if sum_p > 0:
        p_vals /= sum_p
    if sum_q > 0:
        q_vals /= sum_q

    m_vals = 0.5 * (p_vals + q_vals)

    p_clipped = np.clip(p_vals, eps, 1.0)
    q_clipped = np.clip(q_vals, eps, 1.0)
    m_clipped = np.clip(m_vals, eps, 1.0)

    kl_p_m = float(np.sum(p_vals * np.log(p_clipped / m_clipped)))
    kl_q_m = float(np.sum(q_vals * np.log(q_clipped / m_clipped)))

    return max(0.0, 0.5 * kl_p_m + 0.5 * kl_q_m)


def extract_player_real_distributions(
    positions_file: Path, players_dict: dict[str, int]
) -> tuple[dict[int, dict[str, dict[str, float]]], set[str]]:
    """Charge et normalise une seule fois en mémoire les distributions réelles des joueurs."""
    logger.info("Extraction globale des coups réels des joueurs...")
    lazy_pos = pl.scan_csv(positions_file)
    valid_indices = set(players_dict.values())

    # Extraction optimisée Polars par regroupement
    df_counts = (
        lazy_pos.filter(pl.col("player_index").is_in(valid_indices))
        .group_by(["player_index", "fen", "move"])
        .agg(pl.len().alias("count"))
        .collect()
    )

    # Détermination des FENs strictement communes à TOUS les joueurs
    fen_player_counts = (
        df_counts.select(["player_index", "fen"])
        .unique()
        .group_by("fen")
        .agg(pl.len().alias("p_count"))
        .filter(pl.col("p_count") == len(valid_indices))
    )
    common_fens = set(fen_player_counts["fen"].to_list())
    logger.info(f"Nombre de FENs communes à tous les joueurs : {len(common_fens)}")

    # Restriction aux FENs communes et calcul des distributions relatives
    df_filtered = df_counts.filter(pl.col("fen").is_in(common_fens))

    real_distributions: dict[int, dict[str, dict[str, float]]] = {
        p_idx: {} for p_idx in valid_indices
    }

    fen_totals: dict[tuple[int, str], int] = {}
    for row in df_filtered.iter_rows(named=True):
        key = (row["player_index"], row["fen"])
        fen_totals[key] = fen_totals.get(key, 0) + row["count"]

    for row in df_filtered.iter_rows(named=True):
        p_idx, fen, move, cnt = (
            row["player_index"],
            row["fen"],
            row["move"],
            row["count"],
        )
        if fen not in real_distributions[p_idx]:
            real_distributions[p_idx][fen] = {}
        total = fen_totals[(p_idx, fen)]
        real_distributions[p_idx][fen][move] = cnt / total

    del df_counts, df_filtered, fen_totals
    gc.collect()

    return real_distributions, common_fens


def build_inter_player_ground_truth_matrix(
    real_distributions: dict[int, dict[str, dict[str, float]]],
    common_fens: set[str],
    players_dict: dict[str, int],
) -> tuple[np.ndarray, list[str]]:
    """Calcule la matrice JSD entre les joueurs réels sur les FENs communes."""
    player_names = list(players_dict.keys())
    n_players = len(player_names)
    matrix = np.zeros((n_players, n_players), dtype=np.float64)

    for i, p_idx1 in enumerate(players_dict.values()):
        dist1 = real_distributions[p_idx1]
        for j, p_idx2 in enumerate(players_dict.values()):
            if i == j:
                matrix[i, j] = 0.0
                continue

            dist2 = real_distributions[p_idx2]
            jsd_list = [
                compute_jensen_shannon_divergence_fast(dist1[fen], dist2[fen])
                for fen in common_fens
                if fen in dist1 and fen in dist2
            ]

            matrix[i, j] = np.mean(jsd_list) if jsd_list else np.nan

    return matrix, player_names


def build_model_vs_player_matrix(
    model_predictions_file: Path,
    real_distributions: dict[int, dict[str, dict[str, float]]],
    common_fens: set[str],
    players_dict: dict[str, int],
) -> tuple[np.ndarray, list[str]]:
    """Calcule la matrice JSD pour un modèle vs joueurs réels sur les FENs communes."""
    player_names = list(players_dict.keys())
    n_players = len(player_names)
    matrix = np.zeros((n_players, n_players), dtype=np.float64)

    lazy_preds = pl.scan_csv(model_predictions_file).filter(
        pl.col("fen").is_in(common_fens)
    )

    for i, model_p_idx in enumerate(players_dict.values()):
        model_player_name = player_names[i]

        model_data = (
            lazy_preds.filter(pl.col("player_index") == model_p_idx)
            .select(["fen", "moves_probs"])
            .collect()
        )

        model_probs_by_fen: dict[str, dict[str, float]] = {}
        for row in model_data.iter_rows(named=True):
            fen = row["fen"]
            if fen not in model_probs_by_fen:
                raw_probs = row["moves_probs"]
                if isinstance(raw_probs, str):
                    try:
                        model_probs_by_fen[fen] = ast.literal_eval(raw_probs)
                    except Exception:
                        model_probs_by_fen[fen] = {}
                elif isinstance(raw_probs, dict):
                    model_probs_by_fen[fen] = raw_probs
                else:
                    model_probs_by_fen[fen] = {}

        del model_data
        gc.collect()

        for j, real_p_idx in enumerate(players_dict.values()):
            real_dist = real_distributions[real_p_idx]
            target_fens = [
                f for f in common_fens if f in model_probs_by_fen and f in real_dist
            ]

            if not target_fens:
                matrix[i, j] = np.nan
                continue

            jsd_list = [
                compute_jensen_shannon_divergence_fast(
                    model_probs_by_fen[fen], real_dist[fen]
                )
                for fen in target_fens
            ]

            matrix[i, j] = np.mean(jsd_list) if jsd_list else np.nan

    return matrix, player_names


def plot_jsd_heatmap(
    jsd_matrix: np.ndarray,
    player_names: list[str],
    output_path: Path,
    title: str,
    y_label: str = "Model Conditioned Player",
    is_inter_player: bool = False,
):
    """Génère et sauvegarde une heatmap JSD optimisée."""
    plt.figure(figsize=(11, 8.5))

    y_labels = (
        player_names if is_inter_player else [f"Model ({p})" for p in player_names]
    )
    cmap = "YlGnBu" if is_inter_player else "viridis_r"

    sns.heatmap(
        jsd_matrix,
        xticklabels=player_names,
        yticklabels=y_labels,
        annot=True,
        fmt=".3f",
        cmap=cmap,
        cbar_kws={"label": "Jensen-Shannon Divergence"},
        linewidths=0.5,
    )

    plt.title(title, fontsize=12, pad=12, fontweight="bold")
    plt.xlabel("Real Target Player", fontsize=10, fontweight="bold")
    plt.ylabel(y_label, fontsize=10, fontweight="bold")

    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Heatmap enregistrée sous : {output_path}")


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    metadata_file = DATA_DIR / "metadata.csv"
    positions_file = DATA_DIR / "positions.csv"

    if not metadata_file.exists() or not positions_file.exists():
        logger.error(
            "Les fichiers de données metadata.csv ou positions.csv sont introuvables."
        )
        exit(1)

    players_df = getPlayers(str(metadata_file))
    players_dict = createPlayerDict(players_df)

    # 1. Chargement unique des distributions réelles et identification des FENs communes
    real_distributions, common_fens = extract_player_real_distributions(
        positions_file, players_dict
    )

    if not common_fens:
        logger.error(
            "Aucune position (FEN) commune à tous les joueurs n'a été trouvée."
        )
        exit(1)

    # 2. Heatmap Inter-Joueurs Réels (Vérité terrain)
    inter_matrix, player_names = build_inter_player_ground_truth_matrix(
        real_distributions, common_fens, players_dict
    )
    plot_jsd_heatmap(
        inter_matrix,
        player_names,
        OUTPUT_DIR / "inter_player_jsd_heatmap.pdf",
        title="Inter-Player Real Behavior JSD Divergence (Common FENs)",
        y_label="Real Player",
        is_inter_player=True,
    )

    # 3. Génération des Heatmaps pour CHAQUE modèle présent dans PREDICTION_FILES
    for model_name, path in PREDICTION_FILES.items():
        if not path.exists():
            logger.warning(
                f"Fichier introuvable pour le modèle '{model_name}' : {path}, ignoré."
            )
            continue

        logger.info(f"Génération de la heatmap pour le modèle : {model_name}")

        model_matrix, _ = build_model_vs_player_matrix(
            path, real_distributions, common_fens, players_dict
        )

        clean_name = (
            model_name.lower()
            .replace(" ", "_")
            .replace("-", "_")
            .replace(".", "")
            .replace("+", "")
        )
        out_file = OUTPUT_DIR / f"{clean_name}_vs_player_jsd_heatmap.pdf"

        plot_jsd_heatmap(
            model_matrix,
            player_names,
            out_file,
            title=f"Stylistic Alignment Matrix: {model_name} vs Real Players",
            is_inter_player=False,
        )
