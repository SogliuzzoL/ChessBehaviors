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


def compute_kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    """Calcule la divergence de Kullback-Leibler KL(P || Q)."""
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    return float(np.sum(p * np.log(p / q)))


def compute_jensen_shannon_divergence(
    p_dict: dict[str, float], q_dict: dict[str, float]
) -> float:
    """Calcule la divergence de Jensen-Shannon entre deux distributions de probabilités."""
    all_moves = list(set(p_dict.keys()).union(set(q_dict.keys())))
    if not all_moves:
        return 0.0

    p_vals = np.array([p_dict.get(m, 0.0) for m in all_moves], dtype=np.float64)
    q_vals = np.array([q_dict.get(m, 0.0) for m in all_moves], dtype=np.float64)

    sum_p = np.sum(p_vals)
    sum_q = np.sum(q_vals)

    if sum_p > 0:
        p_vals /= sum_p
    if sum_q > 0:
        q_vals /= sum_q

    m_vals = 0.5 * (p_vals + q_vals)

    kl_p_m = compute_kl_divergence(p_vals, m_vals)
    kl_q_m = compute_kl_divergence(q_vals, m_vals)

    jsd = 0.5 * kl_p_m + 0.5 * kl_q_m
    return max(0.0, jsd)


def extract_player_real_distributions(
    positions_file: Path, players_dict: dict[str, int]
) -> tuple[dict[int, dict[str, Counter]], set[str]]:
    """Charge de façon lazy les distributions de coups réelles et extrait l'ensemble de FENs communes."""
    lazy_pos = pl.scan_csv(positions_file)
    player_distributions: dict[int, dict[str, Counter]] = {}

    for player_name, p_idx in players_dict.items():
        logger.info(
            f"Extraction de la distribution réelle pour le joueur : {player_name}"
        )

        player_data = (
            lazy_pos.filter(pl.col("player_index") == p_idx)
            .select(["fen", "move"])
            .collect()
        )

        fen_counts: dict[str, Counter] = {}
        for row in player_data.iter_rows(named=True):
            fen = row["fen"]
            move = row["move"]
            if fen not in fen_counts:
                fen_counts[fen] = Counter()
            fen_counts[fen][move] += 1

        player_distributions[p_idx] = fen_counts
        del player_data
        gc.collect()

    # Recherche de l'intersection exacte des FENs présentes chez TOUS les joueurs
    all_player_fens = [set(dist.keys()) for dist in player_distributions.values()]
    common_fens = set.intersection(*all_player_fens) if all_player_fens else set()

    logger.info(
        f"Nombre total de FENs communes à l'ensemble des joueurs : {len(common_fens)}"
    )
    return player_distributions, common_fens


def build_inter_player_ground_truth_matrix(
    positions_file: Path, players_dict: dict[str, int]
) -> tuple[np.ndarray, list[str]]:
    """Calcule la matrice JSD inter-joueurs réels uniquement sur l'ensemble fixe de FENs communes."""
    player_names = list(players_dict.keys())
    n_players = len(player_names)
    matrix = np.zeros((n_players, n_players), dtype=np.float64)

    player_distributions, common_fens = extract_player_real_distributions(
        positions_file, players_dict
    )

    if not common_fens:
        logger.warning(
            "Aucune FEN stricte partagée par 100% des joueurs, repli sur l'intersection par paire."
        )

    for i, p_idx1 in enumerate(players_dict.values()):
        dist1 = player_distributions[p_idx1]
        for j, p_idx2 in enumerate(players_dict.values()):
            if i == j:
                matrix[i, j] = 0.0
                continue

            dist2 = player_distributions[p_idx2]
            eval_fens = (
                common_fens
                if common_fens
                else set(dist1.keys()).intersection(set(dist2.keys()))
            )

            if not eval_fens:
                matrix[i, j] = np.nan
                continue

            jsd_list = []
            for fen in eval_fens:
                c1 = dist1[fen]
                c2 = dist2[fen]

                tot1 = sum(c1.values())
                tot2 = sum(c2.values())

                p_data1 = {m: cnt / tot1 for m, cnt in c1.items()}
                p_data2 = {m: cnt / tot2 for m, cnt in c2.items()}

                jsd_list.append(compute_jensen_shannon_divergence(p_data1, p_data2))

            matrix[i, j] = np.mean(jsd_list) if jsd_list else np.nan

    return matrix, player_names


def build_model_vs_player_matrix(
    model_predictions_file: Path,
    positions_file: Path,
    players_dict: dict[str, int],
) -> tuple[np.ndarray, list[str]]:
    """Calcule la matrice JSD modèle vs joueurs réels sur un ensemble de FENs strictement fixe et commun."""
    player_names = list(players_dict.keys())
    n_players = len(player_names)
    matrix = np.zeros((n_players, n_players), dtype=np.float64)

    # 1. Extraction des distributions réelles et identification des FENs communes à tous les joueurs
    player_distributions, common_fens = extract_player_real_distributions(
        positions_file, players_dict
    )

    real_distributions = {
        p_idx: {
            fen: {m: cnt / sum(c.values()) for m, cnt in c.items()}
            for fen, c in fen_counts.items()
        }
        for p_idx, fen_counts in player_distributions.items()
    }

    lazy_preds = pl.scan_csv(model_predictions_file)

    # 2. Traitement par modèle conditionné
    for i, model_p_idx in enumerate(players_dict.values()):
        model_player_name = player_names[i]
        logger.info(
            f"Évaluation du modèle conditionné sur : {model_player_name} (Lazy Mode)"
        )

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

        # Détermination du jeu de FENs d'évaluation fixe
        if common_fens:
            # Intersection des FENs communes à tous les joueurs avec celles prédites par ce modèle
            eval_fens = common_fens.intersection(set(model_probs_by_fen.keys()))
        else:
            eval_fens = set(model_probs_by_fen.keys())

        logger.info(
            f"Nombre de FENs évaluées pour {model_player_name} : {len(eval_fens)}"
        )

        # 3. Comparaison croisée sur ces mêmes FENs pour chaque joueur réel j
        for j, real_p_idx in enumerate(players_dict.values()):
            real_dist = real_distributions[real_p_idx]

            target_fens = [f for f in eval_fens if f in real_dist]

            if not target_fens:
                matrix[i, j] = np.nan
                continue

            jsd_list = []
            for fen in target_fens:
                p_model = model_probs_by_fen[fen]
                p_real = real_dist[fen]
                jsd_list.append(compute_jensen_shannon_divergence(p_model, p_real))

            matrix[i, j] = np.mean(jsd_list) if jsd_list else np.nan

    return matrix, player_names


def plot_inter_player_jsd_heatmap(
    jsd_matrix: np.ndarray,
    player_names: list[str],
    output_path: Path,
    title: str = "Inter-Player Real Behavior JSD Divergence",
):
    """Génère la heatmap de divergence JSD entre les joueurs réels du dataset."""
    plt.figure(figsize=(10, 8))

    sns.heatmap(
        jsd_matrix,
        xticklabels=player_names,
        yticklabels=player_names,
        annot=True,
        fmt=".3f",
        cmap="YlGnBu",
        cbar_kws={"label": "Jensen-Shannon Divergence"},
        linewidths=0.5,
    )

    plt.title(title, fontsize=12, pad=12, fontweight="bold")
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Heatmap enregistrée sous : {output_path}")


def plot_model_vs_player_jsd_heatmap(
    jsd_matrix: np.ndarray,
    player_names: list[str],
    output_path: Path,
    model_name: str = "Maia-2 MoE-LoRA",
):
    """Génère la heatmap opposant le modèle conditionné au comportement réel des joueurs."""
    plt.figure(figsize=(11, 8.5))

    sns.heatmap(
        jsd_matrix,
        xticklabels=player_names,
        yticklabels=[f"Model ({p})" for p in player_names],
        annot=True,
        fmt=".3f",
        cmap="viridis_r",
        cbar_kws={"label": "Jensen-Shannon Divergence"},
        linewidths=0.5,
    )

    plt.title(
        f"Stylistic Alignment Matrix: {model_name} vs Real Players",
        fontsize=12,
        pad=12,
        fontweight="bold",
    )
    plt.xlabel("Real Target Player", fontsize=10, fontweight="bold")
    plt.ylabel("Model Conditioned Player", fontsize=10, fontweight="bold")

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
    model_predictions_file = DATA_DIR / "maia2_moe_lora_predictions.csv"

    if not metadata_file.exists() or not positions_file.exists():
        logger.error(
            "Les fichiers de données metadata.csv ou positions.csv sont introuvables."
        )
        exit(1)

    players_df = getPlayers(str(metadata_file))
    players_dict = createPlayerDict(players_df)

    # 1. Matrice de divergence réelle inter-joueurs sur FENs communes
    inter_matrix, player_names = build_inter_player_ground_truth_matrix(
        positions_file, players_dict
    )
    plot_inter_player_jsd_heatmap(
        inter_matrix, player_names, OUTPUT_DIR / "inter_player_jsd_heatmap.pdf"
    )

    # 2. Matrice d'alignement modèle vs joueurs réels sur FENs communes
    if model_predictions_file.exists():
        model_matrix, _ = build_model_vs_player_matrix(
            model_predictions_file, positions_file, players_dict
        )
        plot_model_vs_player_jsd_heatmap(
            model_matrix,
            player_names,
            OUTPUT_DIR / "model_vs_player_jsd_heatmap.pdf",
            model_name="Maia-2 MoE-LoRA",
        )
    else:
        logger.warning(
            f"Le fichier de prédictions {model_predictions_file} n'existe pas encore."
        )
