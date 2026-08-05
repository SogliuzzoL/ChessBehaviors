import csv
from pathlib import Path

# Configuration du dossier de sortie pour les tableaux LaTeX
OUTPUT_DIR = Path("tables")

# Mapping des noms de modèles vers les fichiers JSD réellement générés
RESULTS_CONFIG = {
    # Modèles Policy
    "Maia-2 Baseline": "data/maia_2_baseline_common_fens_jsd.csv",
    "Maia-2 FT": "data/maia_2_ft_common_fens_jsd.csv",
    "Maia-2 Nucleus": "data/maia_2_nucleus_common_fens_jsd.csv",
    "Maia-2 MoE-LoRA": "data/maia_2_moe_lora_common_fens_jsd.csv",
    # Modèles Descent
    "Maia-2 Descent": "data/maia_2_descent_common_fens_jsd.csv",
    "Maia-2 N. + Descent": "data/maia_2_n__descent_common_fens_jsd.csv",
    "Maia-2 FT + N. + Descent": "data/maia_2_ft__n__descent_common_fens_jsd.csv",
    "Maia-2 MoE-LoRA N. + Descent": (
        "data/maia_2_moe_lora_n__descent_common_fens_jsd.csv"
    ),
    # Modèles MCTS
    "Maia-2 MCTS": "data/maia_2_mcts_common_fens_jsd.csv",
    "Maia-2 N. + MCTS": "data/maia_2_n__mcts_common_fens_jsd.csv",
    "Maia-2 FT + N. + MCTS": "data/maia_2_ft__n__mcts_common_fens_jsd.csv",
    "Maia-2 MoE-LoRA N. + MCTS": ("data/maia_2_moe_lora_n__mcts_common_fens_jsd.csv"),
}

POLICY_MODELS = [
    "Maia-2 Baseline",
    "Maia-2 FT",
    "Maia-2 Nucleus",
    "Maia-2 MoE-LoRA",
]

DESCENT_MODELS = [
    "Maia-2 Descent",
    "Maia-2 N. + Descent",
    "Maia-2 FT + N. + Descent",
    "Maia-2 MoE-LoRA N. + Descent",
]

MCTS_MODELS = [
    "Maia-2 MCTS",
    "Maia-2 N. + MCTS",
    "Maia-2 FT + N. + MCTS",
    "Maia-2 MoE-LoRA N. + MCTS",
]


def load_jsd_data_from_csv(
    config: dict[str, str], metric_column: str = "mean_jsd"
) -> dict[str, dict[str, float]]:
    data = {}
    for model_name, path_str in config.items():
        path = Path(path_str)
        data[model_name] = {}
        if path.exists():
            with open(path, mode="r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    player = row["player_name"].strip()
                    val = float(row[metric_column])
                    data[model_name][player] = val
        else:
            print(f"Fichier introuvable ({path_str}), colonne laissée vide.")
    return data


def generate_summary_table(
    data: dict[str, dict[str, float]],
    caption: str = "Overall Jensen-Shannon Divergence (JSD) summary on common FEN positions across model variants.",
    label: str = "tab:jsd_summary_common_fens",
) -> str:
    model_stats = {}
    for model_name, player_dict in data.items():
        valid_vals = [
            v
            for k, v in player_dict.items()
            if k.lower() not in ["average", "mean", "avg"]
            and isinstance(v, (int, float))
        ]
        if valid_vals:
            mean_val = sum(valid_vals) / len(valid_vals)
            variance = sum((x - mean_val) ** 2 for x in valid_vals) / len(valid_vals)
            std_val = variance**0.5
            model_stats[model_name] = (mean_val, std_val)

    best_mean = min(stats[0] for stats in model_stats.values()) if model_stats else None

    latex = []
    latex.append("\\begin{table}[!htbp]")
    latex.append("  \\centering")
    latex.append(f"  \\caption{{{caption}}}")
    latex.append(f"  \\label{{{label}}}")
    latex.append("  \\begin{tabular}{lcc}")
    latex.append("    \\toprule")
    latex.append(
        "    \\textbf{Model Variant} & \\textbf{Mean JSD} & \\textbf{Std Dev} \\\\"
    )
    latex.append("    \\midrule")

    for model_name in data.keys():
        if model_name in model_stats:
            mean_val, std_val = model_stats[model_name]
            mean_str = f"{mean_val:.4f}"
            std_str = f"$\\pm$ {std_val:.4f}"

            if best_mean is not None and abs(mean_val - best_mean) < 1e-7:
                mean_str = f"\\textbf{{{mean_str}}}"
                std_str = f"\\textbf{{{std_str}}}"

            latex.append(f"    {model_name} & {mean_str} & {std_str} \\\\")
        else:
            latex.append(f"    {model_name} & -- & -- \\\\")

    latex.append("    \\bottomrule")
    latex.append("  \\end{tabular}")
    latex.append("\\end{table}")

    return "\n".join(latex)


def generate_player_breakdown_table(
    data: dict[str, dict[str, float]],
    selected_models: list[str] = None,
    caption: str = "Detailed Jensen-Shannon Divergence per player evaluated on common FEN positions.",
    label: str = "tab:jsd_breakdown_common_fens",
) -> str:
    if not selected_models:
        selected_models = list(data.keys())

    models = [m for m in selected_models if m in data]

    players = sorted(
        {
            player
            for model_name in models
            for player in data[model_name].keys()
            if player.lower() not in ["average", "mean", "avg"]
        }
    )

    col_spec = "l" + "c" * len(models)

    latex = []
    latex.append("\\begin{table}[!htbp]")
    latex.append("  \\centering")
    latex.append(f"  \\caption{{{caption}}}")
    latex.append(f"  \\label{{{label}}}")
    latex.append("  \\resizebox{\\linewidth}{!}{%")
    latex.append(f"    \\begin{{tabular}}{{{col_spec}}}")
    latex.append("      \\toprule")

    header = "      \\textbf{Player}"
    for model in models:
        header += f" & \\textbf{{{model}}}"
    header += " \\\\"
    latex.append(header)
    latex.append("      \\midrule")

    for player in players:
        player_jsds = {
            model: data[model].get(player)
            for model in models
            if data[model].get(player) is not None
        }
        min_jsd = min(player_jsds.values()) if player_jsds else None

        row = f"      {player}"
        for model in models:
            jsd_val = data[model].get(player, None)
            if jsd_val is not None:
                val_str = f"{jsd_val:.4f}"
                if min_jsd is not None and abs(jsd_val - min_jsd) < 1e-7:
                    val_str = f"\\textbf{{{val_str}}}"
                row += f" & {val_str}"
            else:
                row += " & --"
        row += " \\\\"
        latex.append(row)

    latex.append("      \\midrule")

    avg_vals = {}
    for model in models:
        model_dict = data[model]
        valid_vals = [
            v
            for k, v in model_dict.items()
            if k.lower() not in ["average", "mean", "avg"]
            and isinstance(v, (int, float))
        ]
        if valid_vals:
            avg_vals[model] = sum(valid_vals) / len(valid_vals)

    min_avg = min(avg_vals.values()) if avg_vals else None

    avg_row = "      \\textbf{Average}"
    for model in models:
        if model in avg_vals:
            avg_val = avg_vals[model]
            val_str = f"{avg_val:.4f}"
            if min_avg is not None and abs(avg_val - min_avg) < 1e-7:
                val_str = f"\\textbf{{{val_str}}}"
            avg_row += f" & {val_str}"
        else:
            avg_row += " & --"

    avg_row += " \\\\"
    latex.append(avg_row)

    latex.append("      \\bottomrule")
    latex.append("    \\end{tabular}%")
    latex.append("  }")
    latex.append("\\end{table}")

    return "\n".join(latex)


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    jsd_data = load_jsd_data_from_csv(RESULTS_CONFIG, metric_column="mean_jsd")

    summary_tex = generate_summary_table(jsd_data)
    with open(
        OUTPUT_DIR / "summary_common_fens_jsd_table.tex", "w", encoding="utf-8"
    ) as f:
        f.write(summary_tex)

    policy_tex = generate_player_breakdown_table(
        jsd_data,
        selected_models=POLICY_MODELS,
        caption="Jensen-Shannon Divergence on common FENs for Direct Policy Variants.",
        label="tab:jsd_policy_common_fens",
    )
    with open(
        OUTPUT_DIR / "policy_common_fens_jsd_table.tex", "w", encoding="utf-8"
    ) as f:
        f.write(policy_tex)

    descent_tex = generate_player_breakdown_table(
        jsd_data,
        selected_models=DESCENT_MODELS,
        caption="Jensen-Shannon Divergence on common FENs for Descent Search Variants.",
        label="tab:jsd_descent_common_fens",
    )
    with open(
        OUTPUT_DIR / "descent_common_fens_jsd_table.tex", "w", encoding="utf-8"
    ) as f:
        f.write(descent_tex)

    mcts_tex = generate_player_breakdown_table(
        jsd_data,
        selected_models=MCTS_MODELS,
        caption="Jensen-Shannon Divergence on common FENs for MCTS Search Variants.",
        label="tab:jsd_mcts_common_fens",
    )
    with open(
        OUTPUT_DIR / "mcts_common_fens_jsd_table.tex", "w", encoding="utf-8"
    ) as f:
        f.write(mcts_tex)

    print(f"Fichiers LaTeX générés avec succès dans le dossier '{OUTPUT_DIR}'.")
