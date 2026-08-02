import csv
from pathlib import Path

# Configuration du dossier de sortie pour les tableaux LaTeX
OUTPUT_DIR = Path("tables")

# Configuration : Dictionnaire associant le nom du modèle au chemin du fichier CSV
RESULTS_CONFIG = {
    # Modèles Policy
    "Maia-2 Baseline": "data/maia2_accuracies.csv",
    "Maia-2 FT": "data/maia2_ft_accuracies.csv",
    "Maia-2 Nucleus": "data/maia2_nucleus_accuracies.csv",
    "Maia-2 MoE-LoRA": "data/maia2_moe_lora_accuracies.csv",
    # Modèles Descent
    "Maia-2 Descent": "data/maia2_descent_50_accuracies.csv",
    "Maia-2 N. + Descent": "data/maia2_nucleus_descent_50_accuracies.csv",
    "Maia-2 FT + N. + Descent": "data/maia2_ft_nucleus_descent_50_accuracies.csv",
    "Maia-2 MoE-LoRA N. + Descent": (
        "data/maia2_moe_lora_nucleus_descent_50_accuracies.csv"
    ),
    # Modèles MCTS
    "Maia-2 MCTS": "data/maia2_mcts_accuracies.csv",
    "Maia-2 N. + MCTS": "data/maia2_nucleus_mcts_accuracies.csv",
    "Maia-2 FT + N. + MCTS": "data/maia2_ft_nucleus_mcts_accuracies.csv",
    "Maia-2 MoE-LoRA N. + MCTS": ("data/maia2_moe_lora_nucleus_mcts_accuracies.csv"),
}

# Groupes logiques pour séparer les tableaux par paradigme (4 modèles par tableau)
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


def load_accuracy_data_from_csv(
    config: dict[str, str],
) -> dict[str, dict[str, float]]:
    """Charge les données d'accuracy depuis les fichiers CSV configurés."""
    data = {}
    for model_name, path_str in config.items():
        path = Path(path_str)
        data[model_name] = {}
        if path.exists():
            with open(path, mode="r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    player = row["player_name"].strip()
                    accuracy = float(row["accuracy"])
                    data[model_name][player] = accuracy
        else:
            print(f"Fichier introuvable ({path_str}), colonne laissée vide.")
    return data


def generate_summary_table(
    data: dict[str, dict[str, float]],
    caption: str = "Overall Move Accuracy summary across model variants.",
    label: str = "tab:accuracy_summary",
) -> str:
    """Génère un tableau de synthèse compact en mettant en gras la meilleure moyenne globale."""
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

    best_mean = max(stats[0] for stats in model_stats.values()) if model_stats else None

    latex = []
    latex.append("\\begin{table}[!htbp]")
    latex.append("  \\centering")
    latex.append(f"  \\caption{{{caption}}}")
    latex.append(f"  \\label{{{label}}}")
    latex.append("  \\begin{tabular}{lcc}")
    latex.append("    \\toprule")
    latex.append(
        "    \\textbf{Model Variant} & \\textbf{Mean Accuracy (\\%)} &"
        " \\textbf{Std Dev (\\%)} \\\\"
    )
    latex.append("    \\midrule")

    for model_name in data.keys():
        if model_name in model_stats:
            mean_val, std_val = model_stats[model_name]
            mean_str = f"{mean_val * 100:.2f}\\%"
            std_str = f"$\\pm$ {std_val * 100:.2f}\\%"

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
    caption: str = "Detailed move accuracy per player.",
    label: str = "tab:move_accuracy_breakdown",
) -> str:
    """Génère un tableau détaillé en mettant en gras la meilleure précision par ligne."""
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

    # Traitement par joueur avec identification du maximum de la ligne
    for player in players:
        player_accs = {
            model: data[model].get(player)
            for model in models
            if data[model].get(player) is not None
        }
        max_acc = max(player_accs.values()) if player_accs else None

        row = f"      {player}"
        for model in models:
            acc = data[model].get(player, None)
            if acc is not None:
                val_str = f"{acc * 100:.2f}\\%"
                if max_acc is not None and abs(acc - max_acc) < 1e-7:
                    val_str = f"\\textbf{{{val_str}}}"
                row += f" & {val_str}"
            else:
                row += " & --"
        row += " \\\\"
        latex.append(row)

    latex.append("      \\midrule")

    # Traitement de la ligne Moyenne avec identification de la meilleure moyenne
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

    max_avg = max(avg_vals.values()) if avg_vals else None

    avg_row = "      \\textbf{Average}"
    for model in models:
        if model in avg_vals:
            avg_val = avg_vals[model]
            val_str = f"{avg_val * 100:.2f}\\%"
            if max_avg is not None and abs(avg_val - max_avg) < 1e-7:
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

    accuracy_data = load_accuracy_data_from_csv(RESULTS_CONFIG)

    # 1. Tableau de synthèse global (tous les modèles)
    summary_tex = generate_summary_table(accuracy_data)
    with open(OUTPUT_DIR / "summary_accuracy_table.tex", "w", encoding="utf-8") as f:
        f.write(summary_tex)

    # 2. Tableau détaillé Policy (4 modèles)
    policy_tex = generate_player_breakdown_table(
        accuracy_data,
        selected_models=POLICY_MODELS,
        caption="Move Accuracy for Direct Policy Variants.",
        label="tab:accuracy_policy",
    )
    with open(OUTPUT_DIR / "policy_accuracy_table.tex", "w", encoding="utf-8") as f:
        f.write(policy_tex)

    # 3. Tableau détaillé Descent (4 modèles)
    descent_tex = generate_player_breakdown_table(
        accuracy_data,
        selected_models=DESCENT_MODELS,
        caption="Move Accuracy for Descent Search Variants.",
        label="tab:accuracy_descent",
    )
    with open(OUTPUT_DIR / "descent_accuracy_table.tex", "w", encoding="utf-8") as f:
        f.write(descent_tex)

    # 4. Tableau détaillé MCTS (4 modèles)
    mcts_tex = generate_player_breakdown_table(
        accuracy_data,
        selected_models=MCTS_MODELS,
        caption="Move Accuracy for MCTS Search Variants.",
        label="tab:accuracy_mcts",
    )
    with open(OUTPUT_DIR / "mcts_accuracy_table.tex", "w", encoding="utf-8") as f:
        f.write(mcts_tex)

    print(
        f"Tous les fichiers .tex ont été enregistrés dans le dossier '{OUTPUT_DIR}' avec les meilleures précisions mises en gras."
    )
