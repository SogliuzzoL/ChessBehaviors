"""
LaTeX table generator script for evaluating stylistic alignment across model variants
using the AE + cuML UMAP + Spatial Grid JSD pipeline[cite: 1, 2].
"""

import csv
from pathlib import Path

# Configuration for output directory
OUTPUT_DIR = Path("tables")

# Mapping model variant names to their respective stylistic result CSV files
RESULTS_CONFIG = {
    # Policy Models
    "Maia-2 Baseline": "data/maia_2_baseline_style.csv",
    "Maia-2 FT": "data/maia_2_ft_style.csv",
    "Maia-2 Nucleus": "data/maia_2_nucleus_style.csv",
    "Maia-2 MoE-LoRA": "data/maia_2_moe_lora_style.csv",
    # Descent Models
    "Maia-2 Descent": "data/maia_2_descent_style.csv",
    "Maia-2 N. + Descent": "data/maia_2_n__descent_style.csv",
    "Maia-2 FT + N. + Descent": "data/maia_2_ft__n__descent_style.csv",
    "Maia-2 MoE-LoRA N. + Descent": "data/maia_2_moe_lora_n__descent_style.csv",
    # MCTS Models
    "Maia-2 MCTS": "data/maia_2_mcts_style.csv",
    "Maia-2 N. + MCTS": "data/maia_2_n__mcts_style.csv",
    "Maia-2 FT + N. + MCTS": "data/maia_2_ft__n__mcts_style.csv",
    "Maia-2 MoE-LoRA N. + MCTS": "data/maia_2_moe_lora_n__mcts_style.csv",
}

# Logical grouping per paradigm
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


def load_style_data_from_csv(
    config: dict[str, str], metric_column: str = "style_jsd"
) -> dict[str, dict[str, float]]:
    """
    Loads stylistic evaluation metrics from configured CSV files.
    """
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
            print(f"File not found: {path_str}. Skipping entries for {model_name}.")
    return data


def generate_summary_table(
    data: dict[str, dict[str, float]],
    caption: str = "Overall Stylistic Jensen-Shannon Divergence (Style JSD) summary across model variants.",
    label: str = "tab:style_jsd_summary",
) -> str:
    """
    Generates a compact summary LaTeX table highlighting the lowest (best) mean divergence.
    """
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

    # Lowest divergence indicates superior behavioral alignment
    best_mean = min(stats[0] for stats in model_stats.values()) if model_stats else None

    latex = []
    latex.append("\\begin{table}[!htbp]")
    latex.append("  \\centering")
    latex.append(f"  \\caption{{{caption}}}")
    latex.append(f"  \\label{{{label}}}")
    latex.append("  \\begin{tabular}{lcc}")
    latex.append("    \\toprule")
    latex.append(
        "    \\textbf{Model Variant} & \\textbf{Mean Style JSD} & \\textbf{Std Dev} \\\\"
    )
    latex.append("    \\midrule")

    for model_name in data:
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
    selected_models: list[str] | None = None,
    caption: str = "Detailed Stylistic Jensen-Shannon Divergence per player.",
    label: str = "tab:style_jsd_breakdown",
) -> str:
    """
    Generates a detailed per-player breakdown LaTeX table bolding the best (lowest) divergence per row.
    """
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
        player_vals = {
            model: data[model].get(player)
            for model in models
            if data[model].get(player) is not None
        }
        min_val = min(player_vals.values()) if player_vals else None

        row = f"      {player}"
        for model in models:
            val = data[model].get(player, None)
            if val is not None:
                val_str = f"{val:.4f}"
                if min_val is not None and abs(val - min_val) < 1e-7:
                    val_str = f"\\textbf{{{val_str}}}"
                row += f" & {val_str}"
            else:
                row += " & --"
        row += " \\\\"
        latex.append(row)

    latex.append("      \\midrule")

    # Compute average row across valid players
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

    # Load stylistic divergence data (use metric_column="style_jsd_distance" if sqrt-JSD distance is preferred)
    style_data = load_style_data_from_csv(RESULTS_CONFIG, metric_column="style_jsd")

    # 1. Overall Summary Table
    summary_tex = generate_summary_table(style_data)
    with open(OUTPUT_DIR / "summary_style_table.tex", "w", encoding="utf-8") as f:
        f.write(summary_tex)

    # 2. Detailed Direct Policy Variants Table
    policy_tex = generate_player_breakdown_table(
        style_data,
        selected_models=POLICY_MODELS,
        caption="Stylistic Jensen-Shannon Divergence for Direct Policy Variants.",
        label="tab:style_jsd_policy",
    )
    with open(OUTPUT_DIR / "policy_style_table.tex", "w", encoding="utf-8") as f:
        f.write(policy_tex)

    # 3. Detailed Descent Search Variants Table
    descent_tex = generate_player_breakdown_table(
        style_data,
        selected_models=DESCENT_MODELS,
        caption="Stylistic Jensen-Shannon Divergence for Descent Search Variants.",
        label="tab:style_jsd_descent",
    )
    with open(OUTPUT_DIR / "descent_style_table.tex", "w", encoding="utf-8") as f:
        f.write(descent_tex)

    # 4. Detailed MCTS Search Variants Table
    mcts_tex = generate_player_breakdown_table(
        style_data,
        selected_models=MCTS_MODELS,
        caption="Stylistic Jensen-Shannon Divergence for MCTS Search Variants.",
        label="tab:style_jsd_mcts",
    )
    with open(OUTPUT_DIR / "mcts_style_table.tex", "w", encoding="utf-8") as f:
        f.write(mcts_tex)

    print(
        f"All stylistic LaTeX tables were successfully generated in the '{OUTPUT_DIR}' directory."
    )
