"""
LaTeX tabular exporter for generating publication-ready summary and breakdown tables
evaluating move prediction accuracy metrics across model variants and subject cohorts.
"""

import csv
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Output directory specification for generated LaTeX tables
OUTPUT_DIR: Path = Path("tables")

# Configuration mapping model identifiers to accuracy evaluation dataset paths
RESULTS_CONFIG: dict[str, str] = {
    # Policy Models
    "Maia-2 Baseline": "data/maia2_accuracies.csv",
    "Maia-2 FT": "data/maia2_ft_accuracies.csv",
    "Maia-2 Nucleus": "data/maia2_nucleus_accuracies.csv",
    "Maia-2 MoE-LoRA": "data/maia2_moe_lora_accuracies.csv",
    # Descent Search Models
    "Maia-2 Descent": "data/maia2_descent_50_accuracies.csv",
    "Maia-2 N. + Descent": "data/maia2_nucleus_descent_50_accuracies.csv",
    "Maia-2 FT + N. + Descent": "data/maia2_ft_nucleus_descent_50_accuracies.csv",
    "Maia-2 MoE-LoRA N. + Descent": (
        "data/maia2_moe_lora_nucleus_descent_50_accuracies.csv"
    ),
    # MCTS Search Models
    "Maia-2 MCTS": "data/maia2_mcts_accuracies.csv",
    "Maia-2 N. + MCTS": "data/maia2_nucleus_mcts_accuracies.csv",
    "Maia-2 FT + N. + MCTS": "data/maia2_ft_nucleus_mcts_accuracies.csv",
    "Maia-2 MoE-LoRA N. + MCTS": ("data/maia2_moe_lora_nucleus_mcts_accuracies.csv"),
}

# Logical model groupings by algorithmic paradigm for table partitioning
POLICY_MODELS: list[str] = [
    "Maia-2 Baseline",
    "Maia-2 FT",
    "Maia-2 Nucleus",
    "Maia-2 MoE-LoRA",
]

DESCENT_MODELS: list[str] = [
    "Maia-2 Descent",
    "Maia-2 N. + Descent",
    "Maia-2 FT + N. + Descent",
    "Maia-2 MoE-LoRA N. + Descent",
]

MCTS_MODELS: list[str] = [
    "Maia-2 MCTS",
    "Maia-2 N. + MCTS",
    "Maia-2 FT + N. + MCTS",
    "Maia-2 MoE-LoRA N. + MCTS",
]


def load_accuracy_data_from_csv(
    config: dict[str, str],
) -> dict[str, dict[str, float]]:
    """Load move accuracy evaluation metrics from configured CSV result files.

    Args:
        config (Dict[str, str]): Map of candidate model identifiers to CSV file paths.

    Returns:
        Dict[str, Dict[str, float]]: Nested mapping of model names to subject-level accuracy values.
    """
    data: dict[str, dict[str, float]] = {}
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
            logger.warning(
                "Target evaluation file missing (%s). Column omitted.", path_str
            )
    return data


def generate_summary_table(
    data: dict[str, dict[str, float]],
    caption: str = "Overall Move Accuracy summary across model variants.",
    label: str = "tab:accuracy_summary",
) -> str:
    """Generate LaTeX tabular source code summarizing global move accuracy metrics per model variant,
    highlighting optimal (maximum accuracy) performance in boldface.

    Args:
        data (Dict[str, Dict[str, float]]): Extracted metric dictionary mapping models to player metrics.
        caption (str, optional): Table caption string. Defaults to global summary caption.
        label (str, optional): Cross-referencing label identifier for LaTeX compilation.
            Defaults to "tab:accuracy_summary".

    Returns:
        str: Formatted LaTeX source code representing the output summary table.
    """
    model_stats: dict[str, tuple[float, float]] = {}
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

    # Accuracy is a performance metric: higher values denote superior accuracy
    best_mean = max(stats[0] for stats in model_stats.values()) if model_stats else None

    latex: list[str] = []
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

    for model_name in data:
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
    selected_models: list[str] | None = None,
    caption: str = "Detailed move accuracy per player.",
    label: str = "tab:move_accuracy_breakdown",
) -> str:
    """Generate detailed per-subject breakdown LaTeX tables, highlighting maximum accuracy values
    per row in boldface.

    Args:
        data (Dict[str, Dict[str, float]]): Extracted metric dictionary.
        selected_models (Optional[List[str]], optional): Subset of model identifiers to visualize. Defaults to None.
        caption (str, optional): Table caption string. Defaults to "Detailed move accuracy per player.".
        label (str, optional): Table cross-referencing label. Defaults to "tab:move_accuracy_breakdown".

    Returns:
        str: Formatted LaTeX source code string representing the breakdown matrix.
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

    latex: list[str] = []
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

    # Evaluate per-subject metrics: maximum accuracy signifies optimal performance
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

    # Evaluate mean metric row: maximum average accuracy signifies optimal baseline performance
    avg_vals: dict[str, float] = {}
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

    logger.info("Extracting accuracy metrics from result files...")
    accuracy_data = load_accuracy_data_from_csv(RESULTS_CONFIG)

    # 1. Global summary table construction (all candidate architectures)
    summary_tex = generate_summary_table(accuracy_data)
    summary_path = OUTPUT_DIR / "summary_accuracy_table.tex"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary_tex)

    # 2. Detailed policy models table construction
    policy_tex = generate_player_breakdown_table(
        accuracy_data,
        selected_models=POLICY_MODELS,
        caption="Move Accuracy for Direct Policy Variants.",
        label="tab:accuracy_policy",
    )
    policy_path = OUTPUT_DIR / "policy_accuracy_table.tex"
    with open(policy_path, "w", encoding="utf-8") as f:
        f.write(policy_tex)

    # 3. Detailed descent models table construction
    descent_tex = generate_player_breakdown_table(
        accuracy_data,
        selected_models=DESCENT_MODELS,
        caption="Move Accuracy for Descent Search Variants.",
        label="tab:accuracy_descent",
    )
    descent_path = OUTPUT_DIR / "descent_accuracy_table.tex"
    with open(descent_path, "w", encoding="utf-8") as f:
        f.write(descent_path)

    # 4. Detailed MCTS models table construction
    mcts_tex = generate_player_breakdown_table(
        accuracy_data,
        selected_models=MCTS_MODELS,
        caption="Move Accuracy for MCTS Search Variants.",
        label="tab:accuracy_mcts",
    )
    mcts_path = OUTPUT_DIR / "mcts_accuracy_table.tex"
    with open(mcts_path, "w", encoding="utf-8") as f:
        f.write(mcts_tex)

    logger.info(
        "LaTeX Accuracy tables successfully generated and saved to directory: %s",
        OUTPUT_DIR,
    )
