"""
LaTeX tabular exporter for generating publication-ready summary and breakdown tables
evaluating Jensen-Shannon Divergence (JSD) metrics across model variants and subject cohorts.
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

# Configuration mapping model identifiers to JSD evaluation dataset paths
RESULTS_CONFIG: dict[str, str] = {
    # Policy Models
    "Maia-2 Baseline": "data/maia_2_baseline_jsd.csv",
    "Maia-2 FT": "data/maia_2_ft_jsd.csv",
    "Maia-2 Nucleus": "data/maia_2_nucleus_jsd.csv",
    "Maia-2 MoE-LoRA": "data/maia_2_moe_lora_jsd.csv",
    # Descent Search Models
    "Maia-2 Descent": "data/maia_2_descent_jsd.csv",
    "Maia-2 N. + Descent": "data/maia_2_n__descent_jsd.csv",
    "Maia-2 FT + N. + Descent": "data/maia_2_ft__n__descent_jsd.csv",
    "Maia-2 MoE-LoRA N. + Descent": ("data/maia_2_moe_lora_n__descent_jsd.csv"),
    # MCTS Search Models
    "Maia-2 MCTS": "data/maia_2_mcts_jsd.csv",
    "Maia-2 N. + MCTS": "data/maia_2_n__mcts_jsd.csv",
    "Maia-2 FT + N. + MCTS": "data/maia_2_ft__n__mcts_jsd.csv",
    "Maia-2 MoE-LoRA N. + MCTS": ("data/maia_2_moe_lora_n__mcts_jsd.csv"),
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


def load_jsd_data_from_csv(
    config: dict[str, str], metric_column: str = "mean_jsd"
) -> dict[str, dict[str, float]]:
    """Load Jensen-Shannon Divergence (JSD) metrics from configured CSV result files.

    Args:
        config (Dict[str, str]): Map of candidate model identifiers to CSV file paths.
        metric_column (str, optional): Target numerical column to extract. Defaults to "mean_jsd".

    Returns:
        Dict[str, Dict[str, float]]: Nested mapping of model names to subject-level JSD values.
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
                    val = float(row[metric_column])
                    data[model_name][player] = val
        else:
            logger.warning(
                "Target evaluation file missing (%s). Column omitted.", path_str
            )
    return data


def generate_summary_table(
    data: dict[str, dict[str, float]],
    caption: str = "Overall Jensen-Shannon Divergence (JSD) summary across model variants.",
    label: str = "tab:jsd_summary",
) -> str:
    """Generate LaTeX tabular source code summarizing global JSD metrics per model variant,
    highlighting optimal (minimum divergence) performance in boldface.

    Args:
        data (Dict[str, Dict[str, float]]): Extracted metric dictionary mapping models to player metrics.
        caption (str, optional): Table caption string. Defaults to global summary caption.
        label (str, optional): Cross-referencing label identifier for LaTeX compilation.
            Defaults to "tab:jsd_summary".

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

    # Jensen-Shannon Divergence is a distance metric: lower values denote superior alignment
    best_mean = min(stats[0] for stats in model_stats.values()) if model_stats else None

    latex: list[str] = []
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
    caption: str = "Detailed Jensen-Shannon Divergence per player.",
    label: str = "tab:jsd_breakdown",
) -> str:
    """Generate detailed per-subject breakdown LaTeX tables, highlighting minimum JSD values
    per row in boldface.

    Args:
        data (Dict[str, Dict[str, float]]): Extracted metric dictionary.
        selected_models (Optional[List[str]], optional): Subset of model identifiers to visualize. Defaults to None.
        caption (str, optional): Table caption string. Defaults to "Detailed Jensen-Shannon Divergence per player.".
        label (str, optional): Table cross-referencing label. Defaults to "tab:jsd_breakdown".

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

    # Evaluate per-subject metrics: minimum divergence signifies optimal distribution alignment
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

    # Evaluate mean metric row: minimum average divergence signifies optimal baseline performance
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

    logger.info("Extracting JSD metrics from result files...")
    jsd_data = load_jsd_data_from_csv(RESULTS_CONFIG, metric_column="mean_jsd")

    # 1. Global summary table construction (all candidate architectures)
    summary_tex = generate_summary_table(jsd_data)
    summary_path = OUTPUT_DIR / "summary_jsd_table.tex"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary_tex)

    # 2. Detailed policy models table construction
    policy_tex = generate_player_breakdown_table(
        jsd_data,
        selected_models=POLICY_MODELS,
        caption="Jensen-Shannon Divergence for Direct Policy Variants.",
        label="tab:jsd_policy",
    )
    policy_path = OUTPUT_DIR / "policy_jsd_table.tex"
    with open(policy_path, "w", encoding="utf-8") as f:
        f.write(policy_tex)

    # 3. Detailed descent models table construction
    descent_tex = generate_player_breakdown_table(
        jsd_data,
        selected_models=DESCENT_MODELS,
        caption="Jensen-Shannon Divergence for Descent Search Variants.",
        label="tab:jsd_descent",
    )
    descent_path = OUTPUT_DIR / "descent_jsd_table.tex"
    with open(descent_path, "w", encoding="utf-8") as f:
        f.write(descent_tex)

    # 4. Detailed MCTS models table construction
    mcts_tex = generate_player_breakdown_table(
        jsd_data,
        selected_models=MCTS_MODELS,
        caption="Jensen-Shannon Divergence for MCTS Search Variants.",
        label="tab:jsd_mcts",
    )
    mcts_path = OUTPUT_DIR / "mcts_jsd_table.tex"
    with open(mcts_path, "w", encoding="utf-8") as f:
        f.write(mcts_tex)

    logger.info(
        "LaTeX JSD tables successfully generated and saved to directory: %s", OUTPUT_DIR
    )
