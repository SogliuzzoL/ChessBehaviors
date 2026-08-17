"""Unified LaTeX tabular exporter for the research report.

Generates:
1. Master Overview Table (All 12 models x 4 core metrics).
2. Focused Main-Body Tables (Tactical deep-dive, JSD, Style diagnostic).
3. Appendix Detailed Breakdown Tables (Per-player tables for all metrics).
"""

import csv
import logging
from pathlib import Path
from typing import Callable

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

OUTPUT_DIR_MAIN: Path = Path("tables/main")
OUTPUT_DIR_APPENDIX: Path = Path("tables/appendix")

ALL_MODELS: list[str] = [
    # Direct Policy Variants
    "Maia-2 Baseline",
    "Maia-2 FT",
    "Maia-2 Nucleus",
    "Maia-2 MoE-LoRA",
    # Descent Search Variants
    "Maia-2 Descent",
    "Maia-2 N. + Descent",
    "Maia-2 FT + N. + Descent",
    "Maia-2 MoE-LoRA N. + Descent",
    # MCTS Search Variants
    "Maia-2 MCTS",
    "Maia-2 N. + MCTS",
    "Maia-2 FT + N. + MCTS",
    "Maia-2 MoE-LoRA N. + MCTS",
]

KEY_MODELS: list[str] = [
    "Maia-2 Baseline",
    "Maia-2 FT",
    "Maia-2 MoE-LoRA",
    "Maia-2 N. + MCTS",
    "Maia-2 MoE-LoRA N. + MCTS",
]

CONFIG_ACCURACY: dict[str, str] = {
    "Maia-2 Baseline": "data/maia2_accuracies.csv",
    "Maia-2 FT": "data/maia2_ft_accuracies.csv",
    "Maia-2 Nucleus": "data/maia2_nucleus_accuracies.csv",
    "Maia-2 MoE-LoRA": "data/maia2_moe_lora_accuracies.csv",
    "Maia-2 Descent": "data/maia2_descent_50_accuracies.csv",
    "Maia-2 N. + Descent": "data/maia2_nucleus_descent_50_accuracies.csv",
    "Maia-2 FT + N. + Descent": "data/maia2_ft_nucleus_descent_50_accuracies.csv",
    "Maia-2 MoE-LoRA N. + Descent": (
        "data/maia2_moe_lora_nucleus_descent_50_accuracies.csv"
    ),
    "Maia-2 MCTS": "data/maia2_mcts_accuracies.csv",
    "Maia-2 N. + MCTS": "data/maia2_nucleus_mcts_accuracies.csv",
    "Maia-2 FT + N. + MCTS": "data/maia2_ft_nucleus_mcts_accuracies.csv",
    "Maia-2 MoE-LoRA N. + MCTS": "data/maia2_moe_lora_nucleus_mcts_accuracies.csv",
}

CONFIG_CP_ERROR: dict[str, str] = {
    "Maia-2 Baseline": "data/maia_2_baseline_cp_error.csv",
    "Maia-2 FT": "data/maia_2_ft_cp_error.csv",
    "Maia-2 Nucleus": "data/maia_2_nucleus_cp_error.csv",
    "Maia-2 MoE-LoRA": "data/maia_2_moe_lora_cp_error.csv",
    "Maia-2 Descent": "data/maia_2_descent_cp_error.csv",
    "Maia-2 N. + Descent": "data/maia_2_n__descent_cp_error.csv",
    "Maia-2 FT + N. + Descent": "data/maia_2_ft__n__descent_cp_error.csv",
    "Maia-2 MoE-LoRA N. + Descent": "data/maia_2_moe_lora_n__descent_cp_error.csv",
    "Maia-2 MCTS": "data/maia_2_mcts_cp_error.csv",
    "Maia-2 N. + MCTS": "data/maia_2_n__mcts_cp_error.csv",
    "Maia-2 FT + N. + MCTS": "data/maia_2_ft__n__mcts_cp_error.csv",
    "Maia-2 MoE-LoRA N. + MCTS": "data/maia_2_moe_lora_n__mcts_cp_error.csv",
}

CONFIG_JSD_COMMON: dict[str, str] = {
    "Maia-2 Baseline": "data/maia_2_baseline_common_fens_jsd.csv",
    "Maia-2 FT": "data/maia_2_ft_common_fens_jsd.csv",
    "Maia-2 Nucleus": "data/maia_2_nucleus_common_fens_jsd.csv",
    "Maia-2 MoE-LoRA": "data/maia_2_moe_lora_common_fens_jsd.csv",
    "Maia-2 Descent": "data/maia_2_descent_common_fens_jsd.csv",
    "Maia-2 N. + Descent": "data/maia_2_n__descent_common_fens_jsd.csv",
    "Maia-2 FT + N. + Descent": "data/maia_2_ft__n__descent_common_fens_jsd.csv",
    "Maia-2 MoE-LoRA N. + Descent": (
        "data/maia_2_moe_lora_n__descent_common_fens_jsd.csv"
    ),
    "Maia-2 MCTS": "data/maia_2_mcts_common_fens_jsd.csv",
    "Maia-2 N. + MCTS": "data/maia_2_n__mcts_common_fens_jsd.csv",
    "Maia-2 FT + N. + MCTS": "data/maia_2_ft__n__mcts_common_fens_jsd.csv",
    "Maia-2 MoE-LoRA N. + MCTS": "data/maia_2_moe_lora_n__mcts_common_fens_jsd.csv",
}

CONFIG_STYLE_ALL: dict[str, str] = {
    "Maia-2 Baseline": "data/maia_2_baseline_style.csv",
    "Maia-2 FT": "data/maia_2_ft_style.csv",
    "Maia-2 Nucleus": "data/maia_2_nucleus_style.csv",
    "Maia-2 MoE-LoRA": "data/maia_2_moe_lora_style.csv",
    "Maia-2 Descent": "data/maia_2_descent_style.csv",
    "Maia-2 N. + Descent": "data/maia_2_n__descent_style.csv",
    "Maia-2 FT + N. + Descent": "data/maia_2_ft__n__descent_style.csv",
    "Maia-2 MoE-LoRA N. + Descent": "data/maia_2_moe_lora_n__descent_style.csv",
    "Maia-2 MCTS": "data/maia_2_mcts_style.csv",
    "Maia-2 N. + MCTS": "data/maia_2_n__mcts_style.csv",
    "Maia-2 FT + N. + MCTS": "data/maia_2_ft__n__mcts_style.csv",
    "Maia-2 MoE-LoRA N. + MCTS": "data/maia_2_moe_lora_n__mcts_style.csv",
}

CONFIG_STYLE_COMMON: dict[str, str] = {
    "Maia-2 Baseline": "data/maia_2_baseline_common_fens_style.csv",
    "Maia-2 FT": "data/maia_2_ft_common_fens_style.csv",
    "Maia-2 Nucleus": "data/maia_2_nucleus_common_fens_style.csv",
    "Maia-2 MoE-LoRA": "data/maia_2_moe_lora_common_fens_style.csv",
    "Maia-2 Descent": "data/maia_2_descent_common_fens_style.csv",
    "Maia-2 N. + Descent": "data/maia_2_n__descent_common_fens_style.csv",
    "Maia-2 FT + N. + Descent": "data/maia_2_ft__n__descent_common_fens_style.csv",
    "Maia-2 MoE-LoRA N. + Descent": (
        "data/maia_2_moe_lora_n__descent_common_fens_style.csv"
    ),
    "Maia-2 MCTS": "data/maia_2_mcts_common_fens_style.csv",
    "Maia-2 N. + MCTS": "data/maia_2_n__mcts_common_fens_style.csv",
    "Maia-2 FT + N. + MCTS": "data/maia_2_ft__n__mcts_common_fens_style.csv",
    "Maia-2 MoE-LoRA N. + MCTS": "data/maia_2_moe_lora_n__mcts_common_fens_style.csv",
}


def load_metric_data(
    config: dict[str, str], metric_col: str
) -> dict[str, dict[str, float]]:
    data: dict[str, dict[str, float]] = {}
    for model_name, path_str in config.items():
        path = Path(path_str)
        data[model_name] = {}
        if path.exists():
            with open(path, mode="r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    player = row["player_name"].strip()
                    val = float(row[metric_col])
                    data[model_name][player] = val
        else:
            logger.warning("Missing file: %s for %s", path_str, model_name)
    return data


def compute_stats(player_dict: dict[str, float]) -> tuple[float, float] | None:
    valid_vals = [
        v
        for k, v in player_dict.items()
        if k.lower() not in ["average", "mean", "avg"] and isinstance(v, (int, float))
    ]
    if not valid_vals:
        return None
    mean_val = sum(valid_vals) / len(valid_vals)
    variance = sum((x - mean_val) ** 2 for x in valid_vals) / len(valid_vals)
    return mean_val, variance**0.5


# =========================================================================
# 1. MASTER OVERVIEW TABLE (ALL 12 MODELS)
# =========================================================================


def generate_master_summary_table(
    acc_data: dict[str, dict[str, float]],
    cp_data: dict[str, dict[str, float]],
    jsd_com_data: dict[str, dict[str, float]],
    style_com_data: dict[str, dict[str, float]],
    models: list[str] = ALL_MODELS,
) -> str:
    """Generate comprehensive summary table comparing all 12 model variants across all 4 key metrics."""
    acc_stats = {m: compute_stats(acc_data.get(m, {})) for m in models}
    cp_stats = {m: compute_stats(cp_data.get(m, {})) for m in models}
    jsd_stats = {m: compute_stats(jsd_com_data.get(m, {})) for m in models}
    style_stats = {m: compute_stats(style_com_data.get(m, {})) for m in models}

    valid_acc = [s[0] for s in acc_stats.values() if s]
    valid_cp = [s[0] for s in cp_stats.values() if s]
    valid_jsd = [s[0] for s in jsd_stats.values() if s]
    valid_style = [s[0] for s in style_stats.values() if s]

    best_acc = max(valid_acc) if valid_acc else None
    best_cp = min(valid_cp) if valid_cp else None
    best_jsd = min(valid_jsd) if valid_jsd else None
    best_style = min(valid_style) if valid_style else None

    latex = [
        "\\begin{table}[!htbp]",
        "  \\centering",
        "  \\caption{Comprehensive performance summary across all model variants and evaluation dimensions.}",
        "  \\label{tab:master_summary}",
        "  \\resizebox{\\linewidth}{!}{%",
        "    \\begin{tabular}{lcccc}",
        "      \\toprule",
        "      \\textbf{Model Variant} & \\textbf{Move Acc (\\%)} & \\textbf{CP Error} & \\textbf{Common JSD} & \\textbf{Common Style JSD} \\\\",
        "      \\midrule",
        "      \\multicolumn{5}{l}{\\textit{Direct Policy Variants}} \\\\[2pt]",
    ]

    for idx, m in enumerate(models):
        if idx == 4:
            latex.append(
                "      \\midrule\n      \\multicolumn{5}{l}{\\textit{Descent Search Variants}} \\\\[2pt]"
            )
        elif idx == 8:
            latex.append(
                "      \\midrule\n      \\multicolumn{5}{l}{\\textit{MCTS Search Variants}} \\\\[2pt]"
            )

        acc_s, cp_s, jsd_s, st_s = (
            acc_stats.get(m),
            cp_stats.get(m),
            jsd_stats.get(m),
            style_stats.get(m),
        )

        acc_str = f"{acc_s[0] * 100:.2f}\\%" if acc_s else "--"
        if acc_s and best_acc and abs(acc_s[0] - best_acc) < 1e-7:
            acc_str = f"\\textbf{{{acc_str}}}"

        cp_str = f"{cp_s[0]:.2f}" if cp_s else "--"
        if cp_s and best_cp and abs(cp_s[0] - best_cp) < 1e-7:
            cp_str = f"\\textbf{{{cp_str}}}"

        jsd_str = f"{jsd_s[0]:.4f}" if jsd_s else "--"
        if jsd_s and best_jsd and abs(jsd_s[0] - best_jsd) < 1e-7:
            jsd_str = f"\\textbf{{{jsd_str}}}"

        st_str = f"{st_s[0]:.4f}" if st_s else "--"
        if st_s and best_style and abs(st_s[0] - best_style) < 1e-7:
            st_str = f"\\textbf{{{st_str}}}"

        latex.append(f"      {m} & {acc_str} & {cp_str} & {jsd_str} & {st_str} \\\\")

    latex.extend(
        [
            "      \\bottomrule",
            "    \\end{tabular}%",
            "  }",
            "\\end{table}",
        ]
    )
    return "\n".join(latex)


# =========================================================================
# 2. FOCUSED MAIN REPORT TABLES
# =========================================================================


def generate_focused_tactical_table(
    acc_data: dict[str, dict[str, float]],
    cp_data: dict[str, dict[str, float]],
    models: list[str] = KEY_MODELS,
) -> str:
    acc_stats = {m: compute_stats(acc_data.get(m, {})) for m in models}
    cp_stats = {m: compute_stats(cp_data.get(m, {})) for m in models}

    valid_acc = [s[0] for s in acc_stats.values() if s]
    valid_cp = [s[0] for s in cp_stats.values() if s]
    best_acc = max(valid_acc) if valid_acc else None
    best_cp = min(valid_cp) if valid_cp else None

    latex = [
        "\\begin{table}[!htbp]",
        "  \\centering",
        "  \\caption{Move Accuracy and Tactical Centipawn Error for core model configurations.}",
        "  \\label{tab:focused_tactical_summary}",
        "  \\begin{tabular}{lcccc}",
        "    \\toprule",
        "    \\textbf{Model Configuration} & \\textbf{Move Acc (\\%)} & \\textbf{Std Dev} & \\textbf{CP Error} & \\textbf{Std Dev} \\\\",
        "    \\midrule",
    ]

    for m in models:
        acc_s = acc_stats.get(m)
        cp_s = cp_stats.get(m)

        if acc_s:
            acc_str = f"{acc_s[0] * 100:.2f}\\%"
            acc_sd = f"$\\pm$ {acc_s[1] * 100:.2f}\\%"
            if best_acc and abs(acc_s[0] - best_acc) < 1e-7:
                acc_str = f"\\textbf{{{acc_str}}}"
                acc_sd = f"\\textbf{{{acc_sd}}}"
        else:
            acc_str, acc_sd = "--", "--"

        if cp_s:
            cp_str = f"{cp_s[0]:.2f}"
            cp_sd = f"$\\pm$ {cp_s[1]:.2f}"
            if best_cp and abs(cp_s[0] - best_cp) < 1e-7:
                cp_str = f"\\textbf{{{cp_str}}}"
                cp_sd = f"\\textbf{{{cp_sd}}}"
        else:
            cp_str, cp_sd = "--", "--"

        latex.append(f"    {m} & {acc_str} & {acc_sd} & {cp_str} & {cp_sd} \\\\")

    latex.extend(
        [
            "    \\bottomrule",
            "  \\end{tabular}",
            "\\end{table}",
        ]
    )
    return "\n".join(latex)


def generate_focused_style_bias_table(
    style_all_data: dict[str, dict[str, float]],
    style_com_data: dict[str, dict[str, float]],
    models: list[str] = KEY_MODELS,
) -> str:
    all_stats = {m: compute_stats(style_all_data.get(m, {})) for m in models}
    com_stats = {m: compute_stats(style_com_data.get(m, {})) for m in models}

    latex = [
        "\\begin{table}[!htbp]",
        "  \\centering",
        "  \\caption{Stylistic JSD diagnostic across Full Dataset vs. Common FEN positions.}",
        "  \\label{tab:focused_style_bias}",
        "  \\begin{tabular}{lcccc}",
        "    \\toprule",
        "    \\textbf{Model Configuration} & \\textbf{Full Dataset} & \\textbf{Std Dev} & \\textbf{Common FENs} & \\textbf{Std Dev} \\\\",
        "    \\midrule",
    ]

    for m in models:
        all_s = all_stats.get(m)
        com_s = com_stats.get(m)

        all_str = f"{all_s[0]:.4f} & $\\pm$ {all_s[1]:.4f}" if all_s else "-- & --"
        com_str = f"{com_s[0]:.4f} & $\\pm$ {com_s[1]:.4f}" if com_s else "-- & --"

        latex.append(f"    {m} & {all_str} & {com_str} \\\\")

    latex.extend(
        [
            "    \\bottomrule",
            "  \\end{tabular}",
            "\\end{table}",
        ]
    )
    return "\n".join(latex)


# =========================================================================
# 3. APPENDIX PLAYER BREAKDOWNS
# =========================================================================


def generate_appendix_breakdown_table(
    data: dict[str, dict[str, float]],
    selected_models: list[str],
    caption: str,
    label: str,
    format_fn: Callable[[float], str] = lambda x: f"{x:.4f}",
    is_percentage: bool = False,
    lower_is_better: bool = True,
) -> str:
    models = [m for m in selected_models if m in data]
    players = sorted(
        {
            p
            for m in models
            for p in data[m].keys()
            if p.lower() not in ["average", "mean", "avg"]
        }
    )

    col_spec = "l" + "c" * len(models)
    latex = [
        "\\begin{table}[!htbp]",
        "  \\centering",
        f"  \\caption{{{caption}}}",
        f"  \\label{{{label}}}",
        "  \\resizebox{\\linewidth}{!}{%",
        f"    \\begin{{tabular}}{{{col_spec}}}",
        "      \\toprule",
    ]

    header = "      \\textbf{Player}"
    for m in models:
        header += f" & \\textbf{{{m}}}"
    header += " \\\\"
    latex.append(header)
    latex.append("      \\midrule")

    for p in players:
        p_vals = {m: data[m].get(p) for m in models if data[m].get(p) is not None}
        best_val = (
            (min(p_vals.values()) if lower_is_better else max(p_vals.values()))
            if p_vals
            else None
        )

        row = f"      {p}"
        for m in models:
            val = data[m].get(p)
            if val is not None:
                val_num = val * 100 if is_percentage else val
                val_str = format_fn(val_num)
                if best_val is not None and abs(val - best_val) < 1e-7:
                    val_str = f"\\textbf{{{val_str}}}"
                row += f" & {val_str}"
            else:
                row += " & --"
        row += " \\\\"
        latex.append(row)

    latex.append("      \\midrule")

    avg_vals: dict[str, float] = {}
    for m in models:
        s = compute_stats(data[m])
        if s:
            avg_vals[m] = s[0]

    best_avg = (
        (min(avg_vals.values()) if lower_is_better else max(avg_vals.values()))
        if avg_vals
        else None
    )

    avg_row = "      \\textbf{Average}"
    for m in models:
        if m in avg_vals:
            avg_val_num = avg_vals[m] * 100 if is_percentage else avg_vals[m]
            val_str = format_fn(avg_val_num)
            if best_avg is not None and abs(avg_vals[m] - best_avg) < 1e-7:
                val_str = f"\\textbf{{{val_str}}}"
            avg_row += f" & {val_str}"
        else:
            avg_row += " & --"
    avg_row += " \\\\"
    latex.append(avg_row)

    latex.extend(
        [
            "      \\bottomrule",
            "    \\end{tabular}%",
            "  }",
            "\\end{table}",
        ]
    )
    return "\n".join(latex)


if __name__ == "__main__":
    OUTPUT_DIR_MAIN.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR_APPENDIX.mkdir(parents=True, exist_ok=True)

    logger.info("Extracting metric datasets from CSV files...")
    acc_data = load_metric_data(CONFIG_ACCURACY, "accuracy")
    cp_data = load_metric_data(CONFIG_CP_ERROR, "cp_error")
    jsd_com_data = load_metric_data(CONFIG_JSD_COMMON, "mean_jsd")
    style_all_data = load_metric_data(CONFIG_STYLE_ALL, "style_jsd")
    style_com_data = load_metric_data(CONFIG_STYLE_COMMON, "style_jsd")

    # 1. Master Table Export
    logger.info("Generating Master Overview table...")
    master_path = OUTPUT_DIR_MAIN / "tab_master_summary.tex"
    with open(master_path, "w", encoding="utf-8") as f:
        f.write(
            generate_master_summary_table(
                acc_data, cp_data, jsd_com_data, style_com_data
            )
        )

    # 2. Focused Tables Export
    logger.info("Generating Focused Main Report tables...")
    foc_tactical_path = OUTPUT_DIR_MAIN / "tab_focused_tactical_summary.tex"
    with open(foc_tactical_path, "w", encoding="utf-8") as f:
        f.write(generate_focused_tactical_table(acc_data, cp_data))

    foc_style_path = OUTPUT_DIR_MAIN / "tab_focused_style_bias.tex"
    with open(foc_style_path, "w", encoding="utf-8") as f:
        f.write(generate_focused_style_bias_table(style_all_data, style_com_data))

    # 3. Appendix Tables Export
    logger.info("Generating Appendix Breakdown tables...")
    app_acc_path = OUTPUT_DIR_APPENDIX / "tab_appendix_accuracy_all.tex"
    with open(app_acc_path, "w", encoding="utf-8") as f:
        f.write(
            generate_appendix_breakdown_table(
                acc_data,
                ALL_MODELS,
                caption="Complete Move Accuracy breakdown per player across all model variants.",
                label="tab:app_accuracy_all",
                format_fn=lambda x: f"{x:.2f}\\%",
                is_percentage=True,
                lower_is_better=False,
            )
        )

    app_cp_path = OUTPUT_DIR_APPENDIX / "tab_appendix_cp_error_all.tex"
    with open(app_cp_path, "w", encoding="utf-8") as f:
        f.write(
            generate_appendix_breakdown_table(
                cp_data,
                ALL_MODELS,
                caption="Complete Centipawn Error breakdown per player across all model variants.",
                label="tab:app_cp_error_all",
                format_fn=lambda x: f"{x:.2f}",
                is_percentage=False,
                lower_is_better=True,
            )
        )

    app_jsd_path = OUTPUT_DIR_APPENDIX / "tab_appendix_common_fens_jsd_all.tex"
    with open(app_jsd_path, "w", encoding="utf-8") as f:
        f.write(
            generate_appendix_breakdown_table(
                jsd_com_data,
                ALL_MODELS,
                caption="Complete JSD breakdown evaluated on Common FEN positions.",
                label="tab:app_jsd_common_all",
                format_fn=lambda x: f"{x:.4f}",
                is_percentage=False,
                lower_is_better=True,
            )
        )

    app_style_path = OUTPUT_DIR_APPENDIX / "tab_appendix_common_fens_style_all.tex"
    with open(app_style_path, "w", encoding="utf-8") as f:
        f.write(
            generate_appendix_breakdown_table(
                style_com_data,
                ALL_MODELS,
                caption="Complete Stylistic JSD breakdown evaluated on Common FEN positions.",
                label="tab:app_style_common_all",
                format_fn=lambda x: f"{x:.4f}",
                is_percentage=False,
                lower_is_better=True,
            )
        )

    logger.info("All LaTeX tables generated successfully.")
