import os

import matplotlib.pyplot as plt
import pandas as pd

# ==========================================
# 🎯 PARAMÈTRE : Chemin vers ton fichier CSV
# ==========================================
FILE_PATH = "data/training_logs_ft_player_0.csv"


def plot_training_logs(file_path: str):
    if not os.path.exists(file_path):
        print(f"❌ Fichier introuvable : {file_path}")
        return

    df = pd.read_csv(file_path)

    batch_df = df[df["type"] == "batch"].copy()
    epoch_df = df[df["type"] == "epoch"].copy()

    if batch_df.empty or epoch_df.empty:
        print("❌ Le fichier ne contient pas les entrées 'batch' et 'epoch' requises.")
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=False)

    # 1. Graphique du haut : Train Loss par Batch & Moyenne Mobile
    ax1.set_xlabel("Global Batch Step")
    ax1.set_ylabel("Loss")
    ax1.plot(
        batch_df["global_step"],
        batch_df["train_loss"],
        alpha=0.25,
        label="Batch Train Loss",
        color="mediumpurple",
    )
    ax1.plot(
        batch_df["global_step"],
        batch_df["train_loss"].rolling(100).mean(),
        label="Train Loss (Moyenne mobile w=100)",
        color="purple",
        linewidth=1.8,
    )

    # Lignes verticales pour marquer les fins d'époques
    epoch_steps = batch_df.groupby("epoch")["global_step"].max().values
    for ep_step in epoch_steps[:-1]:
        ax1.axvline(x=ep_step, color="red", linestyle="--", alpha=0.5)

    player_idx = df["player_index"].iloc[0]
    ax1.set_title(f"Suivi de l'Entraînement - Joueur {player_idx}")
    ax1.legend(loc="upper right")

    # 2. Graphique du bas : Pertes Moyennes et Accuracy de Validation par Époque
    ax2.plot(
        epoch_df["epoch"],
        epoch_df["avg_train_loss"],
        marker="o",
        label="Avg Train Loss",
        color="purple",
        linewidth=2,
    )
    if "val_loss" in epoch_df.columns and epoch_df["val_loss"].notna().any():
        ax2.plot(
            epoch_df["epoch"],
            epoch_df["val_loss"],
            marker="s",
            label="Val Loss",
            color="darkorange",
            linewidth=2,
        )

    ax2.set_xlabel("Époque")
    ax2.set_ylabel("Loss")
    ax2.legend(loc="upper left")

    # Axe secondaire pour l'Accuracy de Validation
    if "val_acc" in epoch_df.columns and epoch_df["val_acc"].notna().any():
        ax3 = ax2.twinx()
        ax3.plot(
            epoch_df["epoch"],
            epoch_df["val_acc"] * 100,
            marker="^",
            label="Val Accuracy (%)",
            color="forestgreen",
            linewidth=2,
        )
        ax3.set_ylabel("Val Accuracy (%)", color="forestgreen")
        ax3.tick_params(axis="y", labelcolor="forestgreen")

        # Annotation du pourcentage au-dessus de chaque point
        for _, row in epoch_df.iterrows():
            if pd.notna(row["val_acc"]):
                ax3.annotate(
                    f"{row['val_acc'] * 100:.2f}%",
                    (row["epoch"], row["val_acc"] * 100),
                    textcoords="offset points",
                    xytext=(0, 8),
                    ha="center",
                    color="forestgreen",
                    fontweight="bold",
                    fontsize=9,
                )

    plt.title("Performances par Époque (Train vs Validation)")
    plt.tight_layout()

    # Sauvegarde de l'image
    output_filename = file_path.replace(".csv", "_analysis.pdf")
    plt.savefig(output_filename, dpi=300)
    print(f"✅ Graphique généré et sauvegardé sous : {output_filename}")
    plt.show()


if __name__ == "__main__":
    plot_training_logs(FILE_PATH)
