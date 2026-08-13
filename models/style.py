import logging
import math

import numpy as np
import torch
from cuml.manifold import UMAP as cumlUMAP
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)

# ==========================================
# 1. AUTOENCODER (2304-D -> 128-D)
# ==========================================


class BoardTransitionAutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(2304, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
        )
        self.decoder = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2304),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent


class TransitionDataset(Dataset):
    def __init__(self, vectors: np.ndarray):
        self.vectors = torch.tensor(vectors, dtype=torch.float32)

    def __len__(self):
        return len(self.vectors)

    def __getitem__(self, idx):
        return self.vectors[idx]


def train_autoencoder(
    vectors: np.ndarray,
    epochs: int = 10,
    batch_size: int = 2048,
    lr: float = 1e-3,
    device: torch.device = None,
) -> BoardTransitionAutoEncoder:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = BoardTransitionAutoEncoder().to(device)
    dataset = TransitionDataset(vectors)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch in dataloader:
            batch = batch.to(device)
            optimizer.zero_grad()
            reconstructed, _ = model(batch)
            loss = criterion(reconstructed, batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch)

    model.eval()
    return model


# ==========================================
# 2. DISCRÉTISATION & JSD (Grille 15x15)
# ==========================================


def discretize_to_grid(
    coords_2d: np.ndarray, grid_size: int = 15, bounds: tuple = None
) -> tuple[np.ndarray, tuple]:
    if bounds is None:
        min_x, max_x = coords_2d[:, 0].min(), coords_2d[:, 0].max()
        min_y, max_y = coords_2d[:, 1].min(), coords_2d[:, 1].max()
        bounds = (min_x, max_x, min_y, max_y)
    else:
        min_x, max_x, min_y, max_y = bounds

    x_bins = np.linspace(min_x, max_x, grid_size + 1)
    y_bins = np.linspace(min_y, max_y, grid_size + 1)

    hist, _, _ = np.histogram2d(coords_2d[:, 0], coords_2d[:, 1], bins=[x_bins, y_bins])
    prob_dist = hist.flatten()

    total = prob_dist.sum()
    if total > 0:
        prob_dist = prob_dist / total

    return prob_dist, bounds


def compute_jsd(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    m = 0.5 * (p + q)
    p_c = np.clip(p, eps, 1.0)
    q_c = np.clip(q, eps, 1.0)
    m_c = np.clip(m, eps, 1.0)

    kl_p_m = np.sum(p * np.log2(p_c / m_c))
    kl_q_m = np.sum(q * np.log2(q_c / m_c))

    jsd = 0.5 * kl_p_m + 0.5 * kl_q_m
    return max(0.0, float(jsd))


# ==========================================
# 3. PIPELINE DE STYLOMÉTRIE (AE + UMAP + JSD)
# ==========================================


def evaluate_style_pipeline(
    p_arr: np.ndarray,
    m_arr: np.ndarray,
    device: torch.device = None,
) -> dict[str, float]:
    """
    Prend les vecteurs 2304-D des transitions (joueur et modèle),
    entraîne l'AE, applique UMAP GPU et calcule la JSD sur la grille 15x15.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Entraînement de l'AutoEncoder
    all_vecs = np.vstack([p_arr, m_arr])
    ae_model = train_autoencoder(all_vecs, epochs=10, batch_size=2048, device=device)

    # 2. Inférence Latente (128-D)
    with torch.no_grad():
        p_latent = ae_model.encoder(torch.tensor(p_arr, device=device)).cpu().numpy()
        m_latent = ae_model.encoder(torch.tensor(m_arr, device=device)).cpu().numpy()

    # 3. cuML UMAP GPU (128-D -> 2-D)
    all_latent = np.vstack([p_latent, m_latent])
    cuml_umap = cumlUMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
    all_2d = cuml_umap.fit_transform(all_latent)

    p_2d = all_2d[: len(p_latent)]
    m_2d = all_2d[len(p_latent) :]

    # 4. Discrétisation spatiale 15x15
    p_dist, bounds = discretize_to_grid(p_2d, grid_size=15)
    m_dist, _ = discretize_to_grid(m_2d, grid_size=15, bounds=bounds)

    # 5. Calcul JSD & Distance JS
    jsd_val = compute_jsd(p_dist, m_dist)
    js_dist = math.sqrt(jsd_val)

    return {
        "style_jsd": round(jsd_val, 6),
        "style_jsd_distance": round(js_dist, 6),
    }
