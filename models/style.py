"""
Core architectural pipeline for behavioral stylometry evaluation.
Includes the AutoEncoder architecture, spatial grid discretization,
Jensen-Shannon Divergence computation, and the global reference space module.
"""

import logging
import math

import numpy as np
import torch
from cuml.manifold import UMAP as cumlUMAP
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


# ==========================================
# 1. AUTOENCODER ARCHITECTURE (2304-D -> 128-D)
# ==========================================


class BoardTransitionAutoEncoder(nn.Module):
    """
    Symmetric fully-connected AutoEncoder designed to compress 2304-dimensional
    board transition vectors into a dense 128-dimensional latent representation.
    """

    def __init__(self) -> None:
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
        """
        Forward pass yielding reconstructed output and intermediate latent codes.
        """
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent


class TransitionDataset(Dataset):
    """PyTorch Dataset wrapper for board transition feature matrices."""

    def __init__(self, vectors: np.ndarray) -> None:
        self.vectors = torch.tensor(vectors, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.vectors)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.vectors[idx]


def train_autoencoder(
    vectors: np.ndarray,
    epochs: int = 10,
    batch_size: int = 2048,
    lr: float = 1e-3,
    device: torch.device | None = None,
) -> BoardTransitionAutoEncoder:
    """
    Trains the transition AutoEncoder using Mean Squared Error loss and the Adam optimizer.
    """
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
# 2. GRID DISCRETIZATION & JSD METRICS
# ==========================================


def discretize_to_grid(
    coords_2d: np.ndarray,
    grid_size: int = 15,
    bounds: tuple[float, float, float, float] | None = None,
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    """
    Projects 2D continuous manifold coordinates onto a discrete spatial grid
    to yield empirical probability distributions over discrete bins.
    """
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
    """
    Computes the symmetric Jensen-Shannon Divergence between two discrete probability distributions.
    """
    m = 0.5 * (p + q)
    p_c = np.clip(p, eps, 1.0)
    q_c = np.clip(q, eps, 1.0)
    m_c = np.clip(m, eps, 1.0)

    kl_p_m = np.sum(p * np.log2(p_c / m_c))
    kl_q_m = np.sum(q * np.log2(q_c / m_c))

    jsd = 0.5 * kl_p_m + 0.5 * kl_q_m
    return max(0.0, float(jsd))


# ==========================================
# 3. GLOBAL STYLE SPACE REFERENCE MODULE
# ==========================================


class GlobalStyleSpace:
    """
    Encapsulates a unified, pre-trained latent space (AutoEncoder + cuML UMAP) fitted
    exclusively on ground-truth human player transitions to guarantee an unbiased,
    invariant spatial evaluation manifold.
    """

    def __init__(
        self,
        ae_model: BoardTransitionAutoEncoder,
        cuml_umap: cumlUMAP,
        device: torch.device,
    ) -> None:
        self.ae_model = ae_model
        self.cuml_umap = cuml_umap
        self.device = device

    @classmethod
    def fit_from_vectors(
        cls,
        reference_vectors: np.ndarray,
        ae_epochs: int = 10,
        device: torch.device | None = None,
        seed: int = 42,
    ) -> "GlobalStyleSpace":
        """
        Fits the global AutoEncoder and UMAP manifold using ground-truth human transitions.
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logger.info(
            f"Fitting Global AutoEncoder on {len(reference_vectors)} reference human transitions..."
        )
        ae_model = train_autoencoder(
            reference_vectors, epochs=ae_epochs, batch_size=2048, device=device
        )

        logger.info("Extracting global 128-D latent embeddings...")
        with torch.no_grad():
            ref_tensor = torch.tensor(reference_vectors, dtype=torch.float32).to(device)
            _, latent_ref = ae_model(ref_tensor)
            latent_ref = latent_ref.cpu().numpy()

        logger.info("Fitting GPU-accelerated cuML UMAP on reference latent manifold...")
        cuml_umap = cumlUMAP(
            n_neighbors=15, min_dist=0.1, n_components=2, random_state=seed
        )
        cuml_umap.fit(latent_ref)

        return cls(ae_model=ae_model, cuml_umap=cuml_umap, device=device)

    def project_to_2d(self, vectors: np.ndarray) -> np.ndarray:
        """
        Projects arbitrary 2304-D transition vectors onto the established 2D reference manifold.
        """
        with torch.no_grad():
            vec_tensor = torch.tensor(vectors, dtype=torch.float32).to(self.device)
            _, latent = self.ae_model(vec_tensor)
            latent_np = latent.cpu().numpy()

        coords_2d = self.cuml_umap.transform(latent_np)
        return coords_2d


def evaluate_style_with_space(
    global_space: GlobalStyleSpace,
    p_arr: np.ndarray,
    m_arr: np.ndarray,
) -> dict[str, float]:
    """
    Evaluates stylistic alignment by projecting target human transitions and model-predicted
    transitions into the shared global reference space and computing discrete JSD.
    """
    # Project transitions into the fixed 2D spatial manifold
    p_2d = global_space.project_to_2d(p_arr)
    m_2d = global_space.project_to_2d(m_arr)

    # Discretize projections onto a 15x15 spatial grid using player coordinate bounds
    p_dist, bounds = discretize_to_grid(p_2d, grid_size=15)
    m_dist, _ = discretize_to_grid(m_2d, grid_size=15, bounds=bounds)

    # Calculate Jensen-Shannon Divergence and metric distance
    jsd_val = compute_jsd(p_dist, m_dist)
    js_dist = math.sqrt(jsd_val)

    return {
        "style_jsd": round(jsd_val, 6),
        "style_jsd_distance": round(js_dist, 6),
    }
