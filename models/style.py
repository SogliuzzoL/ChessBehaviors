"""
Core architectural pipeline for behavioral stylometry evaluation.
Includes the AutoEncoder architecture, spatial grid discretization,
Jensen-Shannon Divergence computation, and the global reference space module.
"""

import gc
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
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent


class TransitionDataset(Dataset):
    """
    PyTorch Dataset wrapper supporting both in-memory NumPy arrays and
    disk-backed Memory-Mapped (np.memmap) matrices for zero-RAM overhead.
    """

    def __init__(self, vectors: np.ndarray | np.memmap) -> None:
        self.vectors = vectors

    def __len__(self) -> int:
        return len(self.vectors)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.from_numpy(np.array(self.vectors[idx], dtype=np.float32))


def train_autoencoder(
    vectors: np.ndarray | np.memmap,
    epochs: int = 10,
    batch_size: int = 2048,
    lr: float = 1e-3,
    device: torch.device | None = None,
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
# 2. GRID DISCRETIZATION & JSD METRICS
# ==========================================


def discretize_to_grid(
    coords_2d: np.ndarray,
    grid_size: int = 15,
    bounds: tuple[float, float, float, float] | None = None,
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
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
# 3. GLOBAL STYLE SPACE REFERENCE MODULE
# ==========================================


class GlobalStyleSpace:
    """
    Encapsulates a unified, pre-trained latent space (AutoEncoder + cuML UMAP) fitted
    exclusively on ground-truth human player transitions.
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
    def fit_from_memmap(
        cls,
        reference_memmap: np.memmap,
        ae_epochs: int = 10,
        device: torch.device | None = None,
        seed: int = 42,
    ) -> "GlobalStyleSpace":
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        n_samples = len(reference_memmap)
        logger.info(
            f"Fitting Global AutoEncoder on {n_samples} disk-mapped reference transitions..."
        )
        ae_model = train_autoencoder(
            reference_memmap, epochs=ae_epochs, batch_size=2048, device=device
        )

        logger.info("Extracting global 128-D latent embeddings in chunked passes...")
        latent_chunks = []
        chunk_size = 20000  # Smaller chunks to protect VRAM

        with torch.no_grad():
            for i in range(0, n_samples, chunk_size):
                chunk_vecs = torch.tensor(
                    np.array(reference_memmap[i : i + chunk_size], dtype=np.float32),
                    device=device,
                )
                _, latent_chunk = ae_model(chunk_vecs)
                latent_chunks.append(latent_chunk.cpu().numpy())
                del chunk_vecs

        latent_ref = np.vstack(latent_chunks).astype(np.float32)
        del latent_chunks
        gc.collect()
        torch.cuda.empty_cache()

        logger.info("Fitting GPU-accelerated cuML UMAP on reference latent manifold...")
        cuml_umap = cumlUMAP(
            n_neighbors=15, min_dist=0.1, n_components=2, random_state=seed
        )
        cuml_umap.fit(latent_ref)

        del latent_ref
        gc.collect()
        torch.cuda.empty_cache()

        return cls(ae_model=ae_model, cuml_umap=cuml_umap, device=device)

    def project_to_2d(self, vectors: np.ndarray, chunk_size: int = 10000) -> np.ndarray:
        """
        Projects 2304-D transition vectors onto the 2D manifold using chunked VRAM inference.
        """
        n_samples = len(vectors)
        latent_chunks = []

        # 1. Chunked AutoEncoder inference to avoid GPU VRAM spikes
        with torch.no_grad():
            for i in range(0, n_samples, chunk_size):
                chunk_vecs = torch.tensor(
                    vectors[i : i + chunk_size], dtype=torch.float32, device=self.device
                )
                _, latent_chunk = self.ae_model(chunk_vecs)
                latent_chunks.append(latent_chunk.cpu().numpy())
                del chunk_vecs

        latent_np = np.vstack(latent_chunks).astype(np.float32)
        del latent_chunks
        gc.collect()
        torch.cuda.empty_cache()

        # 2. Transform through cuML UMAP
        coords_2d = self.cuml_umap.transform(latent_np)

        del latent_np
        gc.collect()
        torch.cuda.empty_cache()

        return coords_2d


def evaluate_style_with_space(
    global_space: GlobalStyleSpace,
    p_arr: np.ndarray,
    m_arr: np.ndarray,
) -> dict[str, float]:
    p_2d = global_space.project_to_2d(p_arr)
    m_2d = global_space.project_to_2d(m_arr)

    p_dist, bounds = discretize_to_grid(p_2d, grid_size=15)
    m_dist, _ = discretize_to_grid(m_2d, grid_size=15, bounds=bounds)

    jsd_val = compute_jsd(p_dist, m_dist)
    js_dist = math.sqrt(jsd_val)

    return {
        "style_jsd": round(jsd_val, 6),
        "style_jsd_distance": round(js_dist, 6),
    }
