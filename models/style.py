"""
Core architectural pipeline for behavioral stylometry evaluation.

Includes the AutoEncoder network architecture, spatial grid discretization,
Jensen-Shannon Divergence (JSD) computation, and the global reference space module.
"""

import gc
import logging
import math

import numpy as np
import torch
from cuml.manifold import UMAP as cumlUMAP
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ==============================================================================
# 1. AUTOENCODER ARCHITECTURE (2304-D -> 128-D)
# ==============================================================================


class BoardTransitionAutoEncoder(nn.Module):
    """Symmetric fully-connected AutoEncoder architecture compressing high-dimensional transition vectors.

    Compresses 2304-dimensional board state transition vectors into a dense 128-dimensional latent
    representation space while maintaining reconstruction fidelity.
    """

    def __init__(self) -> None:
        """Initialize encoder and decoder network modules."""
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
        """Perform forward encoding and decoding reconstruction pass.

        Args:
            x (torch.Tensor): Input batch containing 2304-dimensional transition vectors.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Pair containing reconstructed vector tensor
                and dense bottleneck latent tensor.
        """
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent


class TransitionDataset(Dataset):
    """PyTorch Dataset wrapper supporting both in-memory arrays and disk-backed memmap structures.

    Enables high-throughput batch loading from memory-mapped files without consuming RAM.
    """

    def __init__(self, vectors: np.ndarray | np.memmap) -> None:
        """Initialize dataset wrapper around observation matrix.

        Args:
            vectors (Union[np.ndarray, np.memmap]): Source transition observation matrix.
        """
        self.vectors = vectors

    def __len__(self) -> int:
        """Return total number of observation vectors.

        Returns:
            int: Dataset vector count.
        """
        return len(self.vectors)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Retrieve single preprocessed floating-point tensor sample.

        Args:
            idx (int): Sample observation index.

        Returns:
            torch.Tensor: Single 2304-dimensional transition tensor.
        """
        return torch.from_numpy(np.array(self.vectors[idx], dtype=np.float32))


def train_autoencoder(
    vectors: np.ndarray | np.memmap,
    epochs: int = 10,
    batch_size: int = 2048,
    lr: float = 1e-3,
    device: torch.device | None = None,
) -> BoardTransitionAutoEncoder:
    """Train AutoEncoder model on positional transition observation data.

    Args:
        vectors (Union[np.ndarray, np.memmap]): Positional transition observations.
        epochs (int, optional): Optimization epoch budget. Defaults to 10.
        batch_size (int, optional): Mini-batch processing size. Defaults to 2048.
        lr (float, optional): Adam optimizer initial learning rate. Defaults to 1e-3.
        device (Optional[torch.device], optional): Computation target device. Defaults to None.

    Returns:
        BoardTransitionAutoEncoder: Trained AutoEncoder model instance placed in evaluation mode.
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


# ==============================================================================
# 2. GRID DISCRETIZATION & JSD METRICS
# ==============================================================================


def discretize_to_grid(
    coords_2d: np.ndarray,
    grid_size: int = 15,
    bounds: tuple[float, float, float, float] | None = None,
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    """Discretize continuous 2D coordinates into normalized spatial probability distributions.

    Args:
        coords_2d (np.ndarray): Array containing 2D spatial coordinate projections.
        grid_size (int, optional): Dimension count for spatial discretization grid. Defaults to 15.
        bounds (Optional[Tuple[float, float, float, float]], optional): Explicit spatial domain boundaries
            (min_x, max_x, min_y, max_y). Defaults to None.

    Returns:
        Tuple[np.ndarray, Tuple[float, float, float, float]]: Pair containing flattened
            normalized spatial probability array and defined spatial boundaries.
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
    """Compute Jensen-Shannon Divergence between two probability distributions.

    Args:
        p (np.ndarray): Primary input discrete probability distribution vector.
        q (np.ndarray): Secondary input discrete probability distribution vector.
        eps (float, optional): Epsilon threshold preventing log-zero division errors. Defaults to 1e-12.

    Returns:
        float: Calculated non-negative Jensen-Shannon Divergence score.
    """
    m = 0.5 * (p + q)
    p_c = np.clip(p, eps, 1.0)
    q_c = np.clip(q, eps, 1.0)
    m_c = np.clip(m, eps, 1.0)

    kl_p_m = np.sum(p * np.log2(p_c / m_c))
    kl_q_m = np.sum(q * np.log2(q_c / m_c))

    jsd = 0.5 * kl_p_m + 0.5 * kl_q_m
    return max(0.0, float(jsd))


# ==============================================================================
# 3. GLOBAL STYLE SPACE REFERENCE MODULE
# ==============================================================================


class GlobalStyleSpace:
    """Unified latent representation manifold (AutoEncoder + cuML UMAP) fitted on reference human transitions.

    Attributes:
        ae_model (BoardTransitionAutoEncoder): Pre-trained AutoEncoder dimensionality reduction model.
        cuml_umap (cumlUMAP): GPU-accelerated UMAP manifold transformation model.
        device (torch.device): Primary hardware target execution device.
    """

    def __init__(
        self,
        ae_model: BoardTransitionAutoEncoder,
        cuml_umap: cumlUMAP,
        device: torch.device,
    ) -> None:
        """Initialize the global style space container instance.

        Args:
            ae_model (BoardTransitionAutoEncoder): Trained AutoEncoder instance.
            cuml_umap (cumlUMAP): Fitted UMAP manifold instance.
            device (torch.device): Execution hardware device.
        """
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
        """Construct and fit global style space representations directly from disk-mapped observation matrices.

        Args:
            reference_memmap (np.memmap): Memory-mapped matrix containing baseline human transitions.
            ae_epochs (int, optional): AutoEncoder training epoch count. Defaults to 10.
            device (Optional[torch.device], optional): Computation target device. Defaults to None.
            seed (int, optional): Random seed parameter ensuring manifold determinism. Defaults to 42.

        Returns:
            GlobalStyleSpace: Instantiated and fitted global reference space object.
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        n_samples = len(reference_memmap)
        logger.info(
            "Fitting Global AutoEncoder on %d disk-mapped reference transitions...",
            n_samples,
        )
        ae_model = train_autoencoder(
            reference_memmap, epochs=ae_epochs, batch_size=2048, device=device
        )

        logger.info("Extracting global 128-D latent embeddings in chunked passes...")
        latent_chunks: list[np.ndarray] = []
        chunk_size = (
            20000  # Conservative chunk budget preventing VRAM allocation spikes
        )

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
        """Project high-dimensional transition vectors onto the fitted 2D manifold using chunked inference.

        Args:
            vectors (np.ndarray): Input 2304-dimensional transition vectors.
            chunk_size (int, optional): Inference batch chunk size. Defaults to 10000.

        Returns:
            np.ndarray: Projected 2D manifold coordinate matrix.
        """
        n_samples = len(vectors)
        latent_chunks: list[np.ndarray] = []

        # 1. Chunked AutoEncoder inference pass to prevent GPU memory allocation spikes
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

        # 2. Transform latent representations via fitted GPU cuML UMAP instance
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
    """Evaluate behavioral style alignment metrics between human and model observation distributions.

    Args:
        global_space (GlobalStyleSpace): Fitted reference style space object.
        p_arr (np.ndarray): Primary human subject transition vectors.
        m_arr (np.ndarray): Target candidate model transition vectors.

    Returns:
        Dict[str, float]: Evaluation dictionary containing raw Style JSD divergence
            and square-root Jensen-Shannon distance metrics.
    """
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
