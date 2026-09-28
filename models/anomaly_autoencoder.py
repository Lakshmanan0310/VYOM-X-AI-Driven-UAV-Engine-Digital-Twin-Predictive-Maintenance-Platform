"""
Residual Autoencoder for Unsupervised Anomaly Detection (SIH26054 Section 13)
Trained exclusively on healthy engine residual sequences.
Reconstruction error above statistical threshold triggers the downstream fault diagnosis stage.
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, List, Dict, Optional

RESIDUAL_CHANNELS = [
    "rpm", "manifold_pressure", "fuel_flow", "oil_pressure", "oil_temp",
    "cht_cyl_1", "cht_cyl_2", "cht_cyl_3", "cht_cyl_4",
    "egt_cyl_1", "egt_cyl_2", "egt_cyl_3", "egt_cyl_4",
    "vibration_rms", "bus_voltage"
]

class ResidualAutoencoderNet(nn.Module):
    """
    Symmetric bottleneck autoencoder for normalized residual reconstruction.
    """
    def __init__(self, input_dim: int = len(RESIDUAL_CHANNELS), latent_dim: int = 6):
        super().__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.1),
            nn.Linear(32, 16),
            nn.LeakyReLU(0.1),
            nn.Linear(16, latent_dim),
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.LeakyReLU(0.1),
            nn.Linear(16, 32),
            nn.LeakyReLU(0.1),
            nn.Linear(32, input_dim)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        return reconstruction, latent

class AnomalyDetector:
    """
    Inference wrapper for the Residual Autoencoder with calibrated thresholding.
    """
    def __init__(self, model_path: Optional[str] = None):
        self.device = torch.device("cpu")
        self.model = ResidualAutoencoderNet().to(self.device)
        self.threshold = 1.25  # Calibrated default: Mean + 3.0 * sigma
        self.is_trained = False

        if model_path:
            self.load(model_path)

    def fit_threshold(self, healthy_residuals: np.ndarray, k_sigma: float = 3.0):
        """
        Calibrates anomaly threshold on held-out healthy validation set.
        """
        self.model.eval()
        with torch.no_grad():
            x = torch.tensor(healthy_residuals, dtype=torch.float32).to(self.device)
            recon, _ = self.model(x)
            mse_errors = torch.mean((x - recon) ** 2, dim=1).cpu().numpy()

        mu = float(np.mean(mse_errors))
        sigma = float(np.std(mse_errors))
        self.threshold = mu + k_sigma * sigma
        self.is_trained = True
        return mu, sigma, self.threshold

    def detect(self, residual_dict: Dict[str, float]) -> Tuple[bool, float, float, float]:
        """
        Computes reconstruction error and flags anomaly status.
        Returns: (is_anomaly, raw_mse, threshold, anomaly_index_0_to_1)
        """
        self.model.eval()
        x_vec = np.array([residual_dict.get(ch, 0.0) for ch in RESIDUAL_CHANNELS], dtype=np.float32).reshape(1, -1)
        with torch.no_grad():
            x = torch.tensor(x_vec).to(self.device)
            recon, _ = self.model(x)
            mse = float(torch.mean((x - recon) ** 2).item())

        is_anomaly = (mse >= self.threshold)
        # Calibrated normalized anomaly index: 0.0 (clean) to 1.0 (severe divergence)
        # Scaled smoothly relative to the calibrated 3-sigma threshold
        anom_index = float(1.0 - np.exp(-mse / max(1e-6, 2.0 * self.threshold)))
        return is_anomaly, round(mse, 2), round(self.threshold, 2), round(anom_index, 3)

    def save(self, path: str):
        torch.save({
            "model_state": self.model.state_dict(),
            "threshold": self.threshold
        }, path)

    def load(self, path: str):
        data = torch.load(path, map_location=self.device)
        self.model.load_state_dict(data["model_state"])
        self.threshold = data["threshold"]
        self.is_trained = True
        self.model.eval()
