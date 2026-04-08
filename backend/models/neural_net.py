"""
PyTorch MLP for match outcome prediction (disabled by default).

Only enabled in model_config.yaml when dataset > 2000 matches, because:
  - Neural networks need enough data to outperform gradient boosting
  - Football datasets are typically small (38 matches/team/season × 5 leagues = ~950/season)
  - XGBoost/LightGBM tend to win on tabular data at these scales

Architecture: 3-layer MLP with BatchNorm + Dropout, Softmax output.
Trained with CrossEntropyLoss and Adam optimizer.
"""

from __future__ import annotations

import numpy as np
from loguru import logger

from models.base import BaseModel

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


class _MLP(nn.Module if _TORCH_AVAILABLE else object):
    def __init__(self, input_dim: int, hidden_sizes: list[int], dropout: float):
        if not _TORCH_AVAILABLE:
            return
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_sizes:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 3))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class NeuralNetModel(BaseModel):
    """3-layer MLP with Softmax output for 3-class match outcome prediction."""

    def __init__(self, hidden_sizes: list[int] | None = None, dropout: float = 0.3,
                 epochs: int = 100, batch_size: int = 64, lr: float = 0.001):
        if not _TORCH_AVAILABLE:
            raise ImportError("PyTorch is not installed. Run: pip install torch")
        self.hidden_sizes = hidden_sizes or [256, 128, 64]
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self._model: "_MLP | None" = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> None:
        X_val = kwargs.get("X_val")
        y_val = kwargs.get("y_val")

        self._model = _MLP(X.shape[1], self.hidden_sizes, self.dropout).to(self._device)
        optimizer = optim.Adam(self._model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()

        X_t = torch.FloatTensor(X).to(self._device)
        y_t = torch.LongTensor(y).to(self._device)
        dataset = TensorDataset(X_t, y_t)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        for epoch in range(self.epochs):
            # .train() sets BatchNorm/Dropout to training mode (not Python's eval)
            self._model.train()
            total_loss = 0.0
            for X_batch, y_batch in loader:
                optimizer.zero_grad()
                out = self._model(X_batch)
                loss = criterion(out, y_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            if (epoch + 1) % 10 == 0:
                avg_loss = total_loss / len(loader)
                val_info = ""
                if X_val is not None and y_val is not None:
                    val_proba = self.predict_proba(X_val)
                    val_preds = val_proba.argmax(axis=1)
                    val_acc = (val_preds == y_val).mean()
                    val_info = f", val_acc={val_acc:.3f}"
                logger.debug(f"Epoch {epoch+1}/{self.epochs}: loss={avg_loss:.4f}{val_info}")

        logger.info(f"NeuralNet trained for {self.epochs} epochs on {len(X)} samples.")

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Model not trained — call fit() first.")
        # .eval() puts BatchNorm/Dropout into inference mode (PyTorch method, not Python built-in)
        self._model.eval()
        with torch.no_grad():
            X_t = torch.FloatTensor(X).to(self._device)
            logits = self._model(X_t)
            proba = torch.softmax(logits, dim=1).cpu().numpy()
        return proba
