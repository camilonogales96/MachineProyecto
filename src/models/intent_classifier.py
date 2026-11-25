"""
ETAPA 1 · Clasificador de Intención
Arquitectura: Embeddings + GRU Bidireccional (Bi-GRU) + Capa Densa/Lineal
Entrenado 100% desde cero (sin modelos pre-entrenados, sin RNN clásica, sin LSTM).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class BiGRUIntentClassifier(nn.Module):
    """
    Clasificador de intención para pedidos y consultas de restaurante.
    Arquitectura:
    1. Capa de Embedding (entrenada from scratch)
    2. GRU Bidireccional (Bi-GRU) de múltiples capas
    3. Mecanismo de Pooling sobre estados temporales válidos
    4. Capa Densa / Lineal con regularización Dropout
    """
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        hidden_dim: int = 128,
        num_classes: int = 4,
        num_layers: int = 2,
        dropout: float = 0.3,
        pad_idx: int = 0
    ):
        super().__init__()
        self.pad_idx = pad_idx
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        
        # 1. Embedding
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.embed_dropout = nn.Dropout(dropout)
        
        # 2. GRU Bidireccional (Bi-GRU)
        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        gru_out_dim = hidden_dim * 2  # Bidireccional: forward + backward
        
        # 3. Capa Densa / Lineal de clasificación
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(gru_out_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, input_ids: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            input_ids: [batch_size, seq_len] tensor de IDs de tokens
        Returns:
            logits: [batch_size, num_classes]
        """
        mask = (input_ids != self.pad_idx)  # [batch_size, seq_len]
        
        # [batch_size, seq_len, embedding_dim]
        embedded = self.embed_dropout(self.embedding(input_ids))
        
        # Bi-GRU forward -> [batch_size, seq_len, hidden_dim * 2]
        gru_out, h_n = self.gru(embedded)
        
        # Mean Pooling ponderado por la máscara (ignora tokens PAD)
        mask_expanded = mask.unsqueeze(-1).float()  # [batch_size, seq_len, 1]
        sum_masked = torch.sum(gru_out * mask_expanded, dim=1)
        lengths_clamped = torch.clamp(mask_expanded.sum(dim=1), min=1.0)
        pooled = sum_masked / lengths_clamped  # [batch_size, hidden_dim * 2]
        
        # Capa Densa / Lineal
        logits = self.classifier(pooled)
        return logits

    def predict(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Retorna las predicciones de clase más probables."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(input_ids)
            preds = torch.argmax(logits, dim=-1)
        return preds
