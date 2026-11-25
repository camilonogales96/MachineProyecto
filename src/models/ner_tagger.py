"""
ETAPA 2 · Reconocedor de Entidades Nombradas (NER) y Extracción de Atributos
Arquitectura: Embeddings + GRU Bidireccional (Bi-GRU) + Capa Densa/Lineal + CRF
Entrenado 100% desde cero (sin modelos pre-entrenados, sin RNN clásica, sin LSTM).
"""

import torch
import torch.nn as nn
from typing import List, Tuple, Optional


def log_sum_exp(tensor: torch.Tensor, dim: int = -1, keepdim: bool = False) -> torch.Tensor:
    """Cálculo numéricamente estable de Log-Sum-Exp."""
    max_val, _ = torch.max(tensor, dim=dim, keepdim=True)
    max_val_sq = max_val.squeeze(dim) if not keepdim else max_val
    return max_val_sq + torch.log(torch.sum(torch.exp(tensor - max_val), dim=dim, keepdim=keepdim))


class LinearChainCRF(nn.Module):
    """
    Capa Linear-Chain Conditional Random Field (CRF) nativa en PyTorch.
    Optimiza la probabilidad conjunta de la secuencia de etiquetas BIO respetando
    transiciones válidas (ej. I-PRODUCTO no puede seguir a B-CANTIDAD).
    """
    def __init__(self, num_tags: int):
        super().__init__()
        self.num_tags = num_tags
        self.transitions = nn.Parameter(torch.randn(num_tags, num_tags))
        self.start_transitions = nn.Parameter(torch.randn(num_tags))
        self.end_transitions = nn.Parameter(torch.randn(num_tags))

    def _forward_alg(self, emissions: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, num_tags = emissions.shape
        alphas = self.start_transitions.view(1, num_tags) + emissions[:, 0]
        
        for t in range(1, seq_len):
            emit_score = emissions[:, t].unsqueeze(2)           # [batch, num_tags, 1]
            trans_score = self.transitions.unsqueeze(0)          # [1, num_tags, num_tags]
            alpha_broadcast = alphas.unsqueeze(1)               # [batch, 1, num_tags]
            
            next_alphas = log_sum_exp(alpha_broadcast + trans_score + emit_score, dim=-1)
            mask_t = mask[:, t].unsqueeze(-1)
            alphas = torch.where(mask_t, next_alphas, alphas)
            
        terminal_scores = alphas + self.end_transitions.view(1, num_tags)
        return log_sum_exp(terminal_scores, dim=-1)

    def _gold_score(self, emissions: torch.Tensor, tags: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = emissions.shape
        score = torch.zeros(batch_size, device=emissions.device)
        
        first_tags = tags[:, 0]
        score += self.start_transitions[first_tags]
        score += emissions[torch.arange(batch_size), 0, first_tags]
        
        for t in range(1, seq_len):
            mask_t = mask[:, t]
            prev_t = tags[:, t - 1]
            curr_t = tags[:, t]
            
            trans = self.transitions[curr_t, prev_t]
            emit = emissions[torch.arange(batch_size), t, curr_t]
            
            step_score = trans + emit
            score += torch.where(mask_t, step_score, torch.zeros_like(score))
            
        last_valid_indices = mask.long().sum(dim=1) - 1
        last_tags = tags[torch.arange(batch_size), last_valid_indices]
        score += self.end_transitions[last_tags]
        return score

    def forward(self, emissions: torch.Tensor, tags: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Pérdida Negative Log-Likelihood."""
        forward_score = self._forward_alg(emissions, mask)
        gold_score = self._gold_score(emissions, tags, mask)
        return torch.mean(forward_score - gold_score)

    def decode(self, emissions: torch.Tensor, mask: torch.Tensor) -> List[List[int]]:
        """Algoritmo Viterbi para decodificación de ruta óptima."""
        batch_size, seq_len, num_tags = emissions.shape
        best_paths = []
        
        for b in range(batch_size):
            b_len = int(mask[b].sum().item())
            if b_len == 0:
                best_paths.append([])
                continue
                
            viterbi_vars = self.start_transitions + emissions[b, 0]
            backpointers = []
            
            for t in range(1, b_len):
                next_tag_var = viterbi_vars.unsqueeze(0) + self.transitions
                max_vars, bptrs = torch.max(next_tag_var, dim=-1)
                viterbi_vars = max_vars + emissions[b, t]
                backpointers.append(bptrs.tolist())
                
            terminal_vars = viterbi_vars + self.end_transitions
            best_tag_id = int(torch.argmax(terminal_vars).item())
            
            best_path = [best_tag_id]
            for bptrs_t in reversed(backpointers):
                best_tag_id = bptrs_t[best_tag_id]
                best_path.append(best_tag_id)
            best_path.reverse()
            best_paths.append(best_path)
            
        return best_paths


class BiGRUNERTagger(nn.Module):
    """
    Modelo Bi-GRU + Capa Lineal + CRF para extracción de entidades (NER) en pedidos.
    100% Desde cero con PyTorch.
    """
    def __init__(
        self,
        vocab_size: int,
        num_tags: int,
        embedding_dim: int = 128,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        pad_idx: int = 0
    ):
        super().__init__()
        self.pad_idx = pad_idx
        self.num_tags = num_tags
        
        # 1. Capa de Embedding
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
        
        # 3. Capa Densa / Lineal hacia las etiquetas
        gru_out_dim = hidden_dim * 2
        self.linear_tagger = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(gru_out_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_tags)
        )
        
        # 4. Capa CRF
        self.crf = LinearChainCRF(num_tags)

    def forward(self, input_ids: torch.Tensor, tags: torch.Tensor) -> torch.Tensor:
        """Calcula la pérdida CRF."""
        mask = (input_ids != self.pad_idx)
        embedded = self.embed_dropout(self.embedding(input_ids))
        gru_out, _ = self.gru(embedded)
        emissions = self.linear_tagger(gru_out)
        loss = self.crf(emissions, tags, mask)
        return loss

    def decode(self, input_ids: torch.Tensor) -> List[List[int]]:
        """Predice las secuencias óptimas de etiquetas usando Viterbi."""
        self.eval()
        mask = (input_ids != self.pad_idx)
        with torch.no_grad():
            embedded = self.embedding(input_ids)
            gru_out, _ = self.gru(embedded)
            emissions = self.linear_tagger(gru_out)
            best_paths = self.crf.decode(emissions, mask)
        return best_paths
