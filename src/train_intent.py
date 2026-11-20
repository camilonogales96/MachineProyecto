"""
ETAPA 1 · Script de Entrenamiento: Clasificador de Intención
Arquitectura: Embedding (from scratch) -> Bi-GRU -> Masked Pooling -> Densa -> Softmax
Entrenamiento 100% desde cero sin modelos pre-entrenados ni RNN/LSTM tradicionales.
"""

import os
import sys
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, f1_score, accuracy_score

# Asegurar path del proyecto
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing.vocabulary import Vocabulary
from src.models.intent_classifier import BiGRUIntentClassifier

INTENT_MAP = {
    "pedido": 0,
    "consulta": 1,
    "reclamo": 2,
    "saludo_otro": 3
}
ID2INTENT = {v: k for k, v in INTENT_MAP.items()}


class IntentDataset(Dataset):
    def __init__(self, data_list, vocab: Vocabulary):
        self.samples = []
        for item in data_list:
            tokens = item["tokens"]
            intent = item["intencion"].strip().lower()
            if intent not in INTENT_MAP:
                continue
            input_ids = vocab.encode(tokens)
            label = INTENT_MAP[intent]
            self.samples.append((input_ids, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch, pad_idx):
    input_ids_list = [torch.tensor(s[0], dtype=torch.long) for s in batch]
    labels = torch.tensor([s[1] for s in batch], dtype=torch.long)
    padded_inputs = nn.utils.rnn.pad_sequence(input_ids_list, batch_first=True, padding_value=pad_idx)
    return padded_inputs, labels


def train_intent_model(
    data_dir: str,
    output_dir: str,
    epochs: int = 12,
    batch_size: int = 32,
    lr: float = 0.001,
    device: str = "cpu"
):
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Cargar datos
    train_path = os.path.join(data_dir, "train.json")
    val_path = os.path.join(data_dir, "val.json")
    
    with open(train_path, "r", encoding="utf-8") as f:
        train_data = json.load(f)
    with open(val_path, "r", encoding="utf-8") as f:
        val_data = json.load(f)
        
    print(f"Cargados: {len(train_data)} train, {len(val_data)} val")
    
    # 2. Construir Vocabulario desde cero
    vocab = Vocabulary()
    train_tokens = [d["tokens"] for d in train_data]
    vocab.build_vocab(train_tokens, min_freq=1)
    vocab_path = os.path.join(output_dir, "vocab.json")
    vocab.save(vocab_path)
    print(f"Vocabulario construido: {len(vocab)} tokens guardados en {vocab_path}")
    
    # Guardar mapa de intenciones
    with open(os.path.join(output_dir, "intent_map.json"), "w", encoding="utf-8") as f:
        json.dump(INTENT_MAP, f, indent=2)
        
    # 3. DataLoaders
    train_ds = IntentDataset(train_data, vocab)
    val_ds = IntentDataset(val_data, vocab)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=lambda b: collate_fn(b, vocab.pad_idx))
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            collate_fn=lambda b: collate_fn(b, vocab.pad_idx))
                            
    # 4. Inicializar Modelo Bi-GRU
    model = BiGRUIntentClassifier(
        vocab_size=len(vocab),
        embedding_dim=128,
        hidden_dim=128,
        num_classes=len(INTENT_MAP),
        num_layers=2,
        dropout=0.3,
        pad_idx=vocab.pad_idx
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)
    
    best_val_f1 = 0.0
    best_model_path = os.path.join(output_dir, "intent_model.pt")
    
    print("\n--- INICIANDO ENTRENAMIENTO BI-GRU (ETAPA 1) ---")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total_loss += loss.item()
            
        avg_train_loss = total_loss / len(train_loader)
        
        # Validación
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for x_val, y_val in val_loader:
                x_val, y_val = x_val.to(device), y_val.to(device)
                logits = model(x_val)
                preds = torch.argmax(logits, dim=-1)
                val_preds.extend(preds.cpu().tolist())
                val_targets.extend(y_val.cpu().tolist())
                
        val_acc = accuracy_score(val_targets, val_preds)
        val_f1 = f1_score(val_targets, val_preds, average="macro")
        scheduler.step(val_f1)
        
        print(f"Época {epoch:02d}/{epochs:02d} | Train Loss: {avg_train_loss:.4f} | Val Acc: {val_acc*100:.2f}% | Val F1-Macro: {val_f1:.4f}")
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), best_model_path)
            
    print(f"\nMejor F1-Macro de Validación: {best_val_f1:.4f}")
    print(f"Modelo guardado exitosamente en: {best_model_path}")
    
    # Reporte final detallado en Validación
    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    val_preds, val_targets = [], []
    with torch.no_grad():
        for x_val, y_val in val_loader:
            x_val, y_val = x_val.to(device), y_val.to(device)
            logits = model(x_val)
            preds = torch.argmax(logits, dim=-1)
            val_preds.extend(preds.cpu().tolist())
            val_targets.extend(y_val.cpu().tolist())
            
    target_names = [ID2INTENT[i] for i in range(len(INTENT_MAP))]
    print("\n--- REPORTE FINAL DE CLASIFICACIÓN (VALIDACIÓN) ---")
    print(classification_report(val_targets, val_preds, target_names=target_names, digits=4))


if __name__ == "__main__":
    data_directory = os.path.join(PROJECT_ROOT, "data", "dataset")
    output_directory = os.path.join(PROJECT_ROOT, "models_saved")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    train_intent_model(data_directory, output_directory, epochs=12, device=dev)
