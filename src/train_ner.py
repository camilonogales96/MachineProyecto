"""
ETAPA 2 · Script de Entrenamiento: Reconocedor de Entidades Nombradas (NER) y Atributos
Arquitectura: Embedding (from scratch) -> Bi-GRU -> Linear Tagging -> Linear-Chain CRF
Entrenamiento 100% desde cero con PyTorch y decodificación Viterbi.
"""

import os
import sys
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from seqeval.metrics import classification_report, f1_score

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing.vocabulary import Vocabulary, TagVocabulary
from src.models.ner_tagger import BiGRUNERTagger


class NERDataset(Dataset):
    def __init__(self, data_list, vocab: Vocabulary, tag_vocab: TagVocabulary):
        self.samples = []
        for item in data_list:
            tokens = item["tokens"]
            bio_tags = item.get("bio_tags", [])
            if not tokens or not bio_tags or len(tokens) != len(bio_tags):
                continue
            input_ids = vocab.encode(tokens)
            tag_ids = tag_vocab.encode(bio_tags)
            self.samples.append((input_ids, tag_ids, bio_tags))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def ner_collate_fn(batch, pad_token_id, pad_tag_id):
    input_ids_list = [torch.tensor(s[0], dtype=torch.long) for s in batch]
    tag_ids_list = [torch.tensor(s[1], dtype=torch.long) for s in batch]
    raw_bio_tags = [s[2] for s in batch]

    padded_inputs = nn.utils.rnn.pad_sequence(input_ids_list, batch_first=True, padding_value=pad_token_id)
    padded_tags = nn.utils.rnn.pad_sequence(tag_ids_list, batch_first=True, padding_value=pad_tag_id)

    return padded_inputs, padded_tags, raw_bio_tags


def train_ner_model(
    data_dir: str,
    output_dir: str,
    epochs: int = 12,
    batch_size: int = 32,
    lr: float = 0.003,
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

    # 2. Cargar o construir Vocabulario de Tokens
    vocab_path = os.path.join(output_dir, "vocab.json")
    if os.path.exists(vocab_path):
        vocab = Vocabulary.load(vocab_path)
    else:
        vocab = Vocabulary()
        vocab.build_vocab([d["tokens"] for d in train_data])
        vocab.save(vocab_path)

    # 3. Construir Vocabulario de Tags BIO
    tag_vocab = TagVocabulary(pad_tag="<PAD>")
    all_train_tags = [d["bio_tags"] for d in train_data if "bio_tags" in d]
    tag_vocab.build_vocab(all_train_tags)
    
    tag_vocab_path = os.path.join(output_dir, "tag_vocab.json")
    tag_vocab.save(tag_vocab_path)
    pad_tag_id = tag_vocab.tag2id["<PAD>"]
    print(f"Tags BIO registrados ({len(tag_vocab)}): {list(tag_vocab.tag2id.keys())}")

    # 4. DataLoaders
    train_ds = NERDataset(train_data, vocab, tag_vocab)
    val_ds = NERDataset(val_data, vocab, tag_vocab)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=lambda b: ner_collate_fn(b, vocab.pad_idx, pad_tag_id)
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        collate_fn=lambda b: ner_collate_fn(b, vocab.pad_idx, pad_tag_id)
    )

    # 5. Inicializar Modelo Bi-GRU + CRF
    model = BiGRUNERTagger(
        vocab_size=len(vocab),
        num_tags=len(tag_vocab),
        embedding_dim=128,
        hidden_dim=128,
        num_layers=2,
        dropout=0.3,
        pad_idx=vocab.pad_idx
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    best_val_f1 = 0.0
    best_model_path = os.path.join(output_dir, "ner_model.pt")

    print("\n--- INICIANDO ENTRENAMIENTO BI-GRU + CRF (ETAPA 2) ---")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for x_batch, y_batch, _ in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            loss = model(x_batch, y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        # Validación con Viterbi y seqeval
        model.eval()
        val_pred_tags = []
        val_true_tags = []

        with torch.no_grad():
            for x_val, _, raw_tags in val_loader:
                x_val = x_val.to(device)
                pred_paths = model.decode(x_val)
                for path in pred_paths:
                    pred_str_tags = tag_vocab.decode(path)
                    val_pred_tags.append(pred_str_tags)
                val_true_tags.extend(raw_tags)

        # Truncar predicciones a la longitud real si difiere por padding
        aligned_preds = []
        for p, t in zip(val_pred_tags, val_true_tags):
            aligned_preds.append(p[:len(t)])

        val_f1 = f1_score(val_true_tags, aligned_preds)
        scheduler.step(val_f1)

        print(f"Época {epoch:02d}/{epochs:02d} | Train CRF Loss: {avg_train_loss:.4f} | Val F1 (seqeval): {val_f1*100:.2f}%")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), best_model_path)

    print(f"\nMejor F1 (seqeval) de Validación: {best_val_f1*100:.2f}%")
    print(f"Modelo NER guardado exitosamente en: {best_model_path}")

    # Reporte final detallado
    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    val_pred_tags, val_true_tags = [], []
    with torch.no_grad():
        for x_val, _, raw_tags in val_loader:
            x_val = x_val.to(device)
            pred_paths = model.decode(x_val)
            for path in pred_paths:
                val_pred_tags.append(tag_vocab.decode(path))
            val_true_tags.extend(raw_tags)

    aligned_preds = [p[:len(t)] for p, t in zip(val_pred_tags, val_true_tags)]
    print("\n--- REPORTE FINAL DE ENTIDADES SEQEVAL (VALIDACIÓN) ---")
    print(classification_report(val_true_tags, aligned_preds, digits=4))


if __name__ == "__main__":
    data_directory = os.path.join(PROJECT_ROOT, "data", "dataset")
    output_directory = os.path.join(PROJECT_ROOT, "models_saved")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    train_ner_model(data_directory, output_directory, epochs=12, device=dev)
