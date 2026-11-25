"""
Gestión de Vocabulario para modelos Deep Learning entrenados desde cero.
Manejo de tokens especiales (<PAD>, <UNK>, <SOS>, <EOS>), frecuencias y persistencia.
"""

import json
from collections import Counter
from typing import List, Dict, Optional, Iterable

class Vocabulary:
    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"
    SOS_TOKEN = "<SOS>"
    EOS_TOKEN = "<EOS>"

    def __init__(self, pad_token: str = PAD_TOKEN, unk_token: str = UNK_TOKEN,
                 sos_token: Optional[str] = SOS_TOKEN, eos_token: Optional[str] = EOS_TOKEN):
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.sos_token = sos_token
        self.eos_token = eos_token

        self.token2id: Dict[str, int] = {}
        self.id2token: Dict[int, str] = {}
        self.freqs: Counter = Counter()

        # Registrar tokens especiales
        self._add_special_tokens()

    def _add_special_tokens(self):
        specials = [self.pad_token, self.unk_token]
        if self.sos_token:
            specials.append(self.sos_token)
        if self.eos_token:
            specials.append(self.eos_token)
        for token in specials:
            self._add_token(token)

    def _add_token(self, token: str) -> int:
        if token not in self.token2id:
            idx = len(self.token2id)
            self.token2id[token] = idx
            self.id2token[idx] = token
            return idx
        return self.token2id[token]

    @property
    def pad_idx(self) -> int:
        return self.token2id[self.pad_token]

    @property
    def unk_idx(self) -> int:
        return self.token2id[self.unk_token]

    @property
    def sos_idx(self) -> Optional[int]:
        return self.token2id.get(self.sos_token) if self.sos_token else None

    @property
    def eos_idx(self) -> Optional[int]:
        return self.token2id.get(self.eos_token) if self.eos_token else None

    def __len__(self) -> int:
        return len(self.token2id)

    def build_vocab(self, tokenized_texts: Iterable[List[str]], min_freq: int = 1, max_size: Optional[int] = None):
        """Construye el vocabulario a partir de textos tokenizados."""
        for tokens in tokenized_texts:
            self.freqs.update(tokens)

        # Ordenar por frecuencia descendente
        most_common = self.freqs.most_common()
        for token, count in most_common:
            if count < min_freq:
                break
            if max_size and len(self.token2id) >= max_size:
                break
            self._add_token(token)

    def encode(self, tokens: List[str], add_sos: bool = False, add_eos: bool = False) -> List[int]:
        """Convierte una lista de tokens a sus correspondientes IDs."""
        ids = []
        if add_sos and self.sos_idx is not None:
            ids.append(self.sos_idx)
        for token in tokens:
            ids.append(self.token2id.get(token, self.unk_idx))
        if add_eos and self.eos_idx is not None:
            ids.append(self.eos_idx)
        return ids

    def decode(self, ids: List[int], remove_special: bool = True) -> List[str]:
        """Convierte una lista de IDs a tokens."""
        tokens = []
        specials = {self.pad_idx, self.unk_idx}
        if self.sos_idx is not None:
            specials.add(self.sos_idx)
        if self.eos_idx is not None:
            specials.add(self.eos_idx)

        for idx in ids:
            if remove_special and idx in specials:
                continue
            tokens.append(self.id2token.get(idx, self.unk_token))
        return tokens

    def save(self, filepath: str):
        """Guarda el vocabulario en un archivo JSON."""
        data = {
            "token2id": self.token2id,
            "pad_token": self.pad_token,
            "unk_token": self.unk_token,
            "sos_token": self.sos_token,
            "eos_token": self.eos_token,
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, filepath: str) -> "Vocabulary":
        """Carga un vocabulario desde un archivo JSON."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        vocab = cls(
            pad_token=data.get("pad_token", cls.PAD_TOKEN),
            unk_token=data.get("unk_token", cls.UNK_TOKEN),
            sos_token=data.get("sos_token", cls.SOS_TOKEN),
            eos_token=data.get("eos_token", cls.EOS_TOKEN)
        )
        vocab.token2id = data["token2id"]
        vocab.id2token = {int(v): k for k, v in vocab.token2id.items()}
        return vocab


class TagVocabulary:
    """Vocabulario especializado para etiquetas (ej. BIO tags en NER o clases de intención)."""
    def __init__(self, pad_tag: Optional[str] = "<PAD>"):
        self.pad_tag = pad_tag
        self.tag2id: Dict[str, int] = {}
        self.id2tag: Dict[int, str] = {}
        if self.pad_tag:
            self._add_tag(self.pad_tag)

    def _add_tag(self, tag: str) -> int:
        if tag not in self.tag2id:
            idx = len(self.tag2id)
            self.tag2id[tag] = idx
            self.id2tag[idx] = tag
            return idx
        return self.tag2id[tag]

    def build_vocab(self, tag_lists: Iterable[List[str]]):
        for tags in tag_lists:
            for tag in tags:
                self._add_tag(tag)

    def encode(self, tags: List[str]) -> List[int]:
        return [self.tag2id[t] for t in tags]

    def decode(self, ids: List[int]) -> List[str]:
        return [self.id2tag[i] for i in ids]

    def __len__(self) -> int:
        return len(self.tag2id)

    def save(self, filepath: str):
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump({"tag2id": self.tag2id, "pad_tag": self.pad_tag}, f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> "TagVocabulary":
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        tag_vocab = cls(pad_tag=data.get("pad_tag"))
        tag_vocab.tag2id = data["tag2id"]
        tag_vocab.id2tag = {int(v): k for k, v in tag_vocab.tag2id.items()}
        return tag_vocab
