"""
Tokenizador y preprocesamiento de texto en lenguaje natural en español para pedidos de restaurante.
Sin dependencias de librerías externas ni modelos pre-entrenados.
"""

import re
import unicodedata
from typing import List, Tuple

class SimpleSpanishTokenizer:
    """
    Tokenizador determinista con extracción de offsets para soporte de BIO tagging en NER.
    """
    
    # Patrón de tokenización: palabras con tildes/eñes, números y signos relevantes
    TOKEN_REGEX = re.compile(r"[a-záéíóúüñA-ZÁÉÍÓÚÜÑ0-9]+|[^\s\w]", re.UNICODE)

    def __init__(self, lowercase: bool = True, strip_accents: bool = False):
        self.lowercase = lowercase
        self.strip_accents = strip_accents

    def normalize_text(self, text: str) -> str:
        """Limpia espacios repetidos y normaliza caracteres Unicode."""
        if not text:
            return ""
        # Normalización Unicode NFC
        text = unicodedata.normalize("NFC", text)
        if self.strip_accents:
            # Remueve tildes si se especifica
            text = "".join(
                c for c in unicodedata.normalize("NFD", text)
                if unicodedata.category(c) != "Mn"
            )
        # Normaliza espacios en blanco
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def tokenize(self, text: str) -> List[str]:
        """Convierte texto en lista de tokens."""
        text = self.normalize_text(text)
        if self.lowercase:
            text = text.lower()
        matches = self.TOKEN_REGEX.findall(text)
        return matches

    def tokenize_with_offsets(self, text: str) -> List[Tuple[str, int, int]]:
        """
        Retorna una lista de tuplas: (token, start_char, end_char).
        Esencial para alinear anotaciones de entidades (spans) a nivel token.
        """
        text_norm = self.normalize_text(text)
        tokens_with_spans = []
        for match in self.TOKEN_REGEX.finditer(text_norm):
            token = match.group(0)
            if self.lowercase:
                token = token.lower()
            tokens_with_spans.append((token, match.start(), match.end()))
        return tokens_with_spans


if __name__ == "__main__":
    tokenizer = SimpleSpanishTokenizer()
    ejemplo = "¡Hola buenas tardes! Quisiera pedir 2 hamburguesas dobles sin cebolla, por favor."
    tokens = tokenizer.tokenize(ejemplo)
    print("Tokens:", tokens)
    spans = tokenizer.tokenize_with_offsets(ejemplo)
    print("Spans:", spans[:5])
