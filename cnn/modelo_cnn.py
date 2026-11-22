"""Reconocedor Text-CNN de menciones de productos en secuencias de tokens."""

from typing import Dict, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


ETIQUETAS_PRODUCTO = {
    "O": 0,
    "B-PRODUCTO": 1,
    "I-PRODUCTO": 2,
}

# Alias coloquiales que aparecen en los datos y que se normalizan al menú.
ALIAS_PRODUCTO = {
    "papita": "Papas Fritas",
    "papitas": "Papas Fritas",
    "hamburguesita": "Hamburguesa Simple",
    "hamburguesitas": "Hamburguesa Simple",
    "hamburguesas": "Hamburguesa Simple",
    "hamburguesas dobles": "Hamburguesa Doble",
    "cocas": "Coca Cola",
    "cervezas": "Cerveza",
}


def normalizar_mencion_producto(mencion: str, parser) -> str:
    """Convierte una mención detectada a su nombre canónico del menú.

    Primero aplica alias locales para variantes coloquiales que no están
    incluidas en la ontología base; después usa el normalizador existente.
    """
    mencion_limpia = " ".join(mencion.lower().strip().split())
    mencion_limpia = mencion_limpia.replace(" - ", "-")
    if mencion_limpia in ALIAS_PRODUCTO:
        return ALIAS_PRODUCTO[mencion_limpia]

    sinonimos = getattr(parser, "synonym_to_canonical", {})
    if mencion_limpia in sinonimos:
        return sinonimos[mencion_limpia]

    # Prioriza el alias más específico para no confundir, por ejemplo,
    # "hamburguesa doble" con el alias más general "hamburguesa".
    for sinonimo in sorted(sinonimos, key=len, reverse=True):
        if sinonimo in mencion_limpia:
            return sinonimos[sinonimo]
    return parser.normalize_product_name(mencion_limpia)


class ReconocedorProductosCNN(nn.Module):
    """Etiqueta cada token como exterior o parte de un nombre de producto.

    Es una CNN pura. Conserva la dimensión temporal para poder señalar qué
    palabras forman cada mención. No aplica pooling global, porque este
    eliminaría las posiciones necesarias para extraer los productos del texto.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        num_tags: int = 3,
        filtros_por_kernel: int = 100,
        tamanos_kernel: Sequence[int] = (3, 4, 5),
        dropout: float = 0.3,
        pad_idx: int = 0,
    ):
        """Crea la incrustación, convoluciones y clasificador por token."""
        super().__init__()
        if not tamanos_kernel or any(tamano <= 0 for tamano in tamanos_kernel):
            raise ValueError("Los tamaños de kernel deben ser positivos.")

        self.pad_idx = pad_idx
        self.tamanos_kernel: Tuple[int, ...] = tuple(tamanos_kernel)
        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=pad_idx,
        )
        self.convoluciones = nn.ModuleList(
            nn.Conv1d(
                in_channels=embedding_dim,
                out_channels=filtros_por_kernel,
                kernel_size=tamano,
            )
            for tamano in self.tamanos_kernel
        )
        dimension_caracteristicas = filtros_por_kernel * len(self.tamanos_kernel)
        self.dropout = nn.Dropout(dropout)
        self.clasificador = nn.Linear(dimension_caracteristicas, num_tags)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Predice una etiqueta BIO para cada token de cada mensaje.

        Args:
            input_ids: IDs con forma [tamano_lote, longitud_secuencia].

        Returns:
            Logits con forma [tamano_lote, longitud_secuencia, numero_etiquetas].
        """
        if input_ids.ndim != 2:
            raise ValueError(
                "La entrada debe tener forma [tamano_lote, longitud_secuencia]."
            )
        if input_ids.size(1) == 0:
            raise ValueError("La longitud de secuencia debe ser mayor que cero.")

        # [B, L] -> [B, L, 128], donde B es el lote y L la longitud del texto.
        incrustaciones = self.embedding(input_ids)
        # Conv1d recibe los canales antes de la dimensión temporal: [B, 128, L].
        caracteristicas = incrustaciones.transpose(1, 2)

        salidas_convolucionales = []
        for tamano, convolucion in zip(
            self.tamanos_kernel,
            self.convoluciones,
        ):
            # Se agrega relleno asimétrico cuando haga falta para conservar L.
            relleno_izquierdo = (tamano - 1) // 2
            relleno_derecho = tamano - 1 - relleno_izquierdo
            entrada_rellenada = F.pad(
                caracteristicas,
                (relleno_izquierdo, relleno_derecho),
            )
            # [B, 128, L + tamano - 1] -> [B, 100, L].
            activaciones = F.relu(convolucion(entrada_rellenada))
            salidas_convolucionales.append(activaciones)

        # Concatenación por canal: tres salidas [B, 100, L] -> [B, 300, L].
        caracteristicas_combinadas = torch.cat(
            salidas_convolucionales,
            dim=1,
        )
        # Reordenar para que la capa lineal clasifique cada token:
        # [B, 300, L] -> [B, L, 300].
        caracteristicas_por_token = caracteristicas_combinadas.transpose(1, 2)
        caracteristicas_regulares = self.dropout(caracteristicas_por_token)
        # Capa densa compartida para cada posición: [B, L, 300] -> [B, L, 3].
        return self.clasificador(caracteristicas_regulares)

    def predict(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Devuelve la etiqueta predicha para cada posición del lote.

        La salida contiene IDs de etiquetas con forma [tamano_lote, longitud].
        """
        self.eval()
        with torch.no_grad():
            return self.forward(input_ids).argmax(dim=-1)
