"""Modelo Text-CNN puro para clasificar intenciones de mensajes de restaurante."""

from typing import Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ClasificadorIntencionesCNN(nn.Module):
    """Clasifica cada mensaje en una de las cuatro clases de intención.

    La arquitectura es exclusivamente convolucional: incrustación, tres
    convoluciones 1D, max pooling global, dropout y capa lineal.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        num_classes: int = 4,
        filtros_por_kernel: int = 100,
        tamanos_kernel: Sequence[int] = (3, 4, 5),
        dropout: float = 0.3,
        pad_idx: int = 0,
    ):
        """Crea las capas y los parámetros del clasificador.

        Args:
            vocab_size: cantidad de tokens distintos del vocabulario.
            embedding_dim: dimensión de los vectores aprendidos por Embedding.
            num_classes: cantidad de intenciones que predice el modelo.
            filtros_por_kernel: filtros de salida de cada convolución.
            tamanos_kernel: longitudes de las ventanas convolucionales.
            dropout: proporción de características apagadas durante el entrenamiento.
            pad_idx: índice reservado para el relleno de secuencias.
        """
        super().__init__()
        if not tamanos_kernel or any(tamano <= 0 for tamano in tamanos_kernel):
            raise ValueError("Los tamaños de kernel deben ser positivos.")

        self.pad_idx = pad_idx
        self.tamanos_kernel: Tuple[int, ...] = tuple(tamanos_kernel)

        # Entrada: [tamano_lote, longitud_secuencia], con índices de palabras.
        # Salida: [tamano_lote, longitud_secuencia, dimension_embedding].
        # Esta tabla se inicializa desde cero y aprende sus vectores durante el
        # entrenamiento; no utiliza embeddings preentrenados.
        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=pad_idx,
        )

        # Capa convolucional 1, kernel 3: busca patrones de 3 tokens.
        # Capa convolucional 2, kernel 4: busca patrones de 4 tokens.
        # Capa convolucional 3, kernel 5: busca patrones de 5 tokens.
        # Cada capa aplica 100 filtros en paralelo. Conv1d recibe:
        # [tamano_lote, dimension_embedding, longitud_secuencia].
        self.convoluciones = nn.ModuleList(
            nn.Conv1d(
                in_channels=embedding_dim,
                out_channels=filtros_por_kernel,
                kernel_size=tamano,
            )
            for tamano in self.tamanos_kernel
        )

        # La concatenación final contiene 100 características por cada kernel.
        dimension_salida = filtros_por_kernel * len(self.tamanos_kernel)
        self.dropout = nn.Dropout(dropout)
        self.clasificador = nn.Linear(dimension_salida, num_classes)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Calcula los cuatro logits de intención para cada secuencia del lote.

        Args:
            input_ids: tensor con forma [tamano_lote, longitud_secuencia].

        Returns:
            Logits con forma [tamano_lote, numero_de_clases].

        El método convierte los índices en embeddings, reordena los ejes para
        Conv1d, aplica ReLU y reduce cada mapa de activación con max pooling
        global. Finalmente concatena las características, aplica dropout y
        calcula los logits con una capa lineal.
        """
        if input_ids.ndim != 2:
            raise ValueError(
                "La entrada debe tener forma [tamano_lote, longitud_secuencia]."
            )
        if input_ids.size(1) == 0:
            raise ValueError("La longitud de secuencia debe ser mayor que cero.")

        # [tamano_lote, longitud_secuencia]
        # -> [tamano_lote, longitud_secuencia, dimension_embedding].
        incrustaciones = self.embedding(input_ids)

        # Se cuentan los tokens no usados como relleno para excluir ventanas
        # completamente vacías durante el max pooling.
        longitudes = input_ids.ne(self.pad_idx).sum(dim=1)  # [tamano_lote].

        # Cada Conv1d necesita una secuencia al menos tan larga como su kernel.
        # Se añaden vectores cero al final si el lote es demasiado corto.
        longitud_minima = max(self.tamanos_kernel)
        relleno_extra = max(0, longitud_minima - incrustaciones.size(1))
        if relleno_extra:
            # [B, L, E] -> [B, L + relleno_extra, E].
            incrustaciones = F.pad(incrustaciones, (0, 0, 0, relleno_extra))

        # Conv1d espera los canales en la segunda dimensión:
        # [tamano_lote, longitud_secuencia, dimension_embedding]
        # -> [tamano_lote, dimension_embedding, longitud_secuencia].
        caracteristicas = incrustaciones.transpose(1, 2)

        caracteristicas_globales = []
        for tamano, convolucion in zip(
            self.tamanos_kernel,
            self.convoluciones,
        ):
            # Capa 1, kernel 3: [B, E, L] -> [B, 100, L - 2].
            # Capa 2, kernel 4: [B, E, L] -> [B, 100, L - 3].
            # Capa 3, kernel 5: [B, E, L] -> [B, 100, L - 4].
            # En cada vuelta se calcula únicamente la forma de su kernel.
            salida_convolucion = convolucion(caracteristicas)

            # ReLU conserva la forma y elimina activaciones negativas.
            activaciones = F.relu(salida_convolucion)

            # Solo se agrupan ventanas que contienen tokens reales. Para textos
            # más cortos que el kernel, se conserva la única ventana disponible.
            ventanas_validas_por_ejemplo = (longitudes - tamano + 1).clamp(min=1)
            posiciones = torch.arange(
                activaciones.size(2),
                device=input_ids.device,
            ).unsqueeze(0)  # [1, L - tamano + 1].
            mascara = posiciones < ventanas_validas_por_ejemplo.unsqueeze(1)
            activaciones = activaciones.masked_fill(
                ~mascara.unsqueeze(1),
                torch.finfo(activaciones.dtype).min,
            )

            # Max pooling global temporal:
            # [B, 100, L - tamano + 1] -> [B, 100].
            maximo_global = activaciones.max(dim=2).values
            caracteristicas_globales.append(maximo_global)

        # Concatenación: [B, 100] + [B, 100] + [B, 100] -> [B, 300].
        caracteristicas_concatenadas = torch.cat(
            caracteristicas_globales,
            dim=1,
        )

        # Dropout regulariza las características y conserva la forma [B, 300].
        caracteristicas_regulares = self.dropout(caracteristicas_concatenadas)

        # Capa densa final: [B, 300] -> [B, 4] logits.
        return self.clasificador(caracteristicas_regulares)

    def predict(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Predice la intención más probable para cada mensaje.

        Args:
            input_ids: índices de tokens con forma [tamano_lote, longitud_secuencia].

        Returns:
            Índices de clase con forma [tamano_lote], sin calcular gradientes.
        """
        self.eval()
        with torch.no_grad():
            return self.forward(input_ids).argmax(dim=-1)
