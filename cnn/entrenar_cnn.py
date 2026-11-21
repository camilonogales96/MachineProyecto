"""Entrenamiento Text-CNN independiente para la clasificación de intenciones."""

import argparse
import json
import os
import sys
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.utils.data import DataLoader, Dataset


# Se agrega la raíz para reutilizar los módulos existentes sin modificarlos.
CARPETA_CNN = os.path.dirname(os.path.abspath(__file__))
RAIZ_PROYECTO = os.path.dirname(CARPETA_CNN)
if RAIZ_PROYECTO not in sys.path:
    sys.path.insert(0, RAIZ_PROYECTO)

from cnn.modelo_cnn import ClasificadorIntencionesCNN
from src.preprocessing.tokenizer import SimpleSpanishTokenizer
from src.preprocessing.vocabulary import Vocabulary


MAPA_INTENCIONES = {
    "pedido": 0,
    "consulta": 1,
    "reclamo": 2,
    "saludo_otro": 3,
}
INTENCION_POR_INDICE = {
    indice: intencion for intencion, indice in MAPA_INTENCIONES.items()
}


def preparar_registros(
    registros: List[Dict],
    tokenizador: SimpleSpanishTokenizer,
) -> List[Dict]:
    """Normaliza los ejemplos de entrada para el conjunto de intenciones.

    Conserva solo las cuatro etiquetas conocidas. Si un ejemplo no trae la
    lista de tokens, tokeniza su campo ``texto`` con el tokenizador del proyecto.

    Args:
        registros: ejemplos cargados desde un archivo JSON.
        tokenizador: tokenizador español existente que se reutiliza sin cambios.

    Returns:
        Ejemplos no vacíos con los campos ``tokens`` e ``intencion``.
    """
    registros_preparados = []
    for registro in registros:
        intencion = str(registro.get("intencion", "")).strip().lower()
        if intencion not in MAPA_INTENCIONES:
            continue

        tokens = registro.get("tokens")
        if not tokens:
            tokens = tokenizador.tokenize(str(registro.get("texto", "")))
        if tokens:
            registros_preparados.append(
                {"tokens": tokens, "intencion": intencion}
            )
    return registros_preparados


class ConjuntoIntenciones(Dataset):
    """Adapta los ejemplos al formato que espera un DataLoader de PyTorch."""

    def __init__(self, registros: List[Dict], vocabulario: Vocabulary):
        """Codifica tokens y convierte cada intención a su índice numérico.

        Args:
            registros: ejemplos preparados por ``preparar_registros``.
            vocabulario: vocabulario ya construido a partir del conjunto train.
        """
        self.ejemplos = [
            (
                vocabulario.encode(registro["tokens"]),
                MAPA_INTENCIONES[registro["intencion"]],
            )
            for registro in registros
        ]

    def __len__(self) -> int:
        """Devuelve la cantidad de ejemplos disponibles."""
        return len(self.ejemplos)

    def __getitem__(self, indice: int):
        """Devuelve una secuencia codificada y su etiqueta de intención.

        Returns:
            Una pareja ``(ids_de_tokens, id_de_intencion)``.
        """
        return self.ejemplos[indice]


def combinar_lote(lote, indice_relleno: int):
    """Construye tensores por lote y rellena las secuencias dinámicamente.

    Args:
        lote: ejemplos ``(ids_de_tokens, id_de_intencion)`` del DataLoader.
        indice_relleno: ID del token PAD que completa las secuencias cortas.

    Returns:
        Entradas de forma [tamano_lote, longitud_maxima] y etiquetas de forma
        [tamano_lote].
    """
    secuencias = [
        torch.tensor(ejemplo[0], dtype=torch.long)
        for ejemplo in lote
    ]
    etiquetas = torch.tensor(
        [ejemplo[1] for ejemplo in lote],
        dtype=torch.long,
    )
    # [B, longitudes variables] -> [B, longitud máxima del lote].
    entradas = nn.utils.rnn.pad_sequence(
        secuencias,
        batch_first=True,
        padding_value=indice_relleno,
    )
    return entradas, etiquetas


def guardar_graficas(historial: Dict[str, List[float]], carpeta_salida: str):
    """Dibuja y guarda las curvas de entrenamiento y validación.

    Args:
        historial: valores por época de pérdida, exactitud y F1-Macro.
        carpeta_salida: carpeta donde se escribe ``graficas_cnn.png``.

    La figura tiene un panel para la pérdida de entrenamiento y otro para las
    métricas de validación.
    """
    epocas = range(1, len(historial["perdida_entrenamiento"]) + 1)
    figura, (eje_perdida, eje_metricas) = plt.subplots(
        1,
        2,
        figsize=(12, 4),
    )

    eje_perdida.plot(
        epocas,
        historial["perdida_entrenamiento"],
        label="Pérdida de entrenamiento",
    )
    eje_perdida.set_title("Pérdida por época")
    eje_perdida.set_xlabel("Época")
    eje_perdida.set_ylabel("Pérdida Cross-Entropy")
    eje_perdida.grid(True, alpha=0.3)
    eje_perdida.legend()

    eje_metricas.plot(
        epocas,
        historial["exactitud_validacion"],
        label="Exactitud de validación",
    )
    eje_metricas.plot(
        epocas,
        historial["f1_macro_validacion"],
        label="F1-Macro de validación",
    )
    eje_metricas.set_title("Métricas por época")
    eje_metricas.set_xlabel("Época")
    eje_metricas.set_ylabel("Puntuación")
    eje_metricas.set_ylim(0, 1)
    eje_metricas.grid(True, alpha=0.3)
    eje_metricas.legend()

    figura.tight_layout()
    ruta_grafica = os.path.join(carpeta_salida, "graficas_cnn.png")
    figura.savefig(ruta_grafica, dpi=150)
    plt.close(figura)
    print(f"Gráficas guardadas en: {ruta_grafica}")


def entrenar(
    carpeta_datos: str,
    carpeta_salida: str,
    epocas: int = 12,
    tamano_lote: int = 32,
    tasa_aprendizaje: float = 0.001,
    dispositivo: str = "cpu",
):
    """Ejecuta el ciclo completo de entrenamiento y evaluación.

    Args:
        carpeta_datos: directorio que contiene ``train.json`` y ``val.json``.
        carpeta_salida: directorio exclusivo para pesos, vocabulario y gráfica.
        epocas: número de pasadas completas sobre los datos de entrenamiento.
        tamano_lote: cantidad de ejemplos procesados antes de cada actualización.
        tasa_aprendizaje: tamaño de paso que usa el optimizador Adam.
        dispositivo: ``cpu`` o ``cuda`` según el hardware disponible.

    El vocabulario se construye solo con train. En cada época calcula la pérdida
    de entrenamiento, exactitud y F1-Macro de validación, y guarda los pesos
    correspondientes al mejor F1-Macro.
    """
    os.makedirs(carpeta_salida, exist_ok=True)
    tokenizador = SimpleSpanishTokenizer()

    with open(
        os.path.join(carpeta_datos, "train.json"),
        "r",
        encoding="utf-8",
    ) as archivo:
        registros_entrenamiento = preparar_registros(
            json.load(archivo),
            tokenizador,
        )
    with open(
        os.path.join(carpeta_datos, "val.json"),
        "r",
        encoding="utf-8",
    ) as archivo:
        registros_validacion = preparar_registros(
            json.load(archivo),
            tokenizador,
        )

    if not registros_entrenamiento or not registros_validacion:
        raise ValueError(
            "train.json y val.json deben contener ejemplos válidos de intención."
        )

    print(
        f"Ejemplos: {len(registros_entrenamiento)} entrenamiento, "
        f"{len(registros_validacion)} validación."
    )

    vocabulario = Vocabulary()
    vocabulario.build_vocab(
        [registro["tokens"] for registro in registros_entrenamiento],
        min_freq=1,
    )
    vocabulario.save(os.path.join(carpeta_salida, "vocabulario_cnn.json"))
    with open(
        os.path.join(carpeta_salida, "intenciones_cnn.json"),
        "w",
        encoding="utf-8",
    ) as archivo:
        json.dump(MAPA_INTENCIONES, archivo, ensure_ascii=False, indent=2)

    conjunto_entrenamiento = ConjuntoIntenciones(
        registros_entrenamiento,
        vocabulario,
    )
    conjunto_validacion = ConjuntoIntenciones(
        registros_validacion,
        vocabulario,
    )
    cargador_entrenamiento = DataLoader(
        conjunto_entrenamiento,
        batch_size=tamano_lote,
        shuffle=True,
        collate_fn=lambda lote: combinar_lote(lote, vocabulario.pad_idx),
    )
    cargador_validacion = DataLoader(
        conjunto_validacion,
        batch_size=tamano_lote,
        shuffle=False,
        collate_fn=lambda lote: combinar_lote(lote, vocabulario.pad_idx),
    )

    modelo = ClasificadorIntencionesCNN(
        vocab_size=len(vocabulario),
        embedding_dim=128,
        num_classes=len(MAPA_INTENCIONES),
        filtros_por_kernel=100,
        tamanos_kernel=(3, 4, 5),
        dropout=0.3,
        pad_idx=vocabulario.pad_idx,
    ).to(dispositivo)

    funcion_perdida = nn.CrossEntropyLoss()
    optimizador = torch.optim.Adam(
        modelo.parameters(),
        lr=tasa_aprendizaje,
        weight_decay=1e-5,
    )
    planificador = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizador,
        mode="max",
        factor=0.5,
        patience=2,
    )
    ruta_pesos = os.path.join(carpeta_salida, "pesos_cnn.pt")
    mejor_f1 = -1.0
    historial = {
        "perdida_entrenamiento": [],
        "exactitud_validacion": [],
        "f1_macro_validacion": [],
    }

    print("\n--- ENTRENAMIENTO TEXT-CNN ---")
    for epoca in range(1, epocas + 1):
        modelo.train()
        perdida_total = 0.0
        for entradas, etiquetas in cargador_entrenamiento:
            entradas = entradas.to(dispositivo)
            etiquetas = etiquetas.to(dispositivo)

            optimizador.zero_grad()
            logits = modelo(entradas)  # [tamano_lote, 4].
            perdida = funcion_perdida(logits, etiquetas)
            perdida.backward()
            nn.utils.clip_grad_norm_(modelo.parameters(), max_norm=5.0)
            optimizador.step()
            perdida_total += perdida.item()

        perdida_media = perdida_total / len(cargador_entrenamiento)
        modelo.eval()
        predicciones = []
        etiquetas_reales = []
        with torch.no_grad():
            for entradas, etiquetas in cargador_validacion:
                logits = modelo(entradas.to(dispositivo))
                predicciones.extend(logits.argmax(dim=-1).cpu().tolist())
                etiquetas_reales.extend(etiquetas.tolist())

        exactitud = accuracy_score(etiquetas_reales, predicciones)
        f1_macro = f1_score(
            etiquetas_reales,
            predicciones,
            average="macro",
            zero_division=0,
        )
        planificador.step(f1_macro)
        historial["perdida_entrenamiento"].append(perdida_media)
        historial["exactitud_validacion"].append(exactitud)
        historial["f1_macro_validacion"].append(f1_macro)

        print(
            f"Época {epoca:02d}/{epocas:02d} | "
            f"Train Loss: {perdida_media:.4f} | "
            f"Val Acc: {exactitud * 100:.2f}% | "
            f"F1-Macro: {f1_macro:.4f}"
        )
        if f1_macro > mejor_f1:
            mejor_f1 = f1_macro
            torch.save(modelo.state_dict(), ruta_pesos)

    guardar_graficas(historial, carpeta_salida)
    print(f"Mejor F1-Macro: {mejor_f1:.4f}")
    print(f"Pesos guardados en: {ruta_pesos}")

    modelo.load_state_dict(torch.load(ruta_pesos, map_location=dispositivo))
    modelo.eval()
    predicciones_finales = []
    etiquetas_finales = []
    with torch.no_grad():
        for entradas, etiquetas in cargador_validacion:
            logits = modelo(entradas.to(dispositivo))
            predicciones_finales.extend(logits.argmax(dim=-1).cpu().tolist())
            etiquetas_finales.extend(etiquetas.tolist())

    nombres_intenciones = [
        INTENCION_POR_INDICE[indice]
        for indice in range(len(MAPA_INTENCIONES))
    ]
    print("\n--- REPORTE DE VALIDACIÓN ---")
    print(
        classification_report(
            etiquetas_finales,
            predicciones_finales,
            labels=list(range(len(MAPA_INTENCIONES))),
            target_names=nombres_intenciones,
            digits=4,
            zero_division=0,
        )
    )


def leer_argumentos():
    """Lee rutas y parámetros opcionales escritos en la línea de comandos.

    Returns:
        Un espacio de argumentos con rutas y parámetros de entrenamiento.
    """
    analizador = argparse.ArgumentParser(
        description="Entrena un clasificador Text-CNN para intenciones."
    )
    analizador.add_argument(
        "--carpeta-datos",
        default=os.path.join(RAIZ_PROYECTO, "data", "dataset"),
        help="Carpeta donde están train.json y val.json.",
    )
    analizador.add_argument(
        "--carpeta-salida",
        default=os.path.join(CARPETA_CNN, "salidas"),
        help="Carpeta independiente donde se guardan los resultados CNN.",
    )
    analizador.add_argument("--epocas", type=int, default=12)
    analizador.add_argument("--tamano-lote", type=int, default=32)
    analizador.add_argument("--tasa-aprendizaje", type=float, default=0.001)
    return analizador.parse_args()


if __name__ == "__main__":
    argumentos = leer_argumentos()
    dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
    entrenar(
        carpeta_datos=argumentos.carpeta_datos,
        carpeta_salida=argumentos.carpeta_salida,
        epocas=argumentos.epocas,
        tamano_lote=argumentos.tamano_lote,
        tasa_aprendizaje=argumentos.tasa_aprendizaje,
        dispositivo=dispositivo,
    )
