"""Entrenamiento de Text-CNN para detectar productos del menú en un mensaje."""

import argparse
import json
import os
import sys
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from seqeval.metrics import classification_report, f1_score
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, Dataset


# La raíz permite reutilizar, en modo de solo lectura, los módulos existentes.
CARPETA_CNN = os.path.dirname(os.path.abspath(__file__))
RAIZ_PROYECTO = os.path.dirname(CARPETA_CNN)
if RAIZ_PROYECTO not in sys.path:
    sys.path.insert(0, RAIZ_PROYECTO)

from cnn.modelo_cnn import (
    ALIAS_PRODUCTO,
    ETIQUETAS_PRODUCTO,
    ReconocedorProductosCNN,
    normalizar_mencion_producto,
)
from src.models.relation_extractor import extract_chunks
from src.preprocessing.tokenizer import SimpleSpanishTokenizer
from src.preprocessing.vocabulary import Vocabulary
from src.structuring.parser import OrderParser


INDICE_A_ETIQUETA = {
    indice: etiqueta for etiqueta, indice in ETIQUETAS_PRODUCTO.items()
}
ETIQUETA_RELLENO = -100


def cargar_ontologia(ruta_ontologia: str) -> Tuple[Dict, OrderParser, set]:
    """Carga el menú y prepara el normalizador de nombres existente.

    Returns:
        Ontología, parser del proyecto y conjunto de nombres canónicos del menú.
    """
    with open(ruta_ontologia, "r", encoding="utf-8") as archivo:
        ontologia = json.load(archivo)
    parser = OrderParser(ontologia)
    nombres_menu = {
        producto["nombre"]
        for grupo in ontologia.get("categorias", {}).values()
        for producto in grupo
    }
    return ontologia, parser, nombres_menu


def preparar_registros(
    registros: List[Dict],
    tokenizador: SimpleSpanishTokenizer,
    parser: OrderParser,
    nombres_menu: set,
) -> List[Dict]:
    """Prepara secuencias con etiquetas BIO solo para productos del menú.

    Las etiquetas originales de cantidad, ingredientes, exclusiones y otros
    atributos se convierten en ``O``. Solo se conservan menciones ``PRODUCTO``
    que puedan normalizarse a uno de los productos definidos en la ontología.
    """
    preparados = []
    for numero, registro in enumerate(registros):
        tokens = registro.get("tokens")
        etiquetas_originales = registro.get("bio_tags")
        if not tokens and registro.get("texto"):
            tokens = tokenizador.tokenize(registro["texto"])
        if not tokens or not etiquetas_originales:
            raise ValueError(
                f"El ejemplo {numero} debe tener tokens y etiquetas BIO."
            )
        if len(tokens) != len(etiquetas_originales):
            raise ValueError(
                f"El ejemplo {numero} tiene {len(tokens)} tokens y "
                f"{len(etiquetas_originales)} etiquetas BIO."
            )

        # Solo estas tres etiquetas se entrenan; las demás entidades no son
        # objetivo de esta CNN de reconocimiento de productos.
        etiquetas_producto = ["O"] * len(tokens)
        fragmentos = extract_chunks(tokens, etiquetas_originales)
        for fragmento in fragmentos:
            if fragmento["type"] != "PRODUCTO":
                continue
            nombre = normalizar_mencion_producto(fragmento["text"], parser)
            if nombre not in nombres_menu:
                continue

            inicio = fragmento["start"]
            fin = fragmento["end"]
            etiquetas_producto[inicio] = "B-PRODUCTO"
            for posicion in range(inicio + 1, fin):
                etiquetas_producto[posicion] = "I-PRODUCTO"

        preparados.append(
            {
                "tokens": tokens,
                "etiquetas": etiquetas_producto,
            }
        )
    return preparados


def generar_ejemplos_alias(ontologia: Dict, tokenizador: SimpleSpanishTokenizer) -> List[Dict]:
    """Crea ejemplos sintéticos BIO para los sinónimos de productos del menú.

    Estos ejemplos refuerzan la detección de alias menos frecuentes, como
    ``papitas``, sin modificar los JSON originales. Se agregan únicamente a
    entrenamiento; validación conserva sus ejemplos originales.
    """
    ejemplos = []
    alias_por_producto = {}
    for grupo in ontologia.get("categorias", {}).values():
        for producto in grupo:
            nombre = producto["nombre"]
            alias_por_producto.setdefault(nombre, set()).add(nombre)
            alias_por_producto[nombre].update(producto.get("sinonimos", []))

    for alias, nombre in ALIAS_PRODUCTO.items():
        if nombre in alias_por_producto:
            alias_por_producto[nombre].add(alias)

    contextos = (
        ([], []),
        (["quiero"], []),
        (["quisiera"], ["por", "favor"]),
        (["me", "das"], []),
    )
    for alias_por_nombre in alias_por_producto.values():
        for alias in sorted(alias_por_nombre):
            tokens_alias = tokenizador.tokenize(alias)
            if not tokens_alias:
                continue
            for prefijo, sufijo in contextos:
                tokens = list(prefijo) + tokens_alias + list(sufijo)
                etiquetas = ["O"] * len(tokens)
                inicio = len(prefijo)
                etiquetas[inicio] = "B-PRODUCTO"
                for posicion in range(inicio + 1, inicio + len(tokens_alias)):
                    etiquetas[posicion] = "I-PRODUCTO"
                ejemplos.append({"tokens": tokens, "etiquetas": etiquetas})
    return ejemplos


class ConjuntoProductos(Dataset):
    """Codifica tokens y etiquetas BIO para entrenar o validar la CNN."""

    def __init__(self, registros: List[Dict], vocabulario: Vocabulary):
        self.ejemplos = [
            (
                vocabulario.encode(registro["tokens"]),
                [ETIQUETAS_PRODUCTO[tag] for tag in registro["etiquetas"]],
            )
            for registro in registros
        ]

    def __len__(self) -> int:
        """Devuelve el número de ejemplos."""
        return len(self.ejemplos)

    def __getitem__(self, indice: int):
        """Devuelve IDs de tokens y sus etiquetas BIO."""
        return self.ejemplos[indice]


def combinar_lote(lote, indice_relleno: int):
    """Rellena tokens y etiquetas para formar tensores de longitud uniforme."""
    entradas = [
        torch.tensor(ejemplo[0], dtype=torch.long)
        for ejemplo in lote
    ]
    etiquetas = [
        torch.tensor(ejemplo[1], dtype=torch.long)
        for ejemplo in lote
    ]
    entradas_rellenadas = nn.utils.rnn.pad_sequence(
        entradas,
        batch_first=True,
        padding_value=indice_relleno,
    )
    # -100 hace que CrossEntropyLoss ignore las posiciones de relleno.
    etiquetas_rellenadas = nn.utils.rnn.pad_sequence(
        etiquetas,
        batch_first=True,
        padding_value=ETIQUETA_RELLENO,
    )
    return entradas_rellenadas, etiquetas_rellenadas


def evaluar(
    modelo: ReconocedorProductosCNN,
    cargador: DataLoader,
    dispositivo: str,
    indice_relleno: int,
):
    """Calcula exactitud por token y F1 por mención de producto.

    Returns:
        Métricas, etiquetas verdaderas y etiquetas predichas en formato BIO.
    """
    modelo.eval()
    verdaderas_por_texto = []
    predichas_por_texto = []
    verdaderas_por_token = []
    predichas_por_token = []

    with torch.no_grad():
        for entradas, etiquetas in cargador:
            entradas = entradas.to(dispositivo)
            logits = modelo(entradas)  # [B, L, 3].
            predicciones = logits.argmax(dim=-1).cpu()
            etiquetas = etiquetas.cpu()

            for fila in range(entradas.size(0)):
                longitud = int(entradas[fila].ne(indice_relleno).sum().item())
                etiquetas_reales = etiquetas[fila, :longitud].tolist()
                etiquetas_predichas = predicciones[fila, :longitud].tolist()
                verdaderas_por_texto.append(
                    [INDICE_A_ETIQUETA[indice] for indice in etiquetas_reales]
                )
                predichas_por_texto.append(
                    [INDICE_A_ETIQUETA[indice] for indice in etiquetas_predichas]
                )
                verdaderas_por_token.extend(etiquetas_reales)
                predichas_por_token.extend(etiquetas_predichas)

    exactitud = accuracy_score(verdaderas_por_token, predichas_por_token)
    f1_mencion = f1_score(verdaderas_por_texto, predichas_por_texto)
    return (
        exactitud,
        f1_mencion,
        verdaderas_por_texto,
        predichas_por_texto,
    )


def guardar_graficas(historial: Dict[str, List[float]], carpeta_salida: str):
    """Guarda la pérdida de entrenamiento y las métricas de validación."""
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
    eje_perdida.set_ylabel("Cross-Entropy")
    eje_perdida.grid(True, alpha=0.3)
    eje_perdida.legend()

    eje_metricas.plot(
        epocas,
        historial["exactitud_validacion"],
        label="Exactitud por token",
    )
    eje_metricas.plot(
        epocas,
        historial["f1_mencion_validacion"],
        label="F1 de menciones",
    )
    eje_metricas.set_title("Métricas de validación")
    eje_metricas.set_xlabel("Época")
    eje_metricas.set_ylabel("Puntuación")
    eje_metricas.set_ylim(0, 1)
    eje_metricas.grid(True, alpha=0.3)
    eje_metricas.legend()

    figura.tight_layout()
    ruta = os.path.join(carpeta_salida, "graficas_cnn.png")
    figura.savefig(ruta, dpi=150)
    plt.close(figura)
    print(f"Gráficas guardadas en: {ruta}")


def entrenar(
    carpeta_datos: str,
    carpeta_salida: str,
    ruta_ontologia: str,
    epocas: int = 12,
    tamano_lote: int = 32,
    tasa_aprendizaje: float = 0.001,
    dispositivo: str = "cpu",
):
    """Entrena el detector BIO de productos y guarda sus artefactos aislados."""
    os.makedirs(carpeta_salida, exist_ok=True)
    tokenizador = SimpleSpanishTokenizer()
    ontologia, parser, nombres_menu = cargar_ontologia(ruta_ontologia)

    with open(
        os.path.join(carpeta_datos, "train.json"),
        "r",
        encoding="utf-8",
    ) as archivo:
        entrenamiento = preparar_registros(
            json.load(archivo),
            tokenizador,
            parser,
            nombres_menu,
        )
    with open(
        os.path.join(carpeta_datos, "val.json"),
        "r",
        encoding="utf-8",
    ) as archivo:
        validacion = preparar_registros(
            json.load(archivo),
            tokenizador,
            parser,
            nombres_menu,
        )

    if not entrenamiento or not validacion:
        raise ValueError("Se requieren ejemplos anotados para train y val.")

    ejemplos_alias = generar_ejemplos_alias(ontologia, tokenizador)
    entrenamiento.extend(ejemplos_alias)
    ejemplos_con_producto = sum(
        "B-PRODUCTO" in registro["etiquetas"]
        for registro in entrenamiento
    )
    print(
        f"Ejemplos: {len(entrenamiento)} entrenamiento, "
        f"{len(validacion)} validación; "
        f"{ejemplos_con_producto} contienen productos del menú "
        f"({len(ejemplos_alias)} ejemplos de alias añadidos)."
    )

    vocabulario = Vocabulary()
    vocabulario.build_vocab(
        [registro["tokens"] for registro in entrenamiento],
        min_freq=1,
    )
    vocabulario.save(os.path.join(carpeta_salida, "vocabulario_cnn.json"))
    with open(
        os.path.join(carpeta_salida, "etiquetas_cnn.json"),
        "w",
        encoding="utf-8",
    ) as archivo:
        json.dump(ETIQUETAS_PRODUCTO, archivo, ensure_ascii=False, indent=2)

    conjunto_entrenamiento = ConjuntoProductos(entrenamiento, vocabulario)
    conjunto_validacion = ConjuntoProductos(validacion, vocabulario)
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

    modelo = ReconocedorProductosCNN(
        vocab_size=len(vocabulario),
        embedding_dim=128,
        num_tags=len(ETIQUETAS_PRODUCTO),
        filtros_por_kernel=100,
        tamanos_kernel=(3, 4, 5),
        dropout=0.3,
        pad_idx=vocabulario.pad_idx,
    ).to(dispositivo)
    funcion_perdida = nn.CrossEntropyLoss(ignore_index=ETIQUETA_RELLENO)
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
        "f1_mencion_validacion": [],
    }

    print("\n--- ENTRENAMIENTO CNN PARA PRODUCTOS ---")
    for epoca in range(1, epocas + 1):
        modelo.train()
        perdida_total = 0.0
        for entradas, etiquetas in cargador_entrenamiento:
            entradas = entradas.to(dispositivo)
            etiquetas = etiquetas.to(dispositivo)

            optimizador.zero_grad()
            logits = modelo(entradas)  # [B, L, 3].
            perdida = funcion_perdida(
                logits.reshape(-1, len(ETIQUETAS_PRODUCTO)),
                etiquetas.reshape(-1),
            )
            perdida.backward()
            nn.utils.clip_grad_norm_(modelo.parameters(), max_norm=5.0)
            optimizador.step()
            perdida_total += perdida.item()

        perdida_media = perdida_total / len(cargador_entrenamiento)
        exactitud, f1_mencion, _, _ = evaluar(
            modelo,
            cargador_validacion,
            dispositivo,
            vocabulario.pad_idx,
        )
        planificador.step(f1_mencion)
        historial["perdida_entrenamiento"].append(perdida_media)
        historial["exactitud_validacion"].append(exactitud)
        historial["f1_mencion_validacion"].append(f1_mencion)

        print(
            f"Época {epoca:02d}/{epocas:02d} | "
            f"Train Loss: {perdida_media:.4f} | "
            f"Val Acc por token: {exactitud * 100:.2f}% | "
            f"F1 de menciones: {f1_mencion:.4f}"
        )
        if f1_mencion > mejor_f1:
            mejor_f1 = f1_mencion
            torch.save(modelo.state_dict(), ruta_pesos)

    guardar_graficas(historial, carpeta_salida)
    print(f"Mejor F1 de menciones: {mejor_f1:.4f}")
    print(f"Pesos guardados en: {ruta_pesos}")

    modelo.load_state_dict(torch.load(ruta_pesos, map_location=dispositivo))
    _, _, verdaderas, predichas = evaluar(
        modelo,
        cargador_validacion,
        dispositivo,
        vocabulario.pad_idx,
    )
    print("\n--- REPORTE DE VALIDACIÓN POR MENCIÓN ---")
    print(classification_report(verdaderas, predichas, digits=4))


def leer_argumentos():
    """Lee rutas y parámetros del entrenamiento desde la línea de comandos."""
    analizador = argparse.ArgumentParser(
        description="Entrena una CNN para identificar productos del menú."
    )
    analizador.add_argument(
        "--carpeta-datos",
        default=os.path.join(RAIZ_PROYECTO, "data", "dataset"),
        help="Carpeta con train.json y val.json.",
    )
    analizador.add_argument(
        "--carpeta-salida",
        default=os.path.join(CARPETA_CNN, "salidas_productos"),
        help="Carpeta nueva para no sobrescribir los antiguos pesos CNN.",
    )
    analizador.add_argument(
        "--ruta-ontologia",
        default=os.path.join(RAIZ_PROYECTO, "data", "menu_ontology.json"),
        help="Archivo del menú usado para filtrar y normalizar productos.",
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
        ruta_ontologia=argumentos.ruta_ontologia,
        epocas=argumentos.epocas,
        tamano_lote=argumentos.tamano_lote,
        tasa_aprendizaje=argumentos.tasa_aprendizaje,
        dispositivo=dispositivo,
    )
