"""Reconoce y normaliza menciones de productos usando el modelo Text-CNN."""

import argparse
import json
import os
import sys
from typing import Dict, List, Optional

import torch


CARPETA_CNN = os.path.dirname(os.path.abspath(__file__))
RAIZ_PROYECTO = os.path.dirname(CARPETA_CNN)
if RAIZ_PROYECTO not in sys.path:
    sys.path.insert(0, RAIZ_PROYECTO)

from cnn.modelo_cnn import (
    ETIQUETAS_PRODUCTO,
    ReconocedorProductosCNN,
    normalizar_mencion_producto,
)
from src.preprocessing.tokenizer import SimpleSpanishTokenizer
from src.preprocessing.vocabulary import Vocabulary
from src.structuring.parser import OrderParser


def extraer_menciones(
    tokens: List[str],
    etiquetas: List[str],
    parser: OrderParser,
    nombres_menu: set,
) -> List[Dict[str, str]]:
    """Agrupa etiquetas BIO consecutivas y normaliza los productos reconocidos.

    Se ignoran fragmentos cuyo nombre no corresponda a un producto del menú.
    Una etiqueta ``I-PRODUCTO`` sin una etiqueta ``B-PRODUCTO`` previa se toma
    como el inicio de un fragmento para tolerar errores de predicción.
    """
    menciones = []
    inicio = None

    def cerrar_fragmento(fin: int):
        if inicio is None or fin <= inicio:
            return
        mencion = " ".join(tokens[inicio:fin])
        nombre_canonico = normalizar_mencion_producto(mencion, parser)
        if nombre_canonico in nombres_menu:
            menciones.append(
                {
                    "mencion": mencion,
                    "producto_normalizado": nombre_canonico,
                }
            )

    for posicion, etiqueta in enumerate(etiquetas):
        if etiqueta == "B-PRODUCTO":
            cerrar_fragmento(posicion)
            inicio = posicion
        elif etiqueta == "I-PRODUCTO":
            if inicio is None:
                inicio = posicion
        else:
            cerrar_fragmento(posicion)
            inicio = None

    cerrar_fragmento(len(tokens))
    return menciones


class IdentificadorProductosCNN:
    """Carga la CNN y transforma un mensaje en productos del menú detectados."""

    def __init__(
        self,
        carpeta_salida: Optional[str] = None,
        ruta_ontologia: Optional[str] = None,
        dispositivo: str = "cpu",
    ):
        """Carga vocabulario, modelo y ontología, sin usar el clasificador de intención.

        Args:
            carpeta_salida: carpeta con los artefactos generados por entrenar_cnn.
            ruta_ontologia: menú con los nombres canónicos y sus sinónimos.
            dispositivo: dispositivo de PyTorch, ``cpu`` o ``cuda``.
        """
        self.dispositivo = dispositivo
        self.carpeta_salida = carpeta_salida or os.path.join(
            CARPETA_CNN,
            "salidas_productos",
        )
        ruta_ontologia = ruta_ontologia or os.path.join(
            RAIZ_PROYECTO,
            "data",
            "menu_ontology.json",
        )

        ruta_pesos = os.path.join(self.carpeta_salida, "pesos_cnn.pt")
        ruta_vocabulario = os.path.join(
            self.carpeta_salida,
            "vocabulario_cnn.json",
        )
        artefactos = (ruta_pesos, ruta_vocabulario)
        faltantes = [ruta for ruta in artefactos if not os.path.isfile(ruta)]
        if faltantes:
            raise FileNotFoundError(
                "No se encontraron los artefactos del reconocedor de productos. "
                "Ejecuta primero 'python cnn\\entrenar_cnn.py'. Faltan: "
                + ", ".join(faltantes)
            )

        self.vocabulario = Vocabulary.load(ruta_vocabulario)
        with open(ruta_ontologia, "r", encoding="utf-8") as archivo:
            ontologia = json.load(archivo)
        self.parser = OrderParser(ontologia)
        self.nombres_menu = {
            producto["nombre"]
            for grupo in ontologia.get("categorias", {}).values()
            for producto in grupo
        }
        self.tokenizador = SimpleSpanishTokenizer()

        self.modelo = ReconocedorProductosCNN(
            vocab_size=len(self.vocabulario),
            embedding_dim=128,
            num_tags=len(ETIQUETAS_PRODUCTO),
            filtros_por_kernel=100,
            tamanos_kernel=(3, 4, 5),
            dropout=0.3,
            pad_idx=self.vocabulario.pad_idx,
        ).to(self.dispositivo)
        self.modelo.load_state_dict(
            torch.load(ruta_pesos, map_location=self.dispositivo)
        )
        self.modelo.eval()

    def predecir(self, texto: str) -> Dict[str, List[Dict[str, str]]]:
        """Devuelve las menciones y nombres del menú detectados en el texto.

        La salida solo identifica productos; no predice intención, cantidad ni
        modificaciones del pedido.
        """
        tokens = self.tokenizador.tokenize(texto)
        if not tokens:
            return {"productos_identificados": []}

        entrada = torch.tensor(
            [self.vocabulario.encode(tokens)],
            dtype=torch.long,
            device=self.dispositivo,
        )
        with torch.no_grad():
            predicciones = self.modelo.predict(entrada)[0].cpu().tolist()

        etiqueta_por_indice = {
            indice: etiqueta
            for etiqueta, indice in ETIQUETAS_PRODUCTO.items()
        }
        etiquetas = [
            etiqueta_por_indice[indice]
            for indice in predicciones
        ]
        productos = extraer_menciones(
            tokens,
            etiquetas,
            self.parser,
            self.nombres_menu,
        )
        return {"productos_identificados": productos}


def leer_argumentos():
    """Lee una frase y las rutas opcionales para realizar predicciones."""
    analizador = argparse.ArgumentParser(
        description="Identifica productos del menú usando una CNN."
    )
    analizador.add_argument(
        "--texto",
        help="Procesa una sola frase; si se omite, abre el modo interactivo.",
    )
    analizador.add_argument(
        "--carpeta-salida",
        default=os.path.join(CARPETA_CNN, "salidas_productos"),
        help="Carpeta donde entrenar_cnn guardó los pesos y vocabulario.",
    )
    analizador.add_argument(
        "--ruta-ontologia",
        default=os.path.join(RAIZ_PROYECTO, "data", "menu_ontology.json"),
        help="Archivo del menú para normalizar nombres de productos.",
    )
    return analizador.parse_args()


def mostrar_resultado(resultado: Dict[str, List[Dict[str, str]]]):
    """Imprime las menciones reconocidas y su nombre canónico del menú."""
    productos = resultado["productos_identificados"]
    if productos:
        nombres = [producto["producto_normalizado"] for producto in productos]
        print("Productos identificados: " + ", ".join(nombres))
    else:
        print("No se reconocieron productos del menú.")
    print(json.dumps(resultado, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    argumentos = leer_argumentos()
    dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
    identificador = IdentificadorProductosCNN(
        carpeta_salida=argumentos.carpeta_salida,
        ruta_ontologia=argumentos.ruta_ontologia,
        dispositivo=dispositivo,
    )

    if argumentos.texto:
        mostrar_resultado(identificador.predecir(argumentos.texto))
    else:
        print("Escribe un mensaje; escribe 'salir' para terminar.")
        while True:
            texto = input("\nMensaje: ").strip()
            if texto.lower() in {"salir", "exit", "q"}:
                break
            if texto:
                mostrar_resultado(identificador.predecir(texto))
