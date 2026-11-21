"""Predicción de intención y estructuración de pedidos con el modelo CNN."""

import argparse
import json
import os
import sys
from typing import Any, Dict, Optional

import torch


CARPETA_CNN = os.path.dirname(os.path.abspath(__file__))
RAIZ_PROYECTO = os.path.dirname(CARPETA_CNN)
if RAIZ_PROYECTO not in sys.path:
    sys.path.insert(0, RAIZ_PROYECTO)

from cnn.modelo_cnn import ClasificadorIntencionesCNN
from src.models.ner_tagger import BiGRUNERTagger
from src.models.relation_extractor import OrderRelationExtractor
from src.preprocessing.tokenizer import SimpleSpanishTokenizer
from src.preprocessing.vocabulary import TagVocabulary, Vocabulary
from src.structuring.parser import OrderParser


class PipelinePedidosCNN:
    """Usa Text-CNN para intención y reutiliza NER y parser existentes.

    La CNN reemplaza únicamente la clasificación de intención. Para obtener una
    comanda estructurada se conservan las etapas posteriores ya existentes del
    proyecto: NER, extractor de relaciones, ontología y parser.
    """

    def __init__(
        self,
        carpeta_salida: Optional[str] = None,
        carpeta_modelos_base: Optional[str] = None,
        ruta_ontologia: Optional[str] = None,
        dispositivo: str = "cpu",
    ):
        """Carga los recursos de inferencia de CNN y los recursos base de NER.

        Args:
            carpeta_salida: directorio con pesos y vocabulario CNN.
            carpeta_modelos_base: directorio con los pesos y vocabularios NER.
            ruta_ontologia: archivo JSON con menú y sinónimos.
            dispositivo: dispositivo de PyTorch, normalmente ``cpu`` o ``cuda``.
        """
        self.dispositivo = dispositivo
        self.carpeta_salida = carpeta_salida or os.path.join(
            CARPETA_CNN,
            "salidas",
        )
        self.carpeta_modelos_base = carpeta_modelos_base or os.path.join(
            RAIZ_PROYECTO,
            "models_saved",
        )
        ruta_ontologia = ruta_ontologia or os.path.join(
            RAIZ_PROYECTO,
            "data",
            "menu_ontology.json",
        )

        with open(ruta_ontologia, "r", encoding="utf-8") as archivo:
            ontologia = json.load(archivo)
        self.tokenizador = SimpleSpanishTokenizer()
        self.extractor_relaciones = OrderRelationExtractor(ontologia)
        self.parser = OrderParser(ontologia)

        ruta_pesos_cnn = os.path.join(self.carpeta_salida, "pesos_cnn.pt")
        ruta_vocabulario_cnn = os.path.join(
            self.carpeta_salida,
            "vocabulario_cnn.json",
        )
        ruta_intenciones = os.path.join(
            self.carpeta_salida,
            "intenciones_cnn.json",
        )
        archivos_cnn = (
            ruta_pesos_cnn,
            ruta_vocabulario_cnn,
            ruta_intenciones,
        )
        faltantes = [ruta for ruta in archivos_cnn if not os.path.isfile(ruta)]
        if faltantes:
            raise FileNotFoundError(
                "No se encontraron los artefactos de la CNN. Ejecuta primero "
                "'python cnn\\entrenar_cnn.py'. Faltan: "
                + ", ".join(faltantes)
            )

        self.vocabulario_cnn = Vocabulary.load(ruta_vocabulario_cnn)
        with open(ruta_intenciones, "r", encoding="utf-8") as archivo:
            mapa_intenciones = json.load(archivo)
        self.intencion_por_indice = {
            int(indice): intencion
            for intencion, indice in mapa_intenciones.items()
        }

        self.modelo_intenciones = ClasificadorIntencionesCNN(
            vocab_size=len(self.vocabulario_cnn),
            embedding_dim=128,
            num_classes=len(mapa_intenciones),
            filtros_por_kernel=100,
            tamanos_kernel=(3, 4, 5),
            dropout=0.3,
            pad_idx=self.vocabulario_cnn.pad_idx,
        ).to(self.dispositivo)
        self.modelo_intenciones.load_state_dict(
            torch.load(ruta_pesos_cnn, map_location=self.dispositivo)
        )
        self.modelo_intenciones.eval()

        # El NER conserva su vocabulario y sus pesos originales.
        self.vocabulario_ner = Vocabulary.load(
            os.path.join(self.carpeta_modelos_base, "vocab.json")
        )
        self.vocabulario_etiquetas = TagVocabulary.load(
            os.path.join(self.carpeta_modelos_base, "tag_vocab.json")
        )
        self.modelo_ner = BiGRUNERTagger(
            vocab_size=len(self.vocabulario_ner),
            num_tags=len(self.vocabulario_etiquetas),
            embedding_dim=128,
            hidden_dim=128,
            num_layers=2,
            dropout=0.0,
            pad_idx=self.vocabulario_ner.pad_idx,
        ).to(self.dispositivo)
        ruta_pesos_ner = os.path.join(
            self.carpeta_modelos_base,
            "ner_model.pt",
        )
        self.modelo_ner.load_state_dict(
            torch.load(ruta_pesos_ner, map_location=self.dispositivo)
        )
        self.modelo_ner.eval()

    def procesar(self, texto: str) -> Dict[str, Any]:
        """Clasifica un texto y devuelve la comanda con formato de la pipeline.

        Args:
            texto: mensaje libre del cliente en español.

        Returns:
            Diccionario con texto, intención, tokens, etiquetas BIO, contrato
            JSON y línea sintetizada para cocina. Si la intención no es pedido,
            devuelve el marcador de esa intención y un contrato sin artículos.
        """
        tokens = self.tokenizador.tokenize(texto)
        if not tokens:
            return {
                "texto_original": texto,
                "intencion": "saludo_otro",
                "tokens": [],
                "bio_tags": [],
                "contrato_json": {"intencion": "saludo_otro", "items": []},
                "comanda_cocina": "[VACIO]",
            }

        entrada_cnn = torch.tensor(
            [self.vocabulario_cnn.encode(tokens)],
            dtype=torch.long,
            device=self.dispositivo,
        )
        with torch.no_grad():
            indice_intencion = int(
                self.modelo_intenciones.predict(entrada_cnn).item()
            )
        intencion = self.intencion_por_indice[indice_intencion]

        if intencion != "pedido":
            etiquetas_bio = ["O"] * len(tokens)
            contrato = {
                "intencion": intencion,
                "items": [],
                "restricciones_globales": [],
                "notas": [],
            }
            return {
                "texto_original": texto,
                "intencion": intencion,
                "tokens": tokens,
                "bio_tags": etiquetas_bio,
                "contrato_json": contrato,
                "comanda_cocina": f"[{intencion.upper()}]",
            }

        entrada_ner = torch.tensor(
            [self.vocabulario_ner.encode(tokens)],
            dtype=torch.long,
            device=self.dispositivo,
        )
        rutas = self.modelo_ner.decode(entrada_ner)
        etiquetas_bio = self.vocabulario_etiquetas.decode(rutas[0])
        etiquetas_bio = etiquetas_bio[: len(tokens)]
        if len(etiquetas_bio) < len(tokens):
            etiquetas_bio.extend(["O"] * (len(tokens) - len(etiquetas_bio)))

        elementos = self.extractor_relaciones.extract_relations(
            tokens,
            etiquetas_bio,
        )
        contrato = self.parser.build_contract_json(intencion, elementos)
        comanda = self.parser.format_kitchen_line(contrato)

        return {
            "texto_original": texto,
            "intencion": intencion,
            "tokens": tokens,
            "bio_tags": etiquetas_bio,
            "contrato_json": contrato,
            "comanda_cocina": comanda,
        }


def leer_argumentos():
    """Lee texto y rutas opcionales para la predicción.

    Si no se entrega ``--texto``, el programa inicia un ciclo interactivo.
    """
    analizador = argparse.ArgumentParser(
        description="Predice y estructura pedidos con Text-CNN."
    )
    analizador.add_argument(
        "--texto",
        help="Procesa una frase y termina; si se omite, inicia modo interactivo.",
    )
    analizador.add_argument(
        "--carpeta-salida",
        default=os.path.join(CARPETA_CNN, "salidas"),
        help="Carpeta con los pesos y el vocabulario de la CNN.",
    )
    analizador.add_argument(
        "--carpeta-modelos-base",
        default=os.path.join(RAIZ_PROYECTO, "models_saved"),
        help="Carpeta con los pesos y vocabularios NER existentes.",
    )
    return analizador.parse_args()


def mostrar_resultado(resultado: Dict[str, Any]):
    """Imprime la salida estructurada en un formato legible.

    Args:
        resultado: diccionario devuelto por ``PipelinePedidosCNN.procesar``.
    """
    print(f"Intención: {resultado['intencion']}")
    print(f"Comanda Cocina: {resultado['comanda_cocina']}")
    print(
        "Contrato JSON: "
        + json.dumps(
            resultado["contrato_json"],
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    argumentos = leer_argumentos()
    dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline = PipelinePedidosCNN(
        carpeta_salida=argumentos.carpeta_salida,
        carpeta_modelos_base=argumentos.carpeta_modelos_base,
        dispositivo=dispositivo,
    )

    if argumentos.texto:
        mostrar_resultado(pipeline.procesar(argumentos.texto))
    else:
        print("Escribe un pedido; escribe 'salir' para terminar.")
        while True:
            texto = input("\nPedido: ").strip()
            if texto.lower() in {"salir", "exit", "q"}:
                break
            if texto:
                mostrar_resultado(pipeline.procesar(texto))
