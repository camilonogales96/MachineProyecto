import os
import json
import argparse

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROYECTO_DIR = os.path.dirname(BASE_DIR)
SALIDAS_DIR = os.path.join(BASE_DIR, "salidas")
DATOS_DIR = os.path.join(PROYECTO_DIR, "data", "dataset")


def cargar_dataset():
    registros = []
    for nombre in ("train.json", "val.json", "test.json"):
        ruta = os.path.join(DATOS_DIR, nombre)
        if not os.path.exists(ruta):
            continue
        with open(ruta, "r", encoding="utf-8") as archivo:
            registros.extend(json.load(archivo))
    return [r for r in registros if str(r.get("texto", "")).strip() and str(r.get("comanda", "")).strip()]


_registros = None
_vectorizador = None
_matriz_dataset = None


def cargar_busqueda():
    global _registros, _vectorizador, _matriz_dataset
    if _matriz_dataset is None:
        _registros = cargar_dataset()
        textos = [str(r["texto"]).lower() for r in _registros]
        _vectorizador = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True)
        _matriz_dataset = _vectorizador.fit_transform(textos)


def buscar_comanda(texto, umbral=0.35):
    cargar_busqueda()
    consulta = _vectorizador.transform([texto.lower().strip()])
    similitudes = cosine_similarity(consulta, _matriz_dataset)[0]
    indice = int(np.argmax(similitudes))
    puntuacion = float(similitudes[indice])
    if puntuacion >= umbral:
        return str(_registros[indice]["comanda"]), puntuacion
    return None, puntuacion


def cargar_json(nombre):
    with open(os.path.join(SALIDAS_DIR, nombre), "r", encoding="utf-8") as archivo:
        return json.load(archivo)


def cargar_recursos_lstm():
    metadata = cargar_json("metadata_lstm.json")
    modelo = load_model(os.path.join(SALIDAS_DIR, "modelo_lstm.keras"))
    tokenizador_entrada = tokenizer_from_json(cargar_json("tokenizador_entrada.json"))
    tokenizador_salida = tokenizer_from_json(cargar_json("tokenizador_salida.json"))
    return metadata, modelo, tokenizador_entrada, tokenizador_salida


def predecir(texto, recursos_lstm=None):
    comanda, confianza = buscar_comanda(texto)
    if comanda is not None:
        return comanda

    if recursos_lstm is None:
        recursos_lstm = cargar_recursos_lstm()

    metadata, modelo, tokenizador_entrada, tokenizador_salida = recursos_lstm
    secuencia = tokenizador_entrada.texts_to_sequences([texto.lower().strip()])
    entrada = pad_sequences(secuencia, maxlen=metadata["max_entrada"], padding="post")
    prediccion = modelo.predict(entrada, verbose=0)
    indices = np.argmax(prediccion[0], axis=-1)

    palabras = []
    for indice in indices:
        palabra = tokenizador_salida.index_word.get(int(indice), "")
        if palabra in ("<inicio>", "<unk>", ""):
            continue
        if palabra == "<fin>":
            break
        palabras.append(palabra)
    return " ".join(palabras)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--texto", type=str, help="Texto para predecir")
    args = parser.parse_args()

    if args.texto:
        print(predecir(args.texto))
    else:
        while True:
            pedido = input("Pedido: ").strip()
            if pedido.lower() in {"salir", "exit", "quit", "q"}:
                break
            if pedido:
                print("Predicción:", predecir(pedido))
