import os
import json
import random
import argparse

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, RepeatVector, Dense, TimeDistributed
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


PROYECTO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATOS_DIR = os.path.join(PROYECTO, "data", "dataset")
SALIDAS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "salidas")


def cargar_split(nombre):
    ruta = os.path.join(DATOS_DIR, f"{nombre}.json")
    with open(ruta, "r", encoding="utf-8") as archivo:
        registros = json.load(archivo)

    entradas, objetivos = [], []
    for registro in registros:
        texto = str(registro.get("texto", "")).strip()
        comanda = str(registro.get("comanda", "")).strip()
        if texto and comanda:
            entradas.append(texto.lower())
            objetivos.append(f"<inicio> {comanda.lower()} <fin>")
    return entradas, objetivos


def main():
    parser = argparse.ArgumentParser(description="Entrena un modelo LSTM aislado con el dataset compartido")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    random.seed(42)
    np.random.seed(42)
    tf.random.set_seed(42)

    entrenar_x, entrenar_y = cargar_split("train")
    validar_x, validar_y = cargar_split("val")

    if not entrenar_x:
        raise RuntimeError("No se encontraron ejemplos con texto y comanda en train.json")

    tokenizador_entrada = Tokenizer(oov_token="<unk>")
    tokenizador_entrada.fit_on_texts(entrenar_x)

    tokenizador_salida = Tokenizer(oov_token="<unk>")
    tokenizador_salida.fit_on_texts(entrenar_y)

    seq_entrada = tokenizador_entrada.texts_to_sequences(entrenar_x)
    seq_salida = tokenizador_salida.texts_to_sequences(entrenar_y)

    seq_validar_entrada = tokenizador_entrada.texts_to_sequences(validar_x)
    seq_validar_salida = tokenizador_salida.texts_to_sequences(validar_y)

    max_entrada = min(160, max(len(secuencia) for secuencia in seq_entrada))
    max_salida = min(80, max(len(secuencia) for secuencia in seq_salida))

    x_entrenamiento = pad_sequences(seq_entrada, maxlen=max_entrada, padding="post")
    y_entrenamiento = pad_sequences(seq_salida, maxlen=max_salida, padding="post")

    x_validacion = pad_sequences(seq_validar_entrada, maxlen=max_entrada, padding="post")
    y_validacion = pad_sequences(seq_validar_salida, maxlen=max_salida, padding="post")

    vocabulario_entrada = len(tokenizador_entrada.word_index) + 1
    vocabulario_salida = len(tokenizador_salida.word_index) + 1

    modelo = Sequential([
        Embedding(input_dim=vocabulario_entrada, output_dim=64),
        LSTM(128),
        RepeatVector(max_salida),
        LSTM(128, return_sequences=True),
        TimeDistributed(Dense(vocabulario_salida, activation="softmax"))
    ])

    modelo.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    os.makedirs(SALIDAS_DIR, exist_ok=True)

    modelo.fit(
        x_entrenamiento,
        np.expand_dims(y_entrenamiento, -1),
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_data=(x_validacion, np.expand_dims(y_validacion, -1)),
        callbacks=[EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True)],
        verbose=1
    )

    modelo.save(os.path.join(SALIDAS_DIR, "modelo_lstm.keras"))

    with open(os.path.join(SALIDAS_DIR, "tokenizador_entrada.json"), "w", encoding="utf-8") as archivo:
        json.dump(tokenizador_entrada.to_json(), archivo, ensure_ascii=False)

    with open(os.path.join(SALIDAS_DIR, "tokenizador_salida.json"), "w", encoding="utf-8") as archivo:
        json.dump(tokenizador_salida.to_json(), archivo, ensure_ascii=False)

    metadata = {
        "max_entrada": max_entrada,
        "max_salida": max_salida,
        "ejemplos_entrenamiento": len(entrenar_x),
        "ejemplos_validacion": len(validar_x),
        "dataset": ["train.json", "val.json"]
    }
    with open(os.path.join(SALIDAS_DIR, "metadata_lstm.json"), "w", encoding="utf-8") as archivo:
        json.dump(metadata, archivo, ensure_ascii=False, indent=2)

    print(f"Modelo guardado en: {SALIDAS_DIR}")


if __name__ == "__main__":
    main()
