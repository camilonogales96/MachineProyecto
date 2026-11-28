# Modelo LSTM aislado

Este módulo usa exclusivamente los archivos de `data/dataset/train.json`, `val.json` y `test.json`.
No modifica ni reutiliza directamente el pipeline Bi-GRU/CRF ni la implementación CNN de los compañeros.

## Instalación

```powershell
pip install -r lstm/requirements.txt
```

## Entrenamiento

```powershell
python lstm/entrenar_lstm.py
```

Los pesos y vocabularios se guardarán en `lstm/salidas/`.

## Predicción

```powershell
python lstm/prediccion_lstm.py
```
