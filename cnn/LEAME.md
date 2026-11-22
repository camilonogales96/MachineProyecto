# Reconocimiento de productos del menú con Text-CNN

Esta carpeta contiene un reconocedor independiente de productos. No modifica
los modelos ni los archivos de preprocesamiento existentes. Reutiliza el
tokenizador, el vocabulario, las anotaciones del dataset y la ontología del menú.

La CNN ya no clasifica intenciones completas como `pedido` o `consulta`. Su
objetivo es localizar una o varias menciones de productos dentro del mensaje,
por ejemplo detectar `papitas` y normalizarla como `Papas Fritas`.

## Arquitectura

1. Embedding de 128 dimensiones, aprendido desde cero.
2. Tres convoluciones 1D paralelas con 100 filtros cada una y kernels 3, 4 y 5.
3. ReLU en cada rama y concatenación de sus características por posición.
4. Dropout de 0.3.
5. Capa lineal que predice una etiqueta BIO para cada token: `O`,
   `B-PRODUCTO` o `I-PRODUCTO`.

Se conserva la dimensión temporal en la salida. No se usa max pooling global,
porque se necesitan las posiciones de los tokens para recortar las menciones.
Las predicciones se normalizan con los sinónimos de `data\menu_ontology.json`.
Los alias `papita` y `papitas` se normalizan a `Papas Fritas`.

El catálogo objetivo incluye los 13 productos de la ontología, incluidas las
bebidas. Las anotaciones de cantidad, ingredientes, exclusiones e intención se
ignoran durante el entrenamiento de esta CNN.

## Preparación

Abre PowerShell en la raíz del proyecto (`C:\MachineProyecto`) y activa el
entorno virtual:

```powershell
.\venv\Scripts\Activate.ps1
```

El dataset debe tener `data\dataset\train.json` y `data\dataset\val.json`, con
los campos `tokens` y `bio_tags`. El entrenamiento transforma las etiquetas
existentes y conserva únicamente los fragmentos BIO `PRODUCTO` que se pueden
asociar con un producto del menú.

## Entrenamiento

```powershell
python cnn\entrenar_cnn.py
```

El script informa la pérdida, la exactitud por token y el F1 de menciones.
Guarda sus nuevos archivos en `cnn\salidas_productos`:

- `pesos_cnn.pt`: pesos del mejor modelo según F1 de menciones.
- `vocabulario_cnn.json`: vocabulario construido solamente con `train.json`.
- `etiquetas_cnn.json`: correspondencia entre etiquetas BIO e índices.
- `graficas_cnn.png`: curvas de pérdida y métricas.

Se pueden cambiar los parámetros desde la consola:

```powershell
python cnn\entrenar_cnn.py --epocas 12 --tamano-lote 32
```

Se usa una carpeta de salida nueva para no reemplazar archivos creados por la
versión anterior que clasificaba intenciones.

## Probar mensajes

Después del entrenamiento, abre el modo interactivo:

```powershell
python cnn\prediccion_cnn.py
```

Escribe, por ejemplo:

```text
Quiero papitas y una Coca-Cola
```

La salida muestra las menciones encontradas y sus nombres canónicos. Para
terminar, escribe `salir`.

También se puede predecir una frase directamente:

```powershell
python cnn\prediccion_cnn.py --texto "Quiero papitas y una Coca-Cola"
```

Esta herramienta identifica productos solamente. No predice cantidades,
modificaciones ni una intención conversacional.
