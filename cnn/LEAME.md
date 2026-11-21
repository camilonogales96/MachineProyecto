# Text-CNN para clasificación y predicción de pedidos

Esta carpeta contiene una implementación independiente de Text-CNN. No modifica
los archivos existentes del proyecto. Reutiliza el tokenizador y el vocabulario
del repositorio, y guarda sus propios artefactos en `cnn/salidas`.

## Arquitectura

El clasificador de intenciones es una CNN pura:

1. Embedding de 128 dimensiones entrenado desde cero.
2. Tres convoluciones 1D, con 100 filtros y kernels de tamaño 3, 4 y 5.
3. ReLU y max pooling global temporal para cada kernel.
4. Concatenación de las 300 características.
5. Dropout de 0.3 y capa lineal de cuatro logits: `pedido`, `consulta`,
   `reclamo` y `saludo_otro`.

Para formar una comanda completa, `prediccion_cnn.py` usa la CNN para reconocer
la intención y reutiliza el NER, el extractor de relaciones y el parser
existentes. El modelo de intención sigue siendo CNN; los modelos base se usan
solo en las etapas posteriores de extracción y estructuración.

## Requisitos

Ejecuta los comandos desde la raíz del repositorio, `C:\MachineProyecto`.
Activa el entorno virtual o usa directamente su ejecutable:

```powershell
.\venv\Scripts\Activate.ps1
```

Los datos `data\dataset\train.json` y `data\dataset\val.json` deben estar
preparados. La predicción de comandas también necesita los artefactos NER
existentes en `models_saved`: `vocab.json`, `tag_vocab.json` y `ner_model.pt`.

## Entrenar la CNN

Con el entorno virtual activado:

```powershell
python cnn\entrenar_cnn.py
```

O sin activar el entorno:

```powershell
.\venv\Scripts\python.exe cnn\entrenar_cnn.py
```

El entrenamiento calcula Train Loss, Val Acc y F1-Macro, genera las gráficas y
guarda los resultados en `cnn\salidas`:

- `pesos_cnn.pt`
- `vocabulario_cnn.json`
- `intenciones_cnn.json`
- `graficas_cnn.png`

Se pueden ajustar las épocas y el tamaño de lote:

```powershell
python cnn\entrenar_cnn.py --epocas 12 --tamano-lote 32
```

## Probar una frase

Después de entrenar, inicia el modo interactivo:

```powershell
python cnn\prediccion_cnn.py
```

Escribe un mensaje cuando aparezca `Pedido:`. Por ejemplo:

```text
Hola buenas tardes, quisiera 2 hamburguesas dobles sin cebolla y un refresco
```

La salida contiene la intención, la comanda de cocina y el contrato JSON. Para
terminar, escribe `salir`.

También se puede procesar una sola frase desde la consola:

```powershell
python cnn\prediccion_cnn.py --texto "Quiero dos hamburguesas sin cebolla"
```

Si no existen todavía los artefactos CNN, ejecuta primero el entrenamiento.
