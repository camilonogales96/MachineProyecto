# Sistema de Extracción y Síntesis de Pedidos de Restaurante

Sistema de Procesamiento de Lenguaje Natural (NLP) y Deep Learning desarrollado **100% desde cero (From Scratch)**, sin modelos pre-entrenados ni recurrencias clásicas (sin RNN tradicional, sin LSTM).

Convierte mensajes de clientes con cortesías y rodeos conversacionales en **Comandas Sintetizadas para Cocina** y un **Contrato JSON Estructurado**.

---

## 1. Arquitectura del Sistema

```
Texto Libre del Cliente
   │
   ▼
[Preprocesamiento Propio: Normalización NFC, Tokenizador con Offsets, Vocabulario del Dominio]
   │
   ▼
┌────────────────────────────────────────────────────────┐
│ ETAPA 1 · Clasificación de Intención                   │
│ Embedding Propio (from scratch)                        │
│ └── Bi-GRU (2 capas)                                   │
│     └── Masked Mean-Pooling                            │
│         └── Dropout (0.3)                              │
│             └── Capa Densa/Lineal -> Softmax (4 clases)│
│ Clases: pedido | consulta | reclamo | saludo_otro      │
└──────────────────────────┬─────────────────────────────┘
                           ▼ (Si intención == "pedido")
┌────────────────────────────────────────────────────────┐
│ ETAPA 2 · NER + Extracción de Atributos                │
│ Embedding Propio (from scratch)                        │
│ └── Bi-GRU Bidireccional (2 capas)                     │
│     └── Capa Densa de Emisión                          │
│         └── Linear-Chain CRF Nativo (Viterbi Decoding) │
│ Entidades BIO: PRODUCTO, CANTIDAD, UNIDAD, INGREDIENTE,│
│ MODIFICACION, NEGACION, EXCLUSION, SUST_DE, SUST_A     │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│ ETAPA 3 · Estructuración y Normalización               │
│ 1. Extractor de Relaciones y Ámbitos Sintácticos       │
│ 2. Normalización de Sinónimos contra Ontología Menú   │
│ 3. Generación de Contrato JSON (Fuente de Verdad)      │
│ 4. Generación de Línea Sintetizada para Cocina         │
└──────────────────────────┬─────────────────────────────┘
                           ▼
           Salida de Cocina & Contrato JSON
```

---

## 2. Estructura del Repositorio

```text
MachineProyecto/
├── data/
│   ├── menu_ontology.json          # Menú estructurado, sinónimos, ingredientes y exclusiones
│   ├── clean_datasets.py           # Limpieza y preparación de lotes y recetas
│   ├── generator.py                # Generador sintético de datos con Data Augmentation
│   └── dataset/
│       ├── dataset_pedidos_lote1.csv # Lote 1 con dialectos y etiquetas avanzadas
│       ├── gold_seed_cleaned.json   # Benchmark de oro limpio
│       ├── train.json              # 1.781 ejemplos balanceados
│       ├── val.json                # 378 ejemplos de validación
│       └── test.json               # 380 ejemplos de prueba
├── models_saved/
│   ├── vocab.json                  # Vocabulario de tokens
│   ├── tag_vocab.json              # Vocabulario de tags BIO
│   ├── intent_map.json             # Mapeo de intenciones
│   ├── intent_model.pt             # Pesos entrenados Bi-GRU de Intención
│   └── ner_model.pt                # Pesos entrenados Bi-GRU + CRF de NER
├── src/
│   ├── preprocessing/
│   │   ├── tokenizer.py            # Tokenizador en español con cálculo de offsets
│   │   └── vocabulary.py           # Gestor de vocabularios de tokens y tags BIO
│   ├── models/
│   │   ├── intent_classifier.py    # Modelo Bi-GRU para clasificación de intención
│   │   ├── ner_tagger.py           # Modelo Bi-GRU + Linear-Chain CRF para NER
│   │   └── relation_extractor.py   # Vinculación de cantidades, exclusiones y sustituciones
│   ├── structuring/
│   │   └── parser.py               # Ensamblador de Contrato JSON y formato cocina
│   ├── pipeline.py                 # Pipeline End-to-End orquestador
│   ├── train_intent.py             # Script de entrenamiento Etapa 1
│   ├── train_ner.py                # Script de entrenamiento Etapa 2
│   └── evaluate.py                 # Evaluación global y métricas de negocio
├── requirements.txt                # Dependencias (PyTorch, seqeval, scikit-learn, etc.)
└── README.md
```

---

## 3. Guía de Ejecución

### Paso 1: Preparación de Datos y Limpieza
```bash
python data/clean_datasets.py
python data/generator.py
```

### Paso 2: Entrenamiento del Clasificador de Intención (Etapa 1)
```bash
python src/train_intent.py
```

### Paso 3: Entrenamiento del Modelo NER + CRF (Etapa 2)
```bash
python src/train_ner.py
```

### Paso 4: Evaluación Global y Métricas
```bash
python src/evaluate.py
```

### Paso 5: Prueba del Pipeline End-to-End
```python
from src.pipeline import OrderExtractionPipeline

pipeline = OrderExtractionPipeline()
resultado = pipeline.process("Hola buenas tardes, disculpe, quisiera 2 hamburguesas dobles sin cebolla y una porción de papas por favor")

print(resultado["comanda_cocina"])
# Salida: "2x Hamburguesa Doble sin cebolla, 1x Papas Fritas"

print(resultado["contrato_json"])
```
