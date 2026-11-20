"""
Script de limpieza y normalización para los datasets:
- dataset_pedidos_lote1.csv (imputación de NaNs, parseo de listas JSON, preservación de Gold Benchmark)
- recetas.csv (extracción de vocabulario culinario y nombres de ingredientes)
"""

import os
import json
import pandas as pd
from typing import Dict, Any, List

DATASET_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(DATASET_DIR, "dataset")


def clean_pedidos_lote1(csv_path: str, output_json_path: str) -> List[Dict[str, Any]]:
    """
    Limpia el archivo dataset_pedidos_lote1.csv:
    1. Parsea columnas serializadas como JSON (tokens, etiquetas_bio, tecnicas_lexicas).
    2. Imputa valores nulos en filas de consulta, reclamo y saludo.
    3. Exporta un dataset limpio en formato JSON estructurado.
    """
    print(f"Limpiando {csv_path}...")
    df = pd.read_csv(csv_path)
    
    cleaned_records = []
    
    for idx, row in df.iterrows():
        # Parseo seguro de tokens y etiquetas
        try:
            tokens = json.loads(row["tokens"]) if isinstance(row["tokens"], str) else []
        except Exception:
            tokens = [t.strip("'\" ") for t in str(row["tokens"]).strip("[]").split(",")]

        try:
            bio_tags = json.loads(row["etiquetas_bio"]) if isinstance(row["etiquetas_bio"], str) else []
        except Exception:
            bio_tags = [t.strip("'\" ") for t in str(row["etiquetas_bio"]).strip("[]").split(",")]

        # Imputar contrato nulo para intenciones no-pedido
        intencion = str(row["intencion"]).strip().lower()
        if pd.isna(row["contrato"]):
            contrato = {
                "intencion": intencion,
                "items": [],
                "restricciones_globales": [],
                "notas": []
            }
        else:
            try:
                contrato = json.loads(row["contrato"])
            except Exception:
                contrato = {"intencion": intencion, "raw": str(row["contrato"])}

        # Imputar comanda nula
        if pd.isna(row["comanda"]):
            comanda = f"[{intencion.upper()}]"
        else:
            comanda = str(row["comanda"]).strip()

        # Parsear tecnicas lexicas
        try:
            tecnicas = json.loads(row["tecnicas_lexicas"]) if pd.notna(row["tecnicas_lexicas"]) else []
        except Exception:
            tecnicas = []

        rec = {
            "id": row["id"],
            "grupo_id": row["grupo_id"],
            "split": str(row["split"]).lower(),
            "texto": str(row["texto"]).strip(),
            "intencion": intencion,
            "tokens": tokens,
            "bio_tags": bio_tags,
            "contrato_json": contrato,
            "comanda": comanda,
            "escenario": str(row["escenario"]) if pd.notna(row["escenario"]) else "default",
            "dialecto": str(row["dialecto"]) if pd.notna(row["dialecto"]) else "neutro",
            "tecnicas_lexicas": tecnicas
        }
        cleaned_records.append(rec)
        
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(cleaned_records, f, ensure_ascii=False, indent=2)
        
    print(f"Dataset lote 1 limpiado exitosamente: {len(cleaned_records)} registros guardados en {output_json_path}")
    return cleaned_records


def extract_vocabulary_from_recetas(csv_path: str, output_vocab_path: str, max_samples: int = 10000):
    """
    Extrae vocabulario culinario a partir de recetas.csv para nutrir el menú y los embeddings.
    """
    if not os.path.exists(csv_path):
        print(f"Archivo de recetas no encontrado: {csv_path}")
        return
        
    print(f"Extrayendo vocabulario de {csv_path}...")
    df = pd.read_csv(csv_path, nrows=max_samples)
    
    # Extraer títulos y limpiar
    platos = []
    for t in df["title"].dropna():
        t_clean = t.strip()
        if len(t_clean) > 3:
            platos.append(t_clean)
            
    resumen_vocab = {
        "total_recetas_analizadas": len(df),
        "ejemplos_platos_extraidos": platos[:50]
    }
    
    with open(output_vocab_path, "w", encoding="utf-8") as f:
        json.dump(resumen_vocab, f, ensure_ascii=False, indent=2)
    print(f"Vocabulario culinario extraído en {output_vocab_path}")


if __name__ == "__main__":
    lote1_csv = os.path.join(RAW_DIR, "dataset_pedidos_lote1.csv")
    cleaned_lote1_json = os.path.join(RAW_DIR, "gold_seed_cleaned.json")
    clean_pedidos_lote1(lote1_csv, cleaned_lote1_json)
    
    recetas_csv = os.path.join(RAW_DIR, "recetas.csv")
    recetas_out = os.path.join(RAW_DIR, "vocabulario_recetas.json")
    extract_vocabulary_from_recetas(recetas_csv, recetas_out)
