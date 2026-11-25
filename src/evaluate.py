"""
ETAPA 4 · Evaluación Global del Sistema y Métricas de Negocio
Evalúa el pipeline completo contra el conjunto de Test (incluyendo el Gold Standard de dataset_pedidos_lote1.csv).
Mide: F1-Macro intención, seqeval NER por entidad, Exact Match JSON, Exact Match Comanda y Reducción de Lectura.
"""

import os
import sys
import json
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from seqeval.metrics import classification_report as seqeval_report, f1_score as seqeval_f1

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.pipeline import OrderExtractionPipeline


def evaluate_system(test_json_path: str):
    print("=" * 70)
    print("EVALUACIÓN GLOBAL DEL SISTEMA (TEST SET & GOLD BENCHMARK)")
    print("=" * 70)
    
    with open(test_json_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)
        
    print(f"Total ejemplos en Test: {len(test_data)}")
    
    pipeline = OrderExtractionPipeline()
    
    y_true_intent = []
    y_pred_intent = []
    
    y_true_bio = []
    y_pred_bio = []
    
    json_exact_matches = 0
    comanda_exact_matches = 0
    total_orders = 0
    
    total_chars_original = 0
    total_chars_comanda = 0
    
    sample_showcases = []
    
    for i, item in enumerate(test_data):
        text = item["texto"]
        gold_intent = item["intencion"].strip().lower()
        gold_bio = item.get("bio_tags", [])
        gold_contract = item.get("contrato_json", {})
        gold_comanda = item.get("comanda")
        
        # Inferencia con pipeline
        result = pipeline.process(text)
        pred_intent = result["intencion"]
        pred_bio = result["bio_tags"]
        pred_contract = result["contrato_json"]
        pred_comanda = result["comanda_cocina"]
        
        # Métricas intención
        y_true_intent.append(gold_intent)
        y_pred_intent.append(pred_intent)
        
        # Métricas NER
        if gold_bio:
            # Alinear longitudes
            p_bio = pred_bio[:len(gold_bio)]
            if len(p_bio) < len(gold_bio):
                p_bio.extend(["O"] * (len(gold_bio) - len(p_bio)))
            y_true_bio.append(gold_bio)
            y_pred_bio.append(p_bio)
            
        # Métricas de pedidos
        if gold_intent == "pedido":
            total_orders += 1
            total_chars_original += len(text)
            total_chars_comanda += len(pred_comanda)
            
            # Comparar items en Contrato JSON
            g_items = gold_contract.get("items", [])
            p_items = pred_contract.get("items", [])
            
            # Chequeo de correspondencia de items
            if len(g_items) == len(p_items):
                match = True
                for g_it, p_it in zip(g_items, p_items):
                    if (g_it.get("cantidad") != p_it.get("cantidad") or
                        g_it.get("producto_normalizado") != p_it.get("producto_normalizado") or
                        set(g_it.get("exclusiones", [])) != set(p_it.get("exclusiones", []))):
                        match = False
                        break
                if match:
                    json_exact_matches += 1
                    
            if gold_comanda and gold_comanda.strip().lower() == pred_comanda.strip().lower():
                comanda_exact_matches += 1
                
        # Guardar 3 ejemplos variados para mostrar
        if len(sample_showcases) < 4:
            sample_showcases.append({
                "texto": text,
                "intencion_pred": pred_intent,
                "comanda_pred": pred_comanda,
                "contrato_pred": pred_contract
            })
            
    # 1. Reporte de Intención
    print("\n" + "=" * 30 + " ETAPA 1: INTENCIÓN " + "=" * 30)
    acc = accuracy_score(y_true_intent, y_pred_intent)
    print(f"Accuracy Global: {acc*100:.2f}%\n")
    print(classification_report(y_true_intent, y_pred_intent, digits=4))
    
    # 2. Reporte de NER
    print("\n" + "=" * 30 + " ETAPA 2: ENTIDADES NER (SEQEVAL) " + "=" * 30)
    print(seqeval_report(y_true_bio, y_pred_bio, digits=4))
    f1_bio = seqeval_f1(y_true_bio, y_pred_bio)
    print(f"F1 Global Estricto NER: {f1_bio*100:.2f}%")
    
    # 3. Reporte de Estructuración y Negocio
    print("\n" + "=" * 30 + " MÉTRICAS DE NEGOCIO Y ESTRUCTURACIÓN " + "=" * 30)
    if total_orders > 0:
        json_em_pct = (json_exact_matches / total_orders) * 100
        reduction_pct = ((total_chars_original - total_chars_comanda) / total_chars_original) * 100
        print(f"Total Pedidos Evaluados: {total_orders}")
        print(f"Exact Match Contrato JSON: {json_em_pct:.2f}% ({json_exact_matches}/{total_orders})")
        print(f"Caracteres Totales Texto Original: {total_chars_original}")
        print(f"Caracteres Totales Comanda Cocina: {total_chars_comanda}")
        print(f"Tasa de Reducción de Texto / Tiempo de Lectura: {reduction_pct:.2f}%")
        
    # 4. Muestra de Casos Reales
    print("\n" + "=" * 30 + " DEMOSTRACIÓN DE CASOS REALES " + "=" * 30)
    for idx, sample in enumerate(sample_showcases, 1):
        print(f"\n--- CASO {idx} ---")
        print("Entrada:", sample["texto"])
        print("Intención Predicha:", sample["intencion_pred"])
        print("Comanda para Cocina:", sample["comanda_pred"])
        print("Contrato JSON:", json.dumps(sample["contrato_pred"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    test_file = os.path.join(PROJECT_ROOT, "data", "dataset", "test.json")
    evaluate_system(test_file)
