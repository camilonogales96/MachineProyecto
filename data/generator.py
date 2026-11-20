"""
Generador de dataset enriquecido con dialectos, cortesías, exclusiones y sustituciones (SUST_DE, SUST_A).
Integra el Gold Standard (dataset_pedidos_lote1.csv) para producir conjuntos balanceados:
train.json, val.json, test.json.
"""

import os
import sys
import json
import random
from typing import List, Dict, Any, Tuple

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.preprocessing.tokenizer import SimpleSpanishTokenizer

# Variaciones dialectales y cortesías
CORTESIAS_INICIO_DIALECTOS = {
    "neutro": ["Hola", "Hola buenas tardes", "Buenas noches", "Buen día", "Buenas", "Por favor", ""],
    "rioplatense": ["Buenas", "Hola che", "Buenas noches", "Hola cómo va", "Che buenas", ""],
    "mexicano": ["Hola qué tal", "Buenas tardes", "Hola oigan", "Qué onda buenas", "Hola plis", ""],
    "andino": ["Hola buenas tardes", "Buen día por favor", "Buenas noches disculpe la molestia", "Hola caserito buenas", ""],
    "español_españa": ["Buenas", "Hola buenas", "Hola majos", "Buenas tardes", ""]
}

VERBOS_PEDIDO_DIALECTOS = {
    "neutro": ["quisiera pedir", "me gustaría pedir", "quiero pedir", "voy a querer", "para ordenar", "deseo pedir"],
    "rioplatense": ["poneme", "me traés", "me mandás", "voy a querer", "anotame", "quisiera pedir"],
    "mexicano": ["me das", "me preparas", "voy a querer", "me pones", "quisiera", "para pedir"],
    "andino": ["deme por favor", "me sirve", "anóteme", "quisiera pedirme", "me da"],
    "español_españa": ["ponme", "me pones", "nos pones", "marcha", "quisiera pedir"]
}

CORTESIAS_FIN_DIALECTOS = {
    "neutro": ["por favor", "muchas gracias", "gracias de antemano", "porfa", ""],
    "rioplatense": ["porfa", "gracias che", "joya gracias", "de diez gracias", ""],
    "mexicano": ["porfa", "plis", "muchas gracias", "sale gracias", ""],
    "andino": ["por favorcito", "muchas gracias", "se lo agradecería", "Dios se lo pague", ""],
    "español_españa": ["porfa", "gracias", "muchas gracias", "vale gracias", ""]
}

PREGUNTAS_CONSULTA = [
    ("¿A qué hora cierran hoy?", "consulta"),
    ("¿Tienen servicio de entrega a domicilio?", "consulta"),
    ("¿Cuál es el costo del envío a zona centro?", "consulta"),
    ("¿Aceptan pagos con tarjeta o transferencia?", "consulta"),
    ("¿Cuánto demora aproximadamente el pedido?", "consulta"),
    ("¿Tienen opciones vegetarianas en el menú?", "consulta"),
    ("¿Qué ingredientes lleva la hamburguesa especial?", "consulta"),
    ("Buenas tardes, ¿tienen promociones para hoy?", "consulta"),
    ("Disculpe, ¿hasta qué hora reciben pedidos?", "consulta"),
    ("¿Las hamburguesas vienen con papas fritas incluidas?", "consulta")
]

TEXTOS_RECLAMO = [
    ("Mi pedido llegó frío y falta una gaseosa", "reclamo"),
    ("Llevo esperando más de una hora y no llega mi comida", "reclamo"),
    ("La hamburguesa vino con cebolla y expresamente pedí sin cebolla", "reclamo"),
    ("Me cobraron de más en la cuenta y el repartidor no me dio ticket", "reclamo"),
    ("Las papas fritas están completamente quemadas y duras", "reclamo"),
    ("Pésima atención, el pedido vino equivocado", "reclamo"),
    ("Hola, disculpe pero me llegó la orden incompleta", "reclamo")
]

TEXTOS_SALUDO_OTRO = [
    ("Hola buenas tardes, ¿cómo están?", "saludo_otro"),
    ("Hola amigos, espero que tengan un excelente día", "saludo_otro"),
    ("Buenas noches a todos", "saludo_otro"),
    ("Hola qué tal todo por allá", "saludo_otro"),
    ("Ok muchas gracias por responder", "saludo_otro"),
    ("Listo, entendido, luego les escribo", "saludo_otro"),
    ("Perfecto amigos, que les vaya muy bien", "saludo_otro")
]


class EnhancedOrderDatasetGenerator:
    def __init__(self, ontology_path: str):
        with open(ontology_path, "r", encoding="utf-8") as f:
            self.ontology = json.load(f)
        self.tokenizer = SimpleSpanishTokenizer()
        
        self.productos_flat = []
        for cat, items in self.ontology["categorias"].items():
            for item in items:
                self.productos_flat.append(item)
                
        self.ingredientes = self.ontology["ingredientes_modificables"]
        self.exclusiones = self.ontology["marcadores_exclusion"]
        self.adiciones = self.ontology["marcadores_adicion"]
        self.cantidades = list(self.ontology["cantidades_lexicas"].keys())
        self.unidades = self.ontology["unidades"]

    def _generar_item(self, dialecto: str) -> Tuple[str, Dict[str, Any], List[Tuple[str, str]]]:
        prod_obj = random.choice(self.productos_flat)
        prod_nombre_oral = random.choice(prod_obj["sinonimos"])
        
        cant_str = random.choice(self.cantidades)
        cant_num = self.ontology["cantidades_lexicas"][cant_str]
        
        usar_unidad = random.random() < 0.20
        unidad_str = random.choice(self.unidades) if usar_unidad else None
        
        componentes = [cant_str]
        spans_anotados = [(cant_str, "CANTIDAD")]
        
        if unidad_str:
            componentes.append(unidad_str)
            spans_anotados.append((unidad_str, "UNIDAD"))
            if random.random() < 0.6:
                componentes.append("de")
                
        componentes.append(prod_nombre_oral)
        spans_anotados.append((prod_nombre_oral, "PRODUCTO"))
        
        exclusiones_list = []
        ingredientes_extra_list = []
        sustituciones_list = []
        
        # Posibilidad de Exclusión (35%)
        if random.random() < 0.35:
            ing_exc = random.choice(self.ingredientes)
            marc_exc = random.choice(self.exclusiones)
            componentes.append(marc_exc)
            spans_anotados.append((marc_exc, "NEGACION"))
            componentes.append(ing_exc)
            spans_anotados.append((ing_exc, "EXCLUSION"))
            exclusiones_list.append(ing_exc)
            
        # Posibilidad de Adición / Modificación (30%)
        if random.random() < 0.30:
            ing_add = random.choice(self.ingredientes)
            marc_add = random.choice(self.adiciones)
            componentes.append(marc_add)
            spans_anotados.append((marc_add, "MODIFICACION"))
            componentes.append(ing_add)
            spans_anotados.append((ing_add, "INGREDIENTE"))
            ingredientes_extra_list.append(ing_add)

        # Posibilidad de Sustitución (15%) -> ej: cámbiame el tomate por lechuga
        if random.random() < 0.15:
            ing_de = random.choice(self.ingredientes)
            ing_a = random.choice([x for x in self.ingredientes if x != ing_de])
            marc_sust = random.choice(["cambiando el", "cámbiame el", "cambiando", "en vez de", "cambiar"])
            conector_a = random.choice(["por", "por favor por"])
            
            componentes.append(marc_sust)
            componentes.append(ing_de)
            spans_anotados.append((ing_de, "SUST_DE"))
            componentes.append(conector_a)
            componentes.append(ing_a)
            spans_anotados.append((ing_a, "SUST_A"))
            sustituciones_list.append({"de": ing_de, "a": ing_a})

        item_json = {
            "producto": prod_nombre_oral,
            "producto_normalizado": prod_obj["nombre"],
            "cantidad": cant_num,
            "unidad": unidad_str,
            "ingredientes": ingredientes_extra_list,
            "modificaciones": [],
            "exclusiones": exclusiones_list,
            "sustituciones": sustituciones_list
        }
        return " ".join(componentes), item_json, spans_anotados

    def generar_ejemplo_pedido(self, dialecto: str = "neutro") -> Dict[str, Any]:
        num_items = random.choices([1, 2, 3], weights=[0.55, 0.35, 0.10])[0]
        
        cortesia_ini = random.choice(CORTESIAS_INICIO_DIALECTOS[dialecto])
        verbo = random.choice(VERBOS_PEDIDO_DIALECTOS[dialecto])
        cortesia_fin = random.choice(CORTESIAS_FIN_DIALECTOS[dialecto])
        
        items_texts = []
        items_json = []
        all_spans = []
        
        for _ in range(num_items):
            txt, j_item, spans = self._generar_item(dialecto)
            items_texts.append(txt)
            items_json.append(j_item)
            all_spans.extend(spans)
            
        conectores = [", y también ", ", con ", " y ", ", más ", ", además de "]
        cuerpo = items_texts[0]
        for it_txt in items_texts[1:]:
            cuerpo += random.choice(conectores) + it_txt
            
        partes = [p for p in [cortesia_ini, verbo, cuerpo, cortesia_fin] if p]
        texto = " ".join(" ".join(partes).split())
        
        tokens = self.tokenizer.tokenize(texto)
        bio_tags = self._asignar_bio_tags(tokens, all_spans)
        
        # Comanda limpia
        comanda_partes = []
        for it in items_json:
            p_desc = f"{it['cantidad']}x {it['producto_normalizado']}"
            if it["exclusiones"]:
                p_desc += f" sin {', '.join(it['exclusiones'])}"
            if it["ingredientes"]:
                p_desc += f" con extra {', '.join(it['ingredientes'])}"
            if it["sustituciones"]:
                for s in it["sustituciones"]:
                    p_desc += f" (cambio {s['de']} por {s['a']})"
            comanda_partes.append(p_desc)
        comanda = ", ".join(comanda_partes)

        contrato = {
            "intencion": "pedido",
            "items": items_json,
            "restricciones_globales": [],
            "notas": []
        }
        
        return {
            "texto": texto,
            "tokens": tokens,
            "bio_tags": bio_tags,
            "intencion": "pedido",
            "contrato_json": contrato,
            "comanda": comanda,
            "dialecto": dialecto
        }

    def _asignar_bio_tags(self, tokens: List[str], spans_anotados: List[Tuple[str, str]]) -> List[str]:
        bio_tags = ["O"] * len(tokens)
        spans_anotados.sort(key=lambda x: len(x[0]), reverse=True)
        
        for frase, tag in spans_anotados:
            frase_toks = self.tokenizer.tokenize(frase)
            if not frase_toks:
                continue
            m = len(frase_toks)
            for i in range(len(tokens) - m + 1):
                if tokens[i:i+m] == frase_toks:
                    if all(bio_tags[i+k] == "O" for k in range(m)):
                        bio_tags[i] = f"B-{tag}"
                        for k in range(1, m):
                            bio_tags[i+k] = f"I-{tag}"
        return bio_tags

    def build_full_dataset(self, gold_seed_path: str, n_synthetic: int = 2500) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Combina datos sintéticos aumentados con el Gold Seed de dataset_pedidos_lote1.csv."""
        # 1. Cargar Gold Seed
        gold_train = []
        gold_val = []
        gold_test = []
        if os.path.exists(gold_seed_path):
            with open(gold_seed_path, "r", encoding="utf-8") as f:
                gold_data = json.load(f)
            for g in gold_data:
                sp = g.get("split", "train")
                if sp == "val":
                    gold_val.append(g)
                elif sp == "test":
                    gold_test.append(g)
                else:
                    gold_train.append(g)

        print(f"Gold Seed cargado: {len(gold_train)} train, {len(gold_val)} val, {len(gold_test)} test")

        # 2. Generar datos sintéticos multi-dialecto
        dialectos = ["neutro", "rioplatense", "mexicano", "andino", "español_españa"]
        synthetic = []
        
        n_pedidos = int(n_synthetic * 0.75)
        n_otros = n_synthetic - n_pedidos
        
        for _ in range(n_pedidos):
            dial = random.choice(dialectos)
            synthetic.append(self.generar_ejemplo_pedido(dialecto=dial))
            
        for _ in range(n_otros // 3):
            p, intent = random.choice(PREGUNTAS_CONSULTA)
            toks = self.tokenizer.tokenize(p)
            synthetic.append({"texto": p, "tokens": toks, "bio_tags": ["O"]*len(toks), "intencion": intent,
                              "contrato_json": {"intencion": intent, "items": []}, "comanda": f"[{intent.upper()}]", "dialecto": "neutro"})

        for _ in range(n_otros // 3):
            p, intent = random.choice(TEXTOS_RECLAMO)
            toks = self.tokenizer.tokenize(p)
            synthetic.append({"texto": p, "tokens": toks, "bio_tags": ["O"]*len(toks), "intencion": intent,
                              "contrato_json": {"intencion": intent, "items": []}, "comanda": f"[{intent.upper()}]", "dialecto": "neutro"})

        for _ in range(n_otros // 3):
            p, intent = random.choice(TEXTOS_SALUDO_OTRO)
            toks = self.tokenizer.tokenize(p)
            synthetic.append({"texto": p, "tokens": toks, "bio_tags": ["O"]*len(toks), "intencion": intent,
                              "contrato_json": {"intencion": intent, "items": []}, "comanda": f"[{intent.upper()}]", "dialecto": "neutro"})

        random.shuffle(synthetic)
        
        n_syn_train = int(len(synthetic) * 0.70)
        n_syn_val = int(len(synthetic) * 0.15)
        
        full_train = gold_train + synthetic[:n_syn_train]
        full_val = gold_val + synthetic[n_syn_train:n_syn_train + n_syn_val]
        full_test = gold_test + synthetic[n_syn_train + n_syn_val:]
        
        random.shuffle(full_train)
        random.shuffle(full_val)
        random.shuffle(full_test)
        
        return full_train, full_val, full_test


if __name__ == "__main__":
    ontology_file = os.path.join(current_dir, "menu_ontology.json")
    gold_file = os.path.join(current_dir, "dataset", "gold_seed_cleaned.json")
    generator = EnhancedOrderDatasetGenerator(ontology_file)
    
    train, val, test = generator.build_full_dataset(gold_file, n_synthetic=2500)
    dataset_dir = os.path.join(current_dir, "dataset")
    
    for fname, data in [("train.json", train), ("val.json", val), ("test.json", test)]:
        out = os.path.join(dataset_dir, fname)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Generado {fname}: {len(data)} ejemplos guardados en {out}")
