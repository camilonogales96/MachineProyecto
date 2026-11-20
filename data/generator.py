"""
Generador de dataset sintético con Data Augmentation para pedidos de restaurante.
Genera pares: (Texto libre con cortesías/ruido, Intención, Tokens BIO para NER, Contrato JSON).
Completamente autónomo, sin modelos pre-entrenados ni APIs externas.
"""

import os
import sys
import json
import random
from typing import List, Dict, Any, Tuple

# Asegurar import de tokenizer local
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.preprocessing.tokenizer import SimpleSpanishTokenizer

CORTESIAS_INICIO = [
    "Hola",
    "Hola buenas tardes",
    "Buenas noches",
    "Buen día",
    "Buenas",
    "Hola estimado",
    "Disculpe la molestia",
    "Hola disculpe una pregunta",
    "Buenas tardes disculpe la molestia",
    "Hola qué tal",
    "Oiga buenas",
    "Por favor",
    "Hola amigo",
    "Buenas noches estimado",
    ""
]

VERBOS_PEDIDO = [
    "quisiera pedir",
    "me gustaría pedir",
    "quiero pedir",
    "voy a querer",
    "deme",
    "me trae",
    "mándeme",
    "mandame",
    "deseo ordenar",
    "para ordenar",
    "anóteme",
    "anotame",
    "me envía",
    "quisiera",
    "me da",
    "serían"
]

CORTESIAS_FIN = [
    "por favor",
    "por favorcito",
    "muchas gracias",
    "gracias de antemano",
    "se lo agradecería",
    "porfa",
    "eso sería todo gracias",
    "y nada más gracias",
    "mil gracias",
    ""
]

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


class OrderDatasetGenerator:
    def __init__(self, ontology_path: str):
        with open(ontology_path, "r", encoding="utf-8") as f:
            self.ontology = json.load(f)
        self.tokenizer = SimpleSpanishTokenizer()
        
        # Consolidar todos los productos y variantes
        self.productos_flat = []
        for cat, items in self.ontology["categorias"].items():
            for item in items:
                self.productos_flat.append(item)
                
        self.ingredientes = self.ontology["ingredientes_modificables"]
        self.exclusiones = self.ontology["marcadores_exclusion"]
        self.adiciones = self.ontology["marcadores_adicion"]
        self.cantidades = list(self.ontology["cantidades_lexicas"].keys())
        self.unidades = self.ontology["unidades"]

    def _generar_item_pedido(self) -> Tuple[str, Dict[str, Any], List[Tuple[str, str]]]:
        """
        Genera un componente de pedido con etiquetas explícitas.
        Retorna:
          - Texto del segmento
          - Dict del item (contrato JSON)
          - Lista de tuplas (palabra/frase, tag_entidad) para BIO tagging preciso
        """
        prod_obj = random.choice(self.productos_flat)
        prod_nombre_oral = random.choice(prod_obj["sinonimos"])
        
        # Cantidad
        cant_str = random.choice(self.cantidades)
        cant_num = self.ontology["cantidades_lexicas"][cant_str]
        
        # Opcional unidad (ej: 1 "porción de" papas)
        usar_unidad = random.random() < 0.25
        unidad_str = random.choice(self.unidades) if usar_unidad else None
        
        # Opcional exclusión (ej: sin cebolla)
        usar_exclusion = random.random() < 0.4
        exclusion_ingrediente = None
        exclusion_marcador = None
        if usar_exclusion:
            exclusion_ingrediente = random.choice(self.ingredientes)
            exclusion_marcador = random.choice(self.exclusiones)

        # Opcional adición / modificación (ej: con extra queso, con doble carne)
        usar_adicion = random.random() < 0.35
        adicion_ingrediente = None
        adicion_marcador = None
        if usar_adicion:
            adicion_ingrediente = random.choice(self.ingredientes)
            adicion_marcador = random.choice(self.adiciones)

        # Construir frase del item y registrar spans anotados
        componentes = []
        spans_anotados = []  # (texto, tag)
        
        # Cantidad
        componentes.append(cant_str)
        spans_anotados.append((cant_str, "CANTIDAD"))
        
        if unidad_str:
            componentes.append(unidad_str)
            spans_anotados.append((unidad_str, "UNIDAD"))
            if random.random() < 0.6:
                componentes.append("de")
                
        # Producto
        componentes.append(prod_nombre_oral)
        spans_anotados.append((prod_nombre_oral, "PRODUCTO"))
        
        exclusiones_list = []
        ingredientes_extra_list = []
        
        # Exclusión / Adición en orden aleatorio
        modificadores = []
        if usar_exclusion and exclusion_marcador and exclusion_ingrediente:
            frase_exc = f"{exclusion_marcador} {exclusion_ingrediente}"
            modificadores.append(("exc", exclusion_marcador, exclusion_ingrediente, frase_exc))
            exclusiones_list.append(exclusion_ingrediente)
            
        if usar_adicion and adicion_marcador and adicion_ingrediente:
            frase_add = f"{adicion_marcador} {adicion_ingrediente}"
            modificadores.append(("add", adicion_marcador, adicion_ingrediente, frase_add))
            ingredientes_extra_list.append(adicion_ingrediente)
            
        random.shuffle(modificadores)
        for tipo, marcador, ing, frase in modificadores:
            if tipo == "exc":
                componentes.append(marcador)
                spans_anotados.append((marcador, "NEGACION"))
                componentes.append(ing)
                spans_anotados.append((ing, "EXCLUSION"))
            else:
                componentes.append(marcador)
                spans_anotados.append((marcador, "MODIFICACION"))
                componentes.append(ing)
                spans_anotados.append((ing, "INGREDIENTE"))

        item_json = {
            "producto": prod_nombre_oral,
            "producto_normalizado": prod_obj["nombre"],
            "cantidad": cant_num,
            "unidad": unidad_str,
            "ingredientes": ingredientes_extra_list,
            "modificaciones": [],
            "exclusiones": exclusiones_list,
            "sustituciones": []
        }
        
        item_text = " ".join(componentes)
        return item_text, item_json, spans_anotados

    def generar_ejemplo_pedido(self) -> Dict[str, Any]:
        """Genera un pedido completo con 1 a 3 items y cortesías."""
        num_items = random.choices([1, 2, 3], weights=[0.55, 0.35, 0.10])[0]
        
        cortesia_ini = random.choice(CORTESIAS_INICIO)
        verbo = random.choice(VERBOS_PEDIDO)
        cortesia_fin = random.choice(CORTESIAS_FIN)
        
        items_texts = []
        items_json = []
        all_spans = []
        
        for _ in range(num_items):
            txt, j_item, spans = self._generar_item_pedido()
            items_texts.append(txt)
            items_json.append(j_item)
            all_spans.extend(spans)
            
        # Unir items con conectores naturales
        conectores = [", y también ", ", con ", " y ", ", más "]
        cuerpo_pedido = ""
        for i, it_txt in enumerate(items_texts):
            if i == 0:
                cuerpo_pedido = it_txt
            else:
                conector = random.choice(conectores)
                cuerpo_pedido += conector + it_txt
                
        partes = []
        if cortesia_ini:
            partes.append(cortesia_ini)
        if verbo:
            partes.append(verbo)
        partes.append(cuerpo_pedido)
        if cortesia_fin:
            partes.append(cortesia_fin)
            
        texto_completo = " ".join(p for p in partes if p).strip()
        # Normalizar espacios
        texto_completo = " ".join(texto_completo.split())
        
        # Etiquetado BIO a nivel token
        tokens = self.tokenizer.tokenize(texto_completo)
        bio_tags = self._asignar_bio_tags(texto_completo, tokens, all_spans)
        
        # Resumen limpio (línea para cocina)
        resumen_partes = []
        for it in items_json:
            p_desc = f"{it['cantidad']}x {it['producto_normalizado']}"
            if it["exclusiones"]:
                p_desc += f" sin {', '.join(it['exclusiones'])}"
            if it["ingredientes"]:
                p_desc += f" con extra {', '.join(it['ingredientes'])}"
            resumen_partes.append(p_desc)
        resumen_cocina = ", ".join(resumen_partes)

        contrato_json = {
            "intencion": "pedido",
            "items": items_json,
            "restricciones_globales": [],
            "notas": [],
            "resumen_cocina": resumen_cocina
        }
        
        return {
            "texto": texto_completo,
            "tokens": tokens,
            "bio_tags": bio_tags,
            "intencion": "pedido",
            "contrato_json": contrato_json
        }

    def _asignar_bio_tags(self, texto: str, tokens: List[str], spans_anotados: List[Tuple[str, str]]) -> List[str]:
        """Alinea los spans anotados a los tokens para generar tags BIO."""
        bio_tags = ["O"] * len(tokens)
        
        # Ordenar spans por longitud descendente para evitar colisiones
        spans_anotados.sort(key=lambda x: len(x[0]), reverse=True)
        
        # Buscar cada span en el texto y proyectar sobre tokens
        for frase, tag in spans_anotados:
            frase_tokens = self.tokenizer.tokenize(frase)
            if not frase_tokens:
                continue
            
            # Buscar ocurrencia de frase_tokens en tokens
            m = len(frase_tokens)
            for i in range(len(tokens) - m + 1):
                if tokens[i:i+m] == frase_tokens:
                    # Verificar si ya está anotado
                    if all(bio_tags[i+k] == "O" for k in range(m)):
                        bio_tags[i] = f"B-{tag}"
                        for k in range(1, m):
                            bio_tags[i+k] = f"I-{tag}"
                            
        return bio_tags

    def generar_dataset(self, n_pedidos: int = 2000, n_otros: int = 500) -> List[Dict[str, Any]]:
        """Genera el dataset completo balanceado con pedidos y clases de control."""
        dataset = []
        
        # 1. Pedidos (Etapa 1: 'pedido', Etapa 2: NER + JSON)
        for _ in range(n_pedidos):
            dataset.append(self.generar_ejemplo_pedido())
            
        # 2. Consultas
        for _ in range(n_otros // 3):
            preg, intencion = random.choice(PREGUNTAS_CONSULTA)
            # Agregar variación leve
            saludo = random.choice(CORTESIAS_INICIO)
            txt = f"{saludo} {preg}".strip() if saludo else preg
            tokens = self.tokenizer.tokenize(txt)
            dataset.append({
                "texto": txt,
                "tokens": tokens,
                "bio_tags": ["O"] * len(tokens),
                "intencion": intencion,
                "contrato_json": {"intencion": intencion, "items": []}
            })
            
        # 3. Reclamos
        for _ in range(n_otros // 3):
            rec, intencion = random.choice(TEXTOS_RECLAMO)
            saludo = random.choice(CORTESIAS_INICIO)
            txt = f"{saludo} {rec}".strip() if saludo else rec
            tokens = self.tokenizer.tokenize(txt)
            dataset.append({
                "texto": txt,
                "tokens": tokens,
                "bio_tags": ["O"] * len(tokens),
                "intencion": intencion,
                "contrato_json": {"intencion": intencion, "items": []}
            })
            
        # 4. Saludo / Otro
        for _ in range(n_otros // 3):
            sal, intencion = random.choice(TEXTOS_SALUDO_OTRO)
            tokens = self.tokenizer.tokenize(sal)
            dataset.append({
                "texto": sal,
                "tokens": tokens,
                "bio_tags": ["O"] * len(tokens),
                "intencion": intencion,
                "contrato_json": {"intencion": intencion, "items": []}
            })
            
        random.shuffle(dataset)
        return dataset


if __name__ == "__main__":
    ontology_file = os.path.join(current_dir, "menu_ontology.json")
    generator = OrderDatasetGenerator(ontology_file)
    
    print("Generando dataset...")
    data = generator.generar_dataset(n_pedidos=2000, n_otros=600)
    print(f"Total ejemplos generados: {len(data)}")
    
    # Split train (70%), val (15%), test (15%)
    n_total = len(data)
    n_train = int(n_total * 0.70)
    n_val = int(n_total * 0.15)
    
    train_data = data[:n_train]
    val_data = data[n_train:n_train + n_val]
    test_data = data[n_train + n_val:]
    
    dataset_dir = os.path.join(current_dir, "dataset")
    os.makedirs(dataset_dir, exist_ok=True)
    
    for split_name, split_data in [("train.json", train_data), ("val.json", val_data), ("test.json", test_data)]:
        out_path = os.path.join(dataset_dir, split_name)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(split_data, f, ensure_ascii=False, indent=2)
        print(f"Guardado {split_name}: {len(split_data)} ejemplos en {out_path}")

    # Mostrar ejemplo
    print("\n--- EJEMPLO MUESTRA ---")
    ej = train_data[0]
    print("Texto:", ej["texto"])
    print("Intención:", ej["intencion"])
    print("Tokens/BIO:", list(zip(ej["tokens"], ej["bio_tags"]))[:15])
    print("JSON:", json.dumps(ej["contrato_json"], ensure_ascii=False, indent=2))
