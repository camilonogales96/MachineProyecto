"""
Genera un conjunto ampliado de ejemplos realistas (Gold Expanded) en español
para complementar gold_seed_cleaned.json (32 ejemplos en train -> 120+ reales).

Incluye escenarios conversacionales complejos:
- Indecisiones del cliente (dudas resueltas en pedido y dudas en consulta)
- Cancelaciones de pedidos (totales, por demora, por error)
- Agregados tardíos (aumentar productos a un pedido hecho hace un rato)
- Pedidos sin cantidad explícita, plurales, diminutivos y múltiples dialectos
- Consultas, reclamos y saludos variados sin plantillas repetitivas
"""

import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing.tokenizer import SimpleSpanishTokenizer


def _build_bio_from_spans(
    tokens: List[str],
    tagged_spans: List[Tuple[str, str]],
    tokenizer: SimpleSpanishTokenizer,
) -> List[str]:
    """Construye etiquetas BIO respetando el orden secuencial de aparición."""
    bio = ["O"] * len(tokens)
    cursor = 0
    for phrase, label in tagged_spans:
        p_toks = tokenizer.tokenize(phrase)
        if not p_toks:
            continue
        m = len(p_toks)
        found = False
        for i in range(cursor, len(tokens) - m + 1):
            if tokens[i : i + m] == p_toks and all(bio[i + k] == "O" for k in range(m)):
                bio[i] = f"B-{label}"
                for k in range(1, m):
                    bio[i + k] = f"I-{label}"
                cursor = i + m
                found = True
                break
        if not found:
            for i in range(0, len(tokens) - m + 1):
                if tokens[i : i + m] == p_toks and all(bio[i + k] == "O" for k in range(m)):
                    bio[i] = f"B-{label}"
                    for k in range(1, m):
                        bio[i + k] = f"I-{label}"
                    break
    return bio


def _make_order_record(
    rec_id: str,
    split: str,
    texto: str,
    escenario: str,
    dialecto: str,
    spans: List[Tuple[str, str]],
    items: List[Dict[str, Any]],
    comanda: str,
    tokenizer: SimpleSpanishTokenizer,
    notas: Optional[List[str]] = None,
) -> Dict[str, Any]:
    tokens = tokenizer.tokenize(texto)
    bio_tags = _build_bio_from_spans(tokens, spans, tokenizer)
    full_items = []
    for it in items:
        full_items.append(
            {
                "producto": it["producto"],
                "producto_normalizado": it["producto_normalizado"],
                "cantidad": it.get("cantidad", 1),
                "unidad": it.get("unidad"),
                "ingredientes": it.get("ingredientes", []),
                "modificaciones": it.get("modificaciones", []),
                "exclusiones": it.get("exclusiones", []),
                "sustituciones": it.get("sustituciones", []),
                "restricciones_globales": [],
            }
        )
    return {
        "id": rec_id,
        "grupo_id": f"exp_{escenario}",
        "split": split,
        "texto": texto,
        "intencion": "pedido",
        "tokens": tokens,
        "bio_tags": bio_tags,
        "contrato_json": {
            "intencion": "pedido",
            "items": full_items,
            "restricciones_globales": [],
            "notas": notas or [],
        },
        "comanda": comanda,
        "escenario": escenario,
        "dialecto": dialecto,
        "tecnicas_lexicas": [escenario],
    }


def _make_non_order_record(
    rec_id: str,
    split: str,
    texto: str,
    intencion: str,
    escenario: str,
    dialecto: str,
    tokenizer: SimpleSpanishTokenizer,
) -> Dict[str, Any]:
    tokens = tokenizer.tokenize(texto)
    return {
        "id": rec_id,
        "grupo_id": f"exp_{escenario}",
        "split": split,
        "texto": texto,
        "intencion": intencion,
        "tokens": tokens,
        "bio_tags": ["O"] * len(tokens),
        "contrato_json": {
            "intencion": intencion,
            "items": [],
            "restricciones_globales": [],
            "notas": [escenario],
        },
        "comanda": f"[{intencion.upper()}]",
        "escenario": escenario,
        "dialecto": dialecto,
        "tecnicas_lexicas": [escenario],
    }


def build_expanded_gold() -> List[Dict[str, Any]]:
    tok = SimpleSpanishTokenizer()
    records: List[Dict[str, Any]] = []
    idx = 1000

    def next_id() -> str:
        nonlocal idx
        idx += 1
        return f"exp_{idx:06d}"

    # ------------------------------------------------------------------
    # 1. INDECISIONES DEL CLIENTE (resueltas en pedido)
    # ------------------------------------------------------------------
    indecisiones_pedido = [
        (
            "train",
            "Mmm espera, mejor no quiero la cheeseburger, ponme 1 hamburguesa doble sin cebolla por favor",
            "neutro",
            [("1", "CANTIDAD"), ("hamburguesa doble", "PRODUCTO"), ("sin", "NEGACION"), ("cebolla", "EXCLUSION")],
            [{"producto": "hamburguesa doble", "producto_normalizado": "Hamburguesa Doble", "cantidad": 1, "exclusiones": ["cebolla"]}],
            "1x Hamburguesa Doble sin cebolla",
        ),
        (
            "train",
            "Ay no sé si pedir papas o aros... bueno ya, dame 1 aros de cebolla y 1 sprite",
            "mexicano",
            [("1", "CANTIDAD"), ("aros de cebolla", "PRODUCTO"), ("1", "CANTIDAD"), ("sprite", "PRODUCTO")],
            [
                {"producto": "aros de cebolla", "producto_normalizado": "Aros de Cebolla", "cantidad": 1},
                {"producto": "sprite", "producto_normalizado": "Sprite", "cantidad": 1},
            ],
            "1x Aros de Cebolla, 1x Sprite",
        ),
        (
            "train",
            "Perdón me arrepentí de la cerveza, mejor mándame 2 botellas de agua mineral y unas papas fritas",
            "rioplatense",
            [("2", "CANTIDAD"), ("botellas", "UNIDAD"), ("agua mineral", "PRODUCTO"), ("unas", "CANTIDAD"), ("papas fritas", "PRODUCTO")],
            [
                {"producto": "agua mineral", "producto_normalizado": "Agua Mineral", "cantidad": 2, "unidad": "botellas"},
                {"producto": "papas fritas", "producto_normalizado": "Papas Fritas", "cantidad": 1},
            ],
            "2x Agua Mineral (botellas), 1x Papas Fritas",
        ),
        (
            "train",
            "Estaba dudando bastante pero ya me decidí, quiero 1 hamburguesa especial sin pepinillos y con extra tocino",
            "neutro",
            [("1", "CANTIDAD"), ("hamburguesa especial", "PRODUCTO"), ("sin", "NEGACION"), ("pepinillos", "EXCLUSION"), ("con extra", "MODIFICACION"), ("tocino", "INGREDIENTE")],
            [{"producto": "hamburguesa especial", "producto_normalizado": "Hamburguesa Especial", "cantidad": 1, "exclusiones": ["pepinillos"], "ingredientes": ["tocino"]}],
            "1x Hamburguesa Especial sin pepinillos con extra tocino",
        ),
        (
            "train",
            "Espera, cambié de opinión, en vez de dos simples dame 1 hamburguesa doble sin tomate y 1 coca cola",
            "andino",
            [("1", "CANTIDAD"), ("hamburguesa doble", "PRODUCTO"), ("sin", "NEGACION"), ("tomate", "EXCLUSION"), ("1", "CANTIDAD"), ("coca cola", "PRODUCTO")],
            [
                {"producto": "hamburguesa doble", "producto_normalizado": "Hamburguesa Doble", "cantidad": 1, "exclusiones": ["tomate"]},
                {"producto": "coca cola", "producto_normalizado": "Coca Cola", "cantidad": 1},
            ],
            "1x Hamburguesa Doble sin tomate, 1x Coca Cola",
        ),
        (
            "train",
            "Bueno pensándolo bien mejor sí voy a querer 2 nuggets de pollo y 1 fanta naranja",
            "mexicano",
            [("2", "CANTIDAD"), ("nuggets de pollo", "PRODUCTO"), ("1", "CANTIDAD"), ("fanta naranja", "PRODUCTO")],
            [
                {"producto": "nuggets de pollo", "producto_normalizado": "Nuggets de Pollo", "cantidad": 2},
                {"producto": "fanta naranja", "producto_normalizado": "Fanta", "cantidad": 1},
            ],
            "2x Nuggets de Pollo, 1x Fanta",
        ),
        (
            "train",
            "Uf qué difícil elegir, ya déjame 1 cheeseburger con extra queso cheddar y 1 papas rusticas",
            "español_españa",
            [("1", "CANTIDAD"), ("cheeseburger", "PRODUCTO"), ("con extra", "MODIFICACION"), ("queso cheddar", "INGREDIENTE"), ("1", "CANTIDAD"), ("papas rusticas", "PRODUCTO")],
            [
                {"producto": "cheeseburger", "producto_normalizado": "Cheeseburger", "cantidad": 1, "ingredientes": ["queso cheddar"]},
                {"producto": "papas rusticas", "producto_normalizado": "Papas Rústicas", "cantidad": 1},
            ],
            "1x Cheeseburger con extra queso cheddar, 1x Papas Rústicas",
        ),
        (
            "train",
            "Al final no voy a llevar postre, solo ponme 2 hamburguesas clásicas sin mayonesa",
            "neutro",
            [("2", "CANTIDAD"), ("hamburguesas clásicas", "PRODUCTO"), ("sin", "NEGACION"), ("mayonesa", "EXCLUSION")],
            [{"producto": "hamburguesas clásicas", "producto_normalizado": "Hamburguesa Simple", "cantidad": 2, "exclusiones": ["mayonesa"]}],
            "2x Hamburguesa Simple sin mayonesa",
        ),
        (
            "train",
            "No mejor quita lo que dije antes, anótame 3 cervezas bien frías y 2 porciones de papas fritas",
            "rioplatense",
            [("3", "CANTIDAD"), ("cervezas", "PRODUCTO"), ("2", "CANTIDAD"), ("porciones", "UNIDAD"), ("papas fritas", "PRODUCTO")],
            [
                {"producto": "cervezas", "producto_normalizado": "Cerveza", "cantidad": 3},
                {"producto": "papas fritas", "producto_normalizado": "Papas Fritas", "cantidad": 2, "unidad": "porciones"},
            ],
            "3x Cerveza, 2x Papas Fritas (porciones)",
        ),
        (
            "val",
            "Mmm bueno ya no dudo más, me das 1 hamburguesa especial cambiando el tomate por huevo frito",
            "neutro",
            [("1", "CANTIDAD"), ("hamburguesa especial", "PRODUCTO"), ("tomate", "SUST_DE"), ("huevo frito", "SUST_A")],
            [{"producto": "hamburguesa especial", "producto_normalizado": "Hamburguesa Especial", "cantidad": 1, "sustituciones": [{"de": "tomate", "a": "huevo frito"}]}],
            "1x Hamburguesa Especial (cambio tomate por huevo frito)",
        ),
        (
            "test",
            "Perdón que dude tanto, al final mándame 2 burger doble sin cebolla y 1 agua sin gas",
            "rioplatense",
            [("2", "CANTIDAD"), ("burger doble", "PRODUCTO"), ("sin", "NEGACION"), ("cebolla", "EXCLUSION"), ("1", "CANTIDAD"), ("agua sin gas", "PRODUCTO")],
            [
                {"producto": "burger doble", "producto_normalizado": "Hamburguesa Doble", "cantidad": 2, "exclusiones": ["cebolla"]},
                {"producto": "agua sin gas", "producto_normalizado": "Agua Mineral", "cantidad": 1},
            ],
            "2x Hamburguesa Doble sin cebolla, 1x Agua Mineral",
        ),
        (
            "test",
            "Ya me decidí por fin, ponme 1 cheeseburger sin ketchup y 1 aros de cebolla porfa",
            "español_españa",
            [("1", "CANTIDAD"), ("cheeseburger", "PRODUCTO"), ("sin", "NEGACION"), ("ketchup", "EXCLUSION"), ("1", "CANTIDAD"), ("aros de cebolla", "PRODUCTO")],
            [
                {"producto": "cheeseburger", "producto_normalizado": "Cheeseburger", "cantidad": 1, "exclusiones": ["ketchup"]},
                {"producto": "aros de cebolla", "producto_normalizado": "Aros de Cebolla", "cantidad": 1},
            ],
            "1x Cheeseburger sin ketchup, 1x Aros de Cebolla",
        ),
    ]
    for sp, txt, dial, spans, items, com in indecisiones_pedido:
        records.append(_make_order_record(next_id(), sp, txt, "indecision_cliente", dial, spans, items, com, tok, ["cliente_indeciso"]))

    # ------------------------------------------------------------------
    # 2. AGREGADOS TARDÍOS (aumentar cosas al pedido luego de un rato)
    # ------------------------------------------------------------------
    agregados_tardios = [
        (
            "train",
            "Hola, hace un rato hice un pedido, ¿le puedo agregar 1 porción de papas fritas y 1 coca cola?",
            "neutro",
            [("1", "CANTIDAD"), ("porción", "UNIDAD"), ("papas fritas", "PRODUCTO"), ("1", "CANTIDAD"), ("coca cola", "PRODUCTO")],
            [
                {"producto": "papas fritas", "producto_normalizado": "Papas Fritas", "cantidad": 1, "unidad": "porción"},
                {"producto": "coca cola", "producto_normalizado": "Coca Cola", "cantidad": 1},
            ],
            "1x Papas Fritas (porción), 1x Coca Cola",
        ),
        (
            "train",
            "Disculpa, me olvidé de pedir bebida en mi orden anterior, súmale 2 sprites por favor",
            "andino",
            [("2", "CANTIDAD"), ("sprites", "PRODUCTO")],
            [{"producto": "sprites", "producto_normalizado": "Sprite", "cantidad": 2}],
            "2x Sprite",
        ),
        (
            "train",
            "Oye, todavía alcanzo a aumentarle 1 hamburguesa simple sin cebolla a mi pedido de hace diez minutos?",
            "mexicano",
            [("1", "CANTIDAD"), ("hamburguesa simple", "PRODUCTO"), ("sin", "NEGACION"), ("cebolla", "EXCLUSION")],
            [{"producto": "hamburguesa simple", "producto_normalizado": "Hamburguesa Simple", "cantidad": 1, "exclusiones": ["cebolla"]}],
            "1x Hamburguesa Simple sin cebolla",
        ),
        (
            "train",
            "Buenas de nuevo, llegó un amigo más, agrégale a mi pedido 1 hamburguesa doble con extra queso y 1 cerveza",
            "rioplatense",
            [("1", "CANTIDAD"), ("hamburguesa doble", "PRODUCTO"), ("con extra", "MODIFICACION"), ("queso", "INGREDIENTE"), ("1", "CANTIDAD"), ("cerveza", "PRODUCTO")],
            [
                {"producto": "hamburguesa doble", "producto_normalizado": "Hamburguesa Doble", "cantidad": 1, "ingredientes": ["queso"]},
                {"producto": "cerveza", "producto_normalizado": "Cerveza", "cantidad": 1},
            ],
            "1x Hamburguesa Doble con extra queso, 1x Cerveza",
        ),
        (
            "train",
            "Perdón que vuelva a escribir, me faltó pedir 1 nuggets de pollo y 1 fanta para mi hermano",
            "neutro",
            [("1", "CANTIDAD"), ("nuggets de pollo", "PRODUCTO"), ("1", "CANTIDAD"), ("fanta", "PRODUCTO")],
            [
                {"producto": "nuggets de pollo", "producto_normalizado": "Nuggets de Pollo", "cantidad": 1},
                {"producto": "fanta", "producto_normalizado": "Fanta", "cantidad": 1},
            ],
            "1x Nuggets de Pollo, 1x Fanta",
        ),
        (
            "train",
            "Hola otra vez, si aún no salió el repartidor añádeme 2 aros de cebolla y 1 agua mineral",
            "español_españa",
            [("2", "CANTIDAD"), ("aros de cebolla", "PRODUCTO"), ("1", "CANTIDAD"), ("agua mineral", "PRODUCTO")],
            [
                {"producto": "aros de cebolla", "producto_normalizado": "Aros de Cebolla", "cantidad": 2},
                {"producto": "agua mineral", "producto_normalizado": "Agua Mineral", "cantidad": 1},
            ],
            "2x Aros de Cebolla, 1x Agua Mineral",
        ),
        (
            "train",
            "Disculpe caserito, a lo que pedí hace ratito auméntele 1 cheeseburger sin mostaza porfa",
            "andino",
            [("1", "CANTIDAD"), ("cheeseburger", "PRODUCTO"), ("sin", "NEGACION"), ("mostaza", "EXCLUSION")],
            [{"producto": "cheeseburger", "producto_normalizado": "Cheeseburger", "cantidad": 1, "exclusiones": ["mostaza"]}],
            "1x Cheeseburger sin mostaza",
        ),
        (
            "train",
            "Hola, ya hice mi pedido hace un momento pero quiero sumarle 2 latas de cerveza y 1 papas rusticas",
            "neutro",
            [("2", "CANTIDAD"), ("latas", "UNIDAD"), ("cerveza", "PRODUCTO"), ("1", "CANTIDAD"), ("papas rusticas", "PRODUCTO")],
            [
                {"producto": "cerveza", "producto_normalizado": "Cerveza", "cantidad": 2, "unidad": "latas"},
                {"producto": "papas rusticas", "producto_normalizado": "Papas Rústicas", "cantidad": 1},
            ],
            "2x Cerveza (latas), 1x Papas Rústicas",
        ),
        (
            "train",
            "Antes de que cierren mi comanda, métanle también 1 hamburguesa especial sin lechuga por favor",
            "mexicano",
            [("1", "CANTIDAD"), ("hamburguesa especial", "PRODUCTO"), ("sin", "NEGACION"), ("lechuga", "EXCLUSION")],
            [{"producto": "hamburguesa especial", "producto_normalizado": "Hamburguesa Especial", "cantidad": 1, "exclusiones": ["lechuga"]}],
            "1x Hamburguesa Especial sin lechuga",
        ),
        (
            "val",
            "Hola buenas, hace 15 minutos ordené comida, ¿me puedes sumar 2 coca zero y 1 papas fritas?",
            "neutro",
            [("2", "CANTIDAD"), ("coca zero", "PRODUCTO"), ("1", "CANTIDAD"), ("papas fritas", "PRODUCTO")],
            [
                {"producto": "coca zero", "producto_normalizado": "Coca Cola", "cantidad": 2},
                {"producto": "papas fritas", "producto_normalizado": "Papas Fritas", "cantidad": 1},
            ],
            "2x Coca Cola, 1x Papas Fritas",
        ),
        (
            "test",
            "Che disculpá que joda de nuevo, agregame 1 burger con queso y 1 birra al pedido que te pasé recién",
            "rioplatense",
            [("1", "CANTIDAD"), ("burger con queso", "PRODUCTO"), ("1", "CANTIDAD"), ("birra", "PRODUCTO")],
            [
                {"producto": "burger con queso", "producto_normalizado": "Cheeseburger", "cantidad": 1},
                {"producto": "birra", "producto_normalizado": "Cerveza", "cantidad": 1},
            ],
            "1x Cheeseburger, 1x Cerveza",
        ),
        (
            "test",
            "Se me olvidó añadir algo a mi pedido anterior: ponme también 1 porción de papas fritas sin sal",
            "neutro",
            [("1", "CANTIDAD"), ("porción", "UNIDAD"), ("papas fritas", "PRODUCTO"), ("sin", "NEGACION"), ("sal", "EXCLUSION")],
            [{"producto": "papas fritas", "producto_normalizado": "Papas Fritas", "cantidad": 1, "unidad": "porción", "exclusiones": ["sal"]}],
            "1x Papas Fritas (porción) sin sal",
        ),
    ]
    for sp, txt, dial, spans, items, com in agregados_tardios:
        records.append(_make_order_record(next_id(), sp, txt, "agregado_tardio", dial, spans, items, com, tok, ["agregado_a_pedido_existente"]))

    # ------------------------------------------------------------------
    # 3. PEDIDOS NATURALES (sin cantidad explícita, diminutivos, complejos)
    # ------------------------------------------------------------------
    pedidos_naturales = [
        (
            "train",
            "Buenas, quiero hamburguesa doble sin cebolla y papas fritas con extra queso",
            "neutro",
            [("hamburguesa doble", "PRODUCTO"), ("sin", "NEGACION"), ("cebolla", "EXCLUSION"), ("papas fritas", "PRODUCTO"), ("con extra", "MODIFICACION"), ("queso", "INGREDIENTE")],
            [
                {"producto": "hamburguesa doble", "producto_normalizado": "Hamburguesa Doble", "cantidad": 1, "exclusiones": ["cebolla"]},
                {"producto": "papas fritas", "producto_normalizado": "Papas Fritas", "cantidad": 1, "ingredientes": ["queso"]},
            ],
            "1x Hamburguesa Doble sin cebolla, 1x Papas Fritas con extra queso",
        ),
        (
            "train",
            "Mándame unas papitas, dos cocas y una hamburguesa completa sin tomate porfa",
            "andino",
            [("unas", "CANTIDAD"), ("papitas", "PRODUCTO"), ("dos", "CANTIDAD"), ("cocas", "PRODUCTO"), ("una", "CANTIDAD"), ("hamburguesa completa", "PRODUCTO"), ("sin", "NEGACION"), ("tomate", "EXCLUSION")],
            [
                {"producto": "papitas", "producto_normalizado": "Papas Fritas", "cantidad": 1},
                {"producto": "cocas", "producto_normalizado": "Coca Cola", "cantidad": 2},
                {"producto": "hamburguesa completa", "producto_normalizado": "Hamburguesa Especial", "cantidad": 1, "exclusiones": ["tomate"]},
            ],
            "1x Papas Fritas, 2x Coca Cola, 1x Hamburguesa Especial sin tomate",
        ),
        (
            "train",
            "Hola, me preparas cheeseburger sin pepinillos y cerveza bien helada",
            "mexicano",
            [("cheeseburger", "PRODUCTO"), ("sin", "NEGACION"), ("pepinillos", "EXCLUSION"), ("cerveza", "PRODUCTO")],
            [
                {"producto": "cheeseburger", "producto_normalizado": "Cheeseburger", "cantidad": 1, "exclusiones": ["pepinillos"]},
                {"producto": "cerveza", "producto_normalizado": "Cerveza", "cantidad": 1},
            ],
            "1x Cheeseburger sin pepinillos, 1x Cerveza",
        ),
        (
            "train",
            "Para llevar por favor: tres burgers simples, dos aros de cebolla y tres fantas",
            "neutro",
            [("tres", "CANTIDAD"), ("burgers simples", "PRODUCTO"), ("dos", "CANTIDAD"), ("aros de cebolla", "PRODUCTO"), ("tres", "CANTIDAD"), ("fantas", "PRODUCTO")],
            [
                {"producto": "burgers simples", "producto_normalizado": "Hamburguesa Simple", "cantidad": 3},
                {"producto": "aros de cebolla", "producto_normalizado": "Aros de Cebolla", "cantidad": 2},
                {"producto": "fantas", "producto_normalizado": "Fanta", "cantidad": 3},
            ],
            "3x Hamburguesa Simple, 2x Aros de Cebolla, 3x Fanta",
        ),
        (
            "train",
            "Me traés una hamburguesa especial cambiando la lechuga por tocino y una agua con gas?",
            "rioplatense",
            [("una", "CANTIDAD"), ("hamburguesa especial", "PRODUCTO"), ("lechuga", "SUST_DE"), ("tocino", "SUST_A"), ("una", "CANTIDAD"), ("agua con gas", "PRODUCTO")],
            [
                {"producto": "hamburguesa especial", "producto_normalizado": "Hamburguesa Especial", "cantidad": 1, "sustituciones": [{"de": "lechuga", "a": "tocino"}]},
                {"producto": "agua con gas", "producto_normalizado": "Agua Mineral", "cantidad": 1},
            ],
            "1x Hamburguesa Especial (cambio lechuga por tocino), 1x Agua Mineral",
        ),
        (
            "train",
            "Quisiera 2 hamburguesas con queso sin cebolla, 1 nuggets de pollo con extra salsa barbacoa y 2 coca light",
            "neutro",
            [("2", "CANTIDAD"), ("hamburguesas con queso", "PRODUCTO"), ("sin", "NEGACION"), ("cebolla", "EXCLUSION"), ("1", "CANTIDAD"), ("nuggets de pollo", "PRODUCTO"), ("con extra", "MODIFICACION"), ("salsa barbacoa", "INGREDIENTE"), ("2", "CANTIDAD"), ("coca light", "PRODUCTO")],
            [
                {"producto": "hamburguesas con queso", "producto_normalizado": "Cheeseburger", "cantidad": 2, "exclusiones": ["cebolla"]},
                {"producto": "nuggets de pollo", "producto_normalizado": "Nuggets de Pollo", "cantidad": 1, "ingredientes": ["salsa barbacoa"]},
                {"producto": "coca light", "producto_normalizado": "Coca Cola", "cantidad": 2},
            ],
            "2x Cheeseburger sin cebolla, 1x Nuggets de Pollo con extra salsa barbacoa, 2x Coca Cola",
        ),
        (
            "train",
            "Ponme patatas fritas sin sal y dos chelas por favor",
            "español_españa",
            [("patatas fritas", "PRODUCTO"), ("sin", "NEGACION"), ("sal", "EXCLUSION"), ("dos", "CANTIDAD"), ("chelas", "PRODUCTO")],
            [
                {"producto": "patatas fritas", "producto_normalizado": "Papas Fritas", "cantidad": 1, "exclusiones": ["sal"]},
                {"producto": "chelas", "producto_normalizado": "Cerveza", "cantidad": 2},
            ],
            "1x Papas Fritas sin sal, 2x Cerveza",
        ),
        (
            "train",
            "Deme 1 hamburguesa clásica nada de picante y 1 sprite sin hielo por favorcito",
            "andino",
            [("1", "CANTIDAD"), ("hamburguesa clásica", "PRODUCTO"), ("nada de", "NEGACION"), ("picante", "EXCLUSION"), ("1", "CANTIDAD"), ("sprite", "PRODUCTO"), ("sin", "NEGACION"), ("hielo", "EXCLUSION")],
            [
                {"producto": "hamburguesa clásica", "producto_normalizado": "Hamburguesa Simple", "cantidad": 1, "exclusiones": ["picante"]},
                {"producto": "sprite", "producto_normalizado": "Sprite", "cantidad": 1, "exclusiones": ["hielo"]},
            ],
            "1x Hamburguesa Simple sin picante, 1x Sprite sin hielo",
        ),
        (
            "val",
            "Hola, quiero nuggets de pollo y una coca-cola sin hielo",
            "neutro",
            [("nuggets de pollo", "PRODUCTO"), ("una", "CANTIDAD"), ("coca-cola", "PRODUCTO"), ("sin", "NEGACION"), ("hielo", "EXCLUSION")],
            [
                {"producto": "nuggets de pollo", "producto_normalizado": "Nuggets de Pollo", "cantidad": 1},
                {"producto": "coca-cola", "producto_normalizado": "Coca Cola", "cantidad": 1, "exclusiones": ["hielo"]},
            ],
            "1x Nuggets de Pollo, 1x Coca Cola sin hielo",
        ),
        (
            "test",
            "Buenas noches, me das 2 hamburguesas dobles sin mayonesa y 1 papas al horno",
            "mexicano",
            [("2", "CANTIDAD"), ("hamburguesas dobles", "PRODUCTO"), ("sin", "NEGACION"), ("mayonesa", "EXCLUSION"), ("1", "CANTIDAD"), ("papas al horno", "PRODUCTO")],
            [
                {"producto": "hamburguesas dobles", "producto_normalizado": "Hamburguesa Doble", "cantidad": 2, "exclusiones": ["mayonesa"]},
                {"producto": "papas al horno", "producto_normalizado": "Papas Rústicas", "cantidad": 1},
            ],
            "2x Hamburguesa Doble sin mayonesa, 1x Papas Rústicas",
        ),
    ]
    for sp, txt, dial, spans, items, com in pedidos_naturales:
        records.append(_make_order_record(next_id(), sp, txt, "pedido_natural_variado", dial, spans, items, com, tok))

    # ------------------------------------------------------------------
    # 4. CANCELACIONES DE PEDIDOS (reclamo / consulta según el tono)
    # ------------------------------------------------------------------
    cancelaciones = [
        ("train", "Hola, necesito cancelar mi pedido por favor, surgió un imprevisto", "reclamo", "cancelacion_pedido", "neutro"),
        ("train", "Disculpa, anula la orden que hice hace 5 minutos porque ya no voy a estar en casa", "reclamo", "cancelacion_pedido", "neutro"),
        ("train", "Ya no quiero nada, cancelen mi pedido completo porfa", "reclamo", "cancelacion_pedido", "mexicano"),
        ("train", "Che cancelame el pedido de recién, me equivoqué de local", "reclamo", "cancelacion_pedido", "rioplatense"),
        ("train", "Por favor anule mi pedido anterior, tuve una emergencia familiar", "reclamo", "cancelacion_pedido", "andino"),
        ("train", "Cancela las hamburguesas que pedí, está tardando demasiado y ya me voy", "reclamo", "cancelacion_pedido", "español_españa"),
        ("train", "Mejor cancelamos todo, ya pasaron 40 minutos y nadie confirma", "reclamo", "cancelacion_pedido", "neutro"),
        ("train", "Quiero dar de baja mi orden de hoy, me equivoqué en los productos", "reclamo", "cancelacion_pedido", "neutro"),
        ("train", "No envíen nada por favor, necesito anular el pedido urgente", "reclamo", "cancelacion_pedido", "andino"),
        ("train", "Oye, ¿todavía estoy a tiempo de cancelar mi pedido sin cargo?", "consulta", "consulta_cancelacion", "mexicano"),
        ("train", "Disculpen, ¿cómo puedo cancelar una orden que acabo de enviar?", "consulta", "consulta_cancelacion", "neutro"),
        ("train", "Si cancelo ahora me devuelven el dinero de la tarjeta?", "consulta", "consulta_cancelacion", "rioplatense"),
        ("val", "Hola, por favor cancela mi orden anterior, ya no la voy a necesitar", "reclamo", "cancelacion_pedido", "neutro"),
        ("val", "Buenas, ¿se puede anular un pedido si todavía no salió de cocina?", "consulta", "consulta_cancelacion", "neutro"),
        ("test", "Disculpa las molestias pero cancela todo mi pedido, me surgió algo urgente", "reclamo", "cancelacion_pedido", "neutro"),
        ("test", "Hola, ¿hasta cuántos minutos después puedo cancelar mi pedido?", "consulta", "consulta_cancelacion", "español_españa"),
    ]
    for sp, txt, intent, esc, dial in cancelaciones:
        records.append(_make_non_order_record(next_id(), sp, txt, intent, esc, dial, tok))

    # ------------------------------------------------------------------
    # 5. INDECISIONES EN CONSULTA (pidiendo recomendaciones / precios)
    # ------------------------------------------------------------------
    indecisiones_consulta = [
        ("train", "Hola, no sé qué pedir hoy, ¿cuál hamburguesa me recomiendan?", "consulta", "indecision_consulta", "neutro"),
        ("train", "Estoy dudando entre la hamburguesa doble y la especial, ¿cuál llena más?", "consulta", "indecision_consulta", "neutro"),
        ("train", "No me decido por las papas, ¿las rústicas son picantes?", "consulta", "indecision_consulta", "mexicano"),
        ("train", "Che qué me sugerís si no me gusta el pepinillo ni la mostaza?", "consulta", "indecision_consulta", "rioplatense"),
        ("train", "Mmm todavía estoy viendo el menú, ¿qué trae la hamburguesa especial exactamente?", "consulta", "indecision_consulta", "andino"),
        ("train", "No sé si pedir nuggets o aros de cebolla, ¿cuántas unidades trae cada porción?", "consulta", "indecision_consulta", "neutro"),
        ("train", "Estoy indeciso con la bebida, ¿tienen algo sin azúcar aparte de agua?", "consulta", "indecision_consulta", "español_españa"),
        ("train", "Espera, antes de pedir, ¿puedo cambiar ingredientes sin costo extra?", "consulta", "indecision_consulta", "neutro"),
        ("val", "No sé cuál elegir entre cheeseburger o clásica, ¿la clásica trae queso?", "consulta", "indecision_consulta", "neutro"),
        ("test", "Estoy dudando mucho, ¿qué combo o plato es el más vendido?", "consulta", "indecision_consulta", "mexicano"),
    ]
    for sp, txt, intent, esc, dial in indecisiones_consulta:
        records.append(_make_non_order_record(next_id(), sp, txt, intent, esc, dial, tok))

    # ------------------------------------------------------------------
    # 6. CONSULTAS, RECLAMOS Y SALUDOS REALISTAS ADICIONALES
    # ------------------------------------------------------------------
    otros_reales = [
        # Consultas variadas
        ("train", "¿Hacen entregas hasta la zona norte o solo centro?", "consulta", "consulta_delivery", "neutro"),
        ("train", "Buenas, ¿el pago contra entrega puede ser con QR o solo efectivo?", "consulta", "consulta_pago", "andino"),
        ("train", "¿Cuánto cuesta la hamburguesa doble con papas fritas?", "consulta", "consulta_precio", "neutro"),
        ("train", "Hola, ¿abren los domingos por la noche?", "consulta", "consulta_horario", "rioplatense"),
        ("train", "¿Tienen pan sin gluten para personas celíacas?", "consulta", "consulta_alergenos", "español_españa"),
        ("train", "Disculpe, ¿mi pedido ya salió con el repartidor?", "consulta", "consulta_estado", "neutro"),
        ("train", "¿Puedo pasar a recoger mi orden directo en el local?", "consulta", "consulta_retiro", "mexicano"),
        ("val", "¿Aceptan transferencia bancaria al momento de entregar?", "consulta", "consulta_pago", "neutro"),
        ("test", "Hola, ¿en cuánto tiempo más o menos llega el delivery?", "consulta", "consulta_tiempo", "neutro"),
        # Reclamos variados
        ("train", "Me mandaron fanta en vez de coca cola y las papas llegaron heladas", "reclamo", "reclamo_equivocado", "neutro"),
        ("train", "Pedí expresamente sin tomate porque soy alérgico y igual le pusieron", "reclamo", "reclamo_alergia", "neutro"),
        ("train", "El repartidor se perdió y nunca me llamó, llevo hora y media esperando", "reclamo", "reclamo_demora", "rioplatense"),
        ("train", "Me faltaron los aros de cebolla en la bolsa y sí me los cobraron", "reclamo", "reclamo_faltante", "mexicano"),
        ("train", "La carne de la hamburguesa llegó cruda por dentro, qué mal servicio", "reclamo", "reclamo_calidad", "andino"),
        ("train", "Se derramó toda la bebida encima de la caja de las hamburguesas", "reclamo", "reclamo_empaque", "español_españa"),
        ("train", "No me llegó el vuelto correcto y tampoco trajeron servilletas", "reclamo", "reclamo_cobro", "neutro"),
        ("val", "La bolsa vino abierta y falta una hamburguesa simple de las tres que pagué", "reclamo", "reclamo_faltante", "neutro"),
        ("test", "Hace más de una hora que confirmaron mi pedido y sigue sin llegar nada", "reclamo", "reclamo_demora", "neutro"),
        # Saludos y pausas conversacionales
        ("train", "Hola buenas tardes, dame un minuto que estoy viendo la carta", "saludo_otro", "pausa_menu", "neutro"),
        ("train", "Buenas, enseguida te paso lo que vamos a pedir", "saludo_otro", "pausa_menu", "rioplatense"),
        ("train", "Muchas gracias por la atención, estuvo todo riquísimo", "saludo_otro", "agradecimiento", "neutro"),
        ("train", "Listo, ya recibí el pedido, todo perfecto gracias!", "saludo_otro", "confirmacion_entrega", "mexicano"),
        ("train", "Hola! Solo pasaba a saludar, más tarde hago mi pedido", "saludo_otro", "saludo_simple", "andino"),
        ("train", "Ok perfecto, quedo atento entonces, gracias", "saludo_otro", "cierre_conversacion", "español_españa"),
        ("train", "Un segundo porfa que le estoy preguntando a mi familia qué quieren", "saludo_otro", "pausa_menu", "neutro"),
        ("val", "Dale, en cinco minutos te confirmo qué vamos a cenar", "saludo_otro", "pausa_menu", "rioplatense"),
        ("test", "Todo excelente como siempre, muchas gracias y buenas noches", "saludo_otro", "agradecimiento", "neutro"),
    ]
    for sp, txt, intent, esc, dial in otros_reales:
        records.append(_make_non_order_record(next_id(), sp, txt, intent, esc, dial, tok))

    return records


def main() -> None:
    records = build_expanded_gold()
    out_path = os.path.join(CURRENT_DIR, "dataset", "gold_expanded.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    by_split = {}
    by_intent = {}
    for r in records:
        by_split[r["split"]] = by_split.get(r["split"], 0) + 1
        by_intent[r["intencion"]] = by_intent.get(r["intencion"], 0) + 1

    print(f"Generado {out_path}: {len(records)} ejemplos reales ampliados")
    print(f"  Por split    : {by_split}")
    print(f"  Por intención: {by_intent}")


if __name__ == "__main__":
    main()
