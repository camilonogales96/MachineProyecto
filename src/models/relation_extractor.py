"""
Capa de Relaciones y Vinculación de Entidades.
Asocia CANTIDADES, UNIDADES, EXCLUSIONES, INGREDIENTES y SUSTITUCIONES a su PRODUCTO correspondiente
en función de estructura sintáctica, cercanía y cláusulas conjuntivas.
"""

from typing import List, Dict, Any, Tuple, Optional


def extract_chunks(tokens: List[str], tags: List[str]) -> List[Dict[str, Any]]:
    """
    Convierte una secuencia de tags BIO en una lista de entidades (chunks) con sus posiciones.
    Soporta: PRODUCTO, CANTIDAD, UNIDAD, NEGACION, EXCLUSION, INGREDIENTE, MODIFICACION, SUST_DE, SUST_A.
    """
    chunks = []
    curr_type = None
    curr_tokens = []
    start_idx = 0
    
    for i, (tok, tag) in enumerate(zip(tokens, tags)):
        if tag.startswith("B-"):
            if curr_type:
                chunks.append({
                    "type": curr_type,
                    "text": " ".join(curr_tokens),
                    "tokens": curr_tokens,
                    "start": start_idx,
                    "end": i
                })
            curr_type = tag[2:]
            curr_tokens = [tok]
            start_idx = i
        elif tag.startswith("I-"):
            tag_type = tag[2:]
            if curr_type == tag_type:
                curr_tokens.append(tok)
            else:
                if curr_type:
                    chunks.append({
                        "type": curr_type,
                        "text": " ".join(curr_tokens),
                        "tokens": curr_tokens,
                        "start": start_idx,
                        "end": i
                    })
                curr_type = tag_type
                curr_tokens = [tok]
                start_idx = i
        else: # 'O'
            if curr_type:
                chunks.append({
                    "type": curr_type,
                    "text": " ".join(curr_tokens),
                    "tokens": curr_tokens,
                    "start": start_idx,
                    "end": i
                })
                curr_type = None
                curr_tokens = []
                
    if curr_type:
        chunks.append({
            "type": curr_type,
            "text": " ".join(curr_tokens),
            "tokens": curr_tokens,
            "start": start_idx,
            "end": len(tokens)
        })
        
    return chunks


class OrderRelationExtractor:
    """
    Vincula las entidades extraídas para estructurar los items de un pedido.
    """
    def __init__(self, ontology: Optional[Dict[str, Any]] = None):
        self.ontology = ontology or {}
        self.cantidades_lexicas = self.ontology.get("cantidades_lexicas", {
            "un": 1, "uno": 1, "una": 1, "dos": 2, "tres": 3, "cuatro": 4, "cinco": 5, "seis": 6,
            "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6
        })

    def parse_cantidad(self, cant_text: str) -> int:
        cant_text = cant_text.lower().strip()
        if cant_text.isdigit():
            return int(cant_text)
        return self.cantidades_lexicas.get(cant_text, 1)

    def extract_relations(self, tokens: List[str], bio_tags: List[str]) -> List[Dict[str, Any]]:
        chunks = extract_chunks(tokens, bio_tags)
        
        product_chunks = [c for c in chunks if c["type"] == "PRODUCTO"]
        if not product_chunks:
            return []
            
        items = []
        for i, prod in enumerate(product_chunks):
            prev_prod_end = product_chunks[i - 1]["end"] if i > 0 else 0
            next_prod_start = product_chunks[i + 1]["start"] if i + 1 < len(product_chunks) else len(tokens)
            
            cantidad = 1
            unidad = None
            exclusiones = []
            ingredientes = []
            sust_de = None
            sust_a = None
            sustituciones = []
            
            scope_chunks = [c for c in chunks if c["start"] >= prev_prod_end and c["end"] <= next_prod_start and c != prod]
            
            for sc in scope_chunks:
                t = sc["type"]
                txt = sc["text"]
                if t == "CANTIDAD":
                    cantidad = self.parse_cantidad(txt)
                elif t == "UNIDAD":
                    unidad = txt
                elif t == "EXCLUSION":
                    exclusiones.append(txt)
                elif t in ("INGREDIENTE", "MODIFICACION"):
                    ingredientes.append(txt)
                elif t == "SUST_DE":
                    sust_de = txt
                elif t == "SUST_A":
                    sust_a = txt
                    
            if sust_de and sust_a:
                sustituciones.append({"de": sust_de, "a": sust_a})
                
            items.append({
                "producto": prod["text"],
                "cantidad": cantidad,
                "unidad": unidad,
                "ingredientes": ingredientes,
                "modificaciones": [],
                "exclusiones": exclusiones,
                "sustituciones": sustituciones
            })
            
        return items
