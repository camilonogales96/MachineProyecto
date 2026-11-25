"""
ETAPA 3 · Estructuración y Normalización contra Ontología del Menú
Genera el Contrato JSON (fuente de verdad) y la línea sintetizada legible para cocina/punto de venta.
Soporta sustituciones de ingredientes.
"""

import json
from typing import List, Dict, Any, Optional


class OrderParser:
    def __init__(self, ontology: Dict[str, Any]):
        self.ontology = ontology
        self._build_lookup()

    def _build_lookup(self):
        self.synonym_to_canonical = {}
        for cat, items in self.ontology.get("categorias", {}).items():
            for item in items:
                canonical = item["nombre"]
                for syn in item["sinonimos"]:
                    self.synonym_to_canonical[syn.lower().strip()] = canonical
                self.synonym_to_canonical[canonical.lower().strip()] = canonical

    def normalize_product_name(self, raw_name: str) -> str:
        raw_clean = raw_name.lower().strip()
        if raw_clean in self.synonym_to_canonical:
            return self.synonym_to_canonical[raw_clean]
        
        for syn, canonical in self.synonym_to_canonical.items():
            if syn in raw_clean or raw_clean in syn:
                return canonical
                
        return raw_name.title()

    def build_contract_json(self, intent: str, raw_items: List[Dict[str, Any]]) -> Dict[str, Any]:
        normalized_items = []
        for it in raw_items:
            norm_name = self.normalize_product_name(it["producto"])
            item_entry = {
                "producto": it["producto"],
                "producto_normalizado": norm_name,
                "cantidad": it.get("cantidad", 1),
                "unidad": it.get("unidad"),
                "ingredientes": it.get("ingredientes", []),
                "modificaciones": it.get("modificaciones", []),
                "exclusiones": it.get("exclusiones", []),
                "sustituciones": it.get("sustituciones", []),
                "restricciones_globales": []
            }
            normalized_items.append(item_entry)
            
        contract = {
            "intencion": intent,
            "items": normalized_items,
            "restricciones_globales": [],
            "notas": []
        }
        return contract

    def format_kitchen_line(self, contract: Dict[str, Any]) -> str:
        if contract.get("intencion") != "pedido" or not contract.get("items"):
            return f"[{contract.get('intencion', 'sin_pedido').upper()}]"
            
        parts = []
        for it in contract["items"]:
            desc = f"{it['cantidad']}x {it['producto_normalizado']}"
            if it.get("unidad"):
                desc += f" ({it['unidad']})"
            if it.get("exclusiones"):
                desc += f" sin {', '.join(it['exclusiones'])}"
            if it.get("ingredientes"):
                desc += f" con extra {', '.join(it['ingredientes'])}"
            if it.get("sustituciones"):
                for s in it["sustituciones"]:
                    desc += f" (cambio {s['de']} por {s['a']})"
            parts.append(desc)
            
        return ", ".join(parts)
