"""
ETAPA 3 · Pipeline Integrador End-to-End para Pedidos de Restaurante.
Orquesta el preprocesamiento, clasificación de intención (Bi-GRU), extracción de entidades (Bi-GRU + CRF),
vinculación de relaciones sintácticas y formateo de comanda para cocina.
"""

import os
import sys
import json
import torch
from typing import Dict, Any, Optional

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing.tokenizer import SimpleSpanishTokenizer
from src.preprocessing.vocabulary import Vocabulary, TagVocabulary
from src.models.intent_classifier import BiGRUIntentClassifier
from src.models.ner_tagger import BiGRUNERTagger
from src.models.relation_extractor import OrderRelationExtractor
from src.structuring.parser import OrderParser


class OrderExtractionPipeline:
    def __init__(self, models_dir: Optional[str] = None, ontology_path: Optional[str] = None, device: str = "cpu"):
        self.device = device
        self.models_dir = models_dir or os.path.join(PROJECT_ROOT, "models_saved")
        self.ontology_path = ontology_path or os.path.join(PROJECT_ROOT, "data", "menu_ontology.json")
        
        # 1. Cargar Ontología
        with open(self.ontology_path, "r", encoding="utf-8") as f:
            self.ontology = json.load(f)
            
        # 2. Inicializar componentes auxiliares
        self.tokenizer = SimpleSpanishTokenizer()
        self.relation_extractor = OrderRelationExtractor(self.ontology)
        self.parser = OrderParser(self.ontology)
        
        # 3. Cargar Vocabularios
        vocab_file = os.path.join(self.models_dir, "vocab.json")
        tag_vocab_file = os.path.join(self.models_dir, "tag_vocab.json")
        intent_map_file = os.path.join(self.models_dir, "intent_map.json")
        
        self.vocab = Vocabulary.load(vocab_file)
        self.tag_vocab = TagVocabulary.load(tag_vocab_file)
        with open(intent_map_file, "r", encoding="utf-8") as f:
            self.intent_map = json.load(f)
        self.id2intent = {v: k for k, v in self.intent_map.items()}
        
        # 4. Cargar Modelo de Intención
        self.intent_model = BiGRUIntentClassifier(
            vocab_size=len(self.vocab),
            embedding_dim=128,
            hidden_dim=128,
            num_classes=len(self.intent_map),
            num_layers=2,
            dropout=0.0,
            pad_idx=self.vocab.pad_idx
        ).to(self.device)
        intent_weights = os.path.join(self.models_dir, "intent_model.pt")
        self.intent_model.load_state_dict(torch.load(intent_weights, map_location=self.device))
        self.intent_model.eval()
        
        # 5. Cargar Modelo NER + CRF
        self.ner_model = BiGRUNERTagger(
            vocab_size=len(self.vocab),
            num_tags=len(self.tag_vocab),
            embedding_dim=128,
            hidden_dim=128,
            num_layers=2,
            dropout=0.0,
            pad_idx=self.vocab.pad_idx
        ).to(self.device)
        ner_weights = os.path.join(self.models_dir, "ner_model.pt")
        self.ner_model.load_state_dict(torch.load(ner_weights, map_location=self.device))
        self.ner_model.eval()

    def process(self, text: str) -> Dict[str, Any]:
        """
        Procesa texto crudo de un cliente y retorna la comanda sintetizada y el contrato JSON.
        """
        tokens = self.tokenizer.tokenize(text)
        if not tokens:
            return {
                "texto_original": text,
                "intencion": "saludo_otro",
                "tokens": [],
                "bio_tags": [],
                "contrato_json": {"intencion": "saludo_otro", "items": []},
                "comanda_cocina": "[VACIO]"
            }
            
        input_ids = torch.tensor([self.vocab.encode(tokens)], dtype=torch.long, device=self.device)
        
        # 1. Clasificación de Intención
        with torch.no_grad():
            intent_pred_id = self.intent_model.predict(input_ids).item()
            intent_str = self.id2intent.get(intent_pred_id, "pedido")
            
        # 2. Si no es pedido, derivar directamente
        if intent_str != "pedido":
            bio_tags = ["O"] * len(tokens)
            contract = {
                "intencion": intent_str,
                "items": [],
                "restricciones_globales": [],
                "notas": []
            }
            kitchen_line = f"[{intent_str.upper()}]"
            return {
                "texto_original": text,
                "intencion": intent_str,
                "tokens": tokens,
                "bio_tags": bio_tags,
                "contrato_json": contract,
                "comanda_cocina": kitchen_line
            }
            
        # 3. Extracción de Entidades con Bi-GRU + CRF (Viterbi)
        with torch.no_grad():
            paths = self.ner_model.decode(input_ids)
            pred_tag_ids = paths[0]
            bio_tags = self.tag_vocab.decode(pred_tag_ids)
            # Asegurar misma longitud que tokens
            bio_tags = bio_tags[:len(tokens)]
            if len(bio_tags) < len(tokens):
                bio_tags.extend(["O"] * (len(tokens) - len(bio_tags)))
                
        # 4. Vinculación de Relaciones
        raw_items = self.relation_extractor.extract_relations(tokens, bio_tags)
        
        # 5. Normalización y Contrato JSON
        contract = self.parser.build_contract_json(intent_str, raw_items)
        
        # 6. Formateo de Comanda para Cocina
        kitchen_line = self.parser.format_kitchen_line(contract)
        
        return {
            "texto_original": text,
            "intencion": intent_str,
            "tokens": tokens,
            "bio_tags": bio_tags,
            "contrato_json": contract,
            "comanda_cocina": kitchen_line
        }


if __name__ == "__main__":
    test_phrase = "Hola buenas tardes, disculpe la molestia, quisiera pedir 2 hamburguesas con doble carne y sin cebolla, con una porción de papas por favor"
    print("Probando Pipeline con frase:", test_phrase)
    pipe = OrderExtractionPipeline()
    result = pipe.process(test_phrase)
    print("\n--- RESULTADO PIPELINE ---")
    print("Intención:", result["intencion"])
    print("Comanda Cocina:", result["comanda_cocina"])
    print("Contrato JSON:", json.dumps(result["contrato_json"], ensure_ascii=False, indent=2))
