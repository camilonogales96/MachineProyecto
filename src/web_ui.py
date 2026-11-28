"""
Servidor web ligero (sin dependencias externas) para la interfaz gráfica
del Sistema de Extracción y Síntesis de Pedidos de Restaurante.

Endpoints:
- GET  /              -> Interfaz web (src/ui/index.html)
- GET  /api/status    -> Estado del modelo, vocabulario, estadísticas del dataset y métricas guardadas
- POST /api/predict   -> Ejecuta OrderExtractionPipeline sobre {"texto": "..."}
- POST /api/evaluate  -> Evalúa el modelo en data/dataset/test.json y guarda models_saved/eval_metrics.json
"""

import collections
import json
import os
import sys
import time
import webbrowser
import subprocess
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

UI_HTML_PATH = os.path.join(CURRENT_DIR, "ui", "index.html")
DATASET_DIR = os.path.join(PROJECT_ROOT, "data", "dataset")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models_saved")
ONTOLOGY_PATH = os.path.join(PROJECT_ROOT, "data", "menu_ontology.json")
EVAL_CACHE_PATH = os.path.join(MODELS_DIR, "eval_metrics.json")

INTENT_CLASSES = ["pedido", "consulta", "reclamo", "saludo_otro"]


def compute_f_beta(precision: float, recall: float, beta: float) -> float:
    """Calcula F_beta = ((1 + beta^2) * P * R) / (beta^2 * P + R)."""
    b2 = beta * beta
    denom = b2 * precision + recall
    if denom <= 0.0:
        return 0.0
    return ((1.0 + b2) * precision * recall) / denom


def compute_classification_metrics(
    cm: List[List[int]],
    labels: List[str],
) -> Dict[str, Any]:
    """Calcula las 7 métricas de clasificación desde la matriz de confusión (One-vs-Rest + macro)."""
    total = sum(sum(row) for row in cm)
    correct = sum(cm[i][i] for i in range(len(labels)))
    accuracy = (correct / total) if total > 0 else 0.0
    error_rate = 1.0 - accuracy if total > 0 else 0.0

    per_class: Dict[str, Dict[str, Any]] = {}
    for idx, cls_name in enumerate(labels):
        tp = cm[idx][idx]
        fn = sum(cm[idx]) - tp
        fp = sum(cm[r][idx] for r in range(len(labels))) - tp
        tn = total - tp - fp - fn
        support = tp + fn

        precision = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0  # Sensibilidad / TPR
        specificity = (tn / (tn + fp)) if (tn + fp) > 0 else 0.0  # Especificidad / TNR
        f1 = compute_f_beta(precision, recall, 1.0)
        f05 = compute_f_beta(precision, recall, 0.5)
        f2 = compute_f_beta(precision, recall, 2.0)

        per_class[cls_name] = {
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "tn": int(tn),
            "support": int(support),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "specificity": round(specificity, 4),
            "f1": round(f1, 4),
            "f0_5": round(f05, 4),
            "f2": round(f2, 4),
        }

    n_cls = max(1, len(labels))
    macro_precision = sum(c["precision"] for c in per_class.values()) / n_cls
    macro_recall = sum(c["recall"] for c in per_class.values()) / n_cls
    macro_specificity = sum(c["specificity"] for c in per_class.values()) / n_cls
    macro_f1 = sum(c["f1"] for c in per_class.values()) / n_cls
    macro_f05 = sum(c["f0_5"] for c in per_class.values()) / n_cls
    macro_f2 = sum(c["f2"] for c in per_class.values()) / n_cls

    return {
        "intent_accuracy": round(accuracy, 4),
        "intent_error_rate": round(error_rate, 4),
        "intent_precision_macro": round(macro_precision, 4),
        "intent_recall_macro": round(macro_recall, 4),
        "intent_specificity_macro": round(macro_specificity, 4),
        "intent_f1_macro": round(macro_f1, 4),
        "intent_f0_5_macro": round(macro_f05, 4),
        "intent_f2_macro": round(macro_f2, 4),
        "per_class": per_class,
    }


class PipelineService:
    """Gestiona la carga del pipeline PyTorch, estadísticas del dataset y evaluación."""

    def __init__(self, device: str = "cpu"):
        self.device = device
        self.pipeline = None
        self.load_error: Optional[str] = None
        self.lstm_process = None
        self.lstm_output = ""
        self._load_pipeline()

    def _load_pipeline(self) -> None:
        try:
            from src.pipeline import OrderExtractionPipeline

            self.pipeline = OrderExtractionPipeline(device=self.device)
            self.load_error = None
        except Exception as exc:
            self.pipeline = None
            self.load_error = f"{type(exc).__name__}: {exc}"

    def get_status(self) -> Dict[str, Any]:
        stats: Dict[str, Any] = {
            "model_loaded": self.pipeline is not None,
            "engine": "Bi-GRU + Bi-GRU-CRF (PyTorch)" if self.pipeline else "Sin PyTorch (instalar requirements.txt)",
            "load_error": self.load_error,
            "device": self.device,
            "vocab_size": 0,
            "splits": {},
            "bio_entities": {},
            "menu_products": 0,
            "eval_metrics": None,
        }

        vocab_path = os.path.join(MODELS_DIR, "vocab.json")
        if os.path.exists(vocab_path):
            with open(vocab_path, "r", encoding="utf-8") as f:
                vdata = json.load(f)
            t2i = vdata.get("token2id") or vdata.get("word2id") or {}
            stats["vocab_size"] = len(t2i)

        if os.path.exists(ONTOLOGY_PATH):
            with open(ONTOLOGY_PATH, "r", encoding="utf-8") as f:
                ont = json.load(f)
            stats["menu_products"] = sum(
                len(items) for items in ont.get("categorias", {}).values()
            )

        bio_counter: collections.Counter = collections.Counter()
        for split_name in ("train", "val", "test"):
            spath = os.path.join(DATASET_DIR, f"{split_name}.json")
            if os.path.exists(spath):
                with open(spath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                intents = dict(collections.Counter(d.get("intencion", "desconocida") for d in data))
                gold_count = sum(1 for d in data if "id" in d)
                stats["splits"][split_name] = {
                    "total": len(data),
                    "gold": gold_count,
                    "synthetic": len(data) - gold_count,
                    "intents": intents,
                }
                if split_name == "train":
                    for item in data:
                        for tag in item.get("bio_tags", []):
                            if tag.startswith("B-"):
                                bio_counter[tag[2:]] += 1

        stats["bio_entities"] = dict(bio_counter.most_common())

        if os.path.exists(EVAL_CACHE_PATH):
            try:
                with open(EVAL_CACHE_PATH, "r", encoding="utf-8") as f:
                    stats["eval_metrics"] = json.load(f)
            except Exception:
                stats["eval_metrics"] = None

        return stats

    def predict_lstm(self, texto: str) -> Dict[str, Any]:
        lstm_dir = os.path.join(PROJECT_ROOT, "lstm")
        script = os.path.join(lstm_dir, "prediccion_lstm.py")
        outputs_dir = os.path.join(lstm_dir, "salidas")
        required = ["modelo_lstm.keras", "metadata_lstm.json", "tokenizador_entrada.json", "tokenizador_salida.json"]

        if not os.path.exists(script):
            return {"ok": False, "error": "No se encontró lstm/prediccion_lstm.py"}
        if not all(os.path.exists(os.path.join(outputs_dir, name)) for name in required):
            return {"ok": False, "error": "Entrena el modelo LSTM antes de realizar una predicción."}

        t0 = time.perf_counter()
        try:
            completed = subprocess.run(
                [sys.executable, script, "--texto", texto],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )
        except Exception as exc:
            return {"ok": False, "error": f"Error ejecutando el modelo LSTM: {exc}"}

        if completed.returncode != 0:
            error = (completed.stderr or completed.stdout or "Error desconocido").strip()
            return {"ok": False, "error": error[-3000:]}

        return {
            "ok": True,
            "texto": texto,
            "comanda": completed.stdout.strip(),
            "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
        }

    def predict(self, texto: str) -> Dict[str, Any]:
        t0 = time.perf_counter()
        if self.pipeline is None:
            return {
                "ok": False,
                "error": (
                    "El modelo PyTorch no está cargado en este entorno. "
                    f"Detalle: {self.load_error}. "
                    "Instala dependencias con: pip install -r requirements.txt"
                ),
            }

        res = self.pipeline.process(texto)
        latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        res["latency_ms"] = latency_ms
        res["ok"] = True
        return res

    def evaluate_test_set(self) -> Dict[str, Any]:
        if self.pipeline is None:
            return {
                "ok": False,
                "error": (
                    "No se puede ejecutar la evaluación porque PyTorch no está instalado. "
                    f"Detalle: {self.load_error}"
                ),
            }

        test_path = os.path.join(DATASET_DIR, "test.json")
        if not os.path.exists(test_path):
            return {"ok": False, "error": "No se encontró data/dataset/test.json."}

        from sklearn.metrics import confusion_matrix

        with open(test_path, "r", encoding="utf-8") as f:
            test_data = json.load(f)

        y_true_intent: List[str] = []
        y_pred_intent: List[str] = []
        y_true_bio: List[List[str]] = []
        y_pred_bio: List[List[str]] = []

        bio_tok_ok = 0
        bio_tok_tot = 0
        json_exact_matches = 0
        order_exact_matches = 0
        total_orders = 0
        total_chars_orig = 0
        total_chars_order = 0

        t_start = time.perf_counter()
        for item in test_data:
            text = item["texto"]
            gold_intent = item["intencion"].strip().lower()
            gold_bio = item.get("bio_tags", [])
            gold_contract = item.get("contrato_json", {})
            gold_order = item.get("comanda")

            result = self.pipeline.process(text)
            pred_intent = result["intencion"]
            pred_bio = result["bio_tags"]
            pred_contract = result["contrato_json"]
            pred_order = result["comanda_cocina"]

            y_true_intent.append(gold_intent)
            y_pred_intent.append(pred_intent)

            if gold_bio:
                aligned = pred_bio[: len(gold_bio)]
                if len(aligned) < len(gold_bio):
                    aligned.extend(["O"] * (len(gold_bio) - len(aligned)))
                y_true_bio.append(gold_bio)
                y_pred_bio.append(aligned)
                for g, p in zip(gold_bio, aligned):
                    bio_tok_tot += 1
                    if g == p:
                        bio_tok_ok += 1

            if gold_intent == "pedido":
                total_orders += 1
                total_chars_orig += len(text)
                total_chars_order += len(pred_order)

                g_items = gold_contract.get("items", [])
                p_items = pred_contract.get("items", [])
                if len(g_items) == len(p_items):
                    match = True
                    for g_it, p_it in zip(g_items, p_items):
                        if (
                            g_it.get("cantidad") != p_it.get("cantidad")
                            or g_it.get("producto_normalizado") != p_it.get("producto_normalizado")
                            or set(g_it.get("exclusiones", [])) != set(p_it.get("exclusiones", []))
                        ):
                            match = False
                            break
                    if match:
                        json_exact_matches += 1

                if gold_order and gold_order.strip().lower() == pred_order.strip().lower():
                    order_exact_matches += 1

        elapsed_ms = (time.perf_counter() - t_start) * 1000
        avg_latency_ms = round(elapsed_ms / max(1, len(test_data)), 2)

        cm = confusion_matrix(y_true_intent, y_pred_intent, labels=INTENT_CLASSES).tolist()
        cls_metrics = compute_classification_metrics(cm, INTENT_CLASSES)

        ner_f1 = None
        try:
            from seqeval.metrics import f1_score as seqeval_f1

            ner_f1 = round(float(seqeval_f1(y_true_bio, y_pred_bio)), 4)
        except Exception:
            ner_f1 = None

        ner_tok_acc = round(bio_tok_ok / max(1, bio_tok_tot), 4)
        json_em = round(json_exact_matches / max(1, total_orders), 4)
        order_em = round(order_exact_matches / max(1, total_orders), 4)
        text_reduction = round(
            (total_chars_orig - total_chars_order) / max(1, total_chars_orig), 4
        )

        metrics = {
            "evaluated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "test_samples": len(test_data),
            "total_orders": total_orders,
            **cls_metrics,
            "confusion_labels": INTENT_CLASSES,
            "confusion_matrix": cm,
            "ner_f1_strict": ner_f1,
            "ner_token_accuracy": ner_tok_acc,
            "json_exact_match": json_em,
            "order_exact_match": order_em,
            "text_reduction": text_reduction,
            "avg_latency_ms": avg_latency_ms,
        }

        os.makedirs(MODELS_DIR, exist_ok=True)
        with open(EVAL_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)

        return {"ok": True, "eval_metrics": metrics}

    def get_lstm_status(self) -> Dict[str, Any]:
        lstm_dir = os.path.join(PROJECT_ROOT, "lstm")
        outputs_dir = os.path.join(lstm_dir, "salidas")
        required = ["modelo_lstm.keras", "metadata_lstm.json", "tokenizador_entrada.json", "tokenizador_salida.json"]
        model_ready = all(os.path.exists(os.path.join(outputs_dir, name)) for name in required)
        training = self.lstm_process is not None and self.lstm_process.poll() is None

        return {
            "ok": True,
            "model_ready": model_ready,
            "training": training,
            "train_script": os.path.exists(os.path.join(lstm_dir, "entrenar_lstm.py")),
            "output": self.lstm_output[-4000:],
        }

    def train_lstm(self, epochs: int = 40, batch_size: int = 32) -> Dict[str, Any]:
        lstm_dir = os.path.join(PROJECT_ROOT, "lstm")
        script = os.path.join(lstm_dir, "entrenar_lstm.py")
        if not os.path.exists(script):
            return {"ok": False, "error": "No se encontró lstm/entrenar_lstm.py"}
        if self.lstm_process is not None and self.lstm_process.poll() is None:
            return {"ok": False, "error": "Ya hay un entrenamiento LSTM en ejecución."}

        epochs = max(1, min(int(epochs), 200))
        batch_size = max(1, min(int(batch_size), 256))
        self.lstm_output = ""
        self.lstm_process = subprocess.Popen(
            [sys.executable, script, "--epochs", str(epochs), "--batch-size", str(batch_size)],
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        import threading

        def capture_output() -> None:
            if self.lstm_process and self.lstm_process.stdout:
                for line in self.lstm_process.stdout:
                    self.lstm_output += line

        threading.Thread(target=capture_output, daemon=True).start()
        return {"ok": True, "message": "Entrenamiento LSTM iniciado."}

    def get_lstm_metrics(self) -> Dict[str, Any]:
        lstm_dir = os.path.join(PROJECT_ROOT, "lstm")
        outputs_dir = os.path.join(lstm_dir, "salidas")
        metadata_path = os.path.join(outputs_dir, "metadata_lstm.json")
        history_path = os.path.join(outputs_dir, "historial_lstm.json")

        metadata = {}
        history = {}
        if os.path.exists(metadata_path):
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        if os.path.exists(history_path):
            with open(history_path, "r", encoding="utf-8") as f:
                history = json.load(f)

        test_samples = 0
        test_path = os.path.join(DATASET_DIR, "test.json")
        if os.path.exists(test_path):
            with open(test_path, "r", encoding="utf-8") as f:
                test_samples = len(json.load(f))

        history_data = history.get("history", {})
        losses = history_data.get("loss", [])
        val_losses = history_data.get("val_loss", [])
        accuracies = history_data.get("accuracy", [])
        val_accuracies = history_data.get("val_accuracy", [])

        return {
            "ok": True,
            "available": bool(history),
            "metadata": metadata,
            "history": history,
            "test_samples": test_samples,
            "final_loss": losses[-1] if losses else None,
            "final_val_loss": val_losses[-1] if val_losses else None,
            "final_accuracy": accuracies[-1] if accuracies else None,
            "final_val_accuracy": val_accuracies[-1] if val_accuracies else None,
        }


def create_handler(service: PipelineService):
    class OrderUIHandler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            sys.stdout.write(f"[WEB] {self.address_string()} - {format % args}\n")

        def _send_json(self, payload: Dict[str, Any], status: int = 200) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path in ("/", "/index.html"):
                if not os.path.exists(UI_HTML_PATH):
                    self.send_error(404, "No se encontró src/ui/index.html")
                    return
                with open(UI_HTML_PATH, "rb") as f:
                    html = f.read()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(html)))
                self.end_headers()
                self.wfile.write(html)
                return

            if self.path == "/api/status":
                self._send_json(service.get_status())
                return

            if self.path == "/api/lstm_status":
                self._send_json(service.get_lstm_status())
                return

            if self.path == "/api/lstm_metrics":
                self._send_json(service.get_lstm_metrics())
                return

            self.send_error(404, "Ruta no encontrada")

        def do_POST(self) -> None:
            if self.path == "/api/predict":
                content_len = int(self.headers.get("Content-Length", 0))
                raw = self.rfile.read(content_len)
                try:
                    data = json.loads(raw.decode("utf-8"))
                except Exception:
                    self._send_json({"ok": False, "error": "JSON inválido"}, status=400)
                    return

                texto = str(data.get("texto", "")).strip()
                if not texto:
                    self._send_json({"ok": False, "error": "Ingresa un mensaje para analizar."}, status=400)
                    return

                result = service.predict(texto)
                status = 200 if result.get("ok") else 503
                self._send_json(result, status=status)
                return

            if self.path == "/api/evaluate":
                result = service.evaluate_test_set()
                status = 200 if result.get("ok") else 503
                self._send_json(result, status=status)
                return

            if self.path == "/api/train_lstm":
                content_len = int(self.headers.get("Content-Length", 0))
                raw = self.rfile.read(content_len)
                try:
                    data = json.loads(raw.decode("utf-8"))
                except Exception:
                    self._send_json({"ok": False, "error": "JSON inválido"}, status=400)
                    return

                epochs = data.get("epochs", 40)
                batch_size = data.get("batch_size", 32)
                result = service.train_lstm(epochs=epochs, batch_size=batch_size)
                status = 200 if result.get("ok") else 503
                self._send_json(result, status=status)
                return

            if self.path == "/api/predict_lstm":
                content_len = int(self.headers.get("Content-Length", 0))
                raw = self.rfile.read(content_len)
                try:
                    data = json.loads(raw.decode("utf-8"))
                except Exception:
                    self._send_json({"ok": False, "error": "JSON inválido"}, status=400)
                    return

                texto = str(data.get("texto", "")).strip()
                if not texto:
                    self._send_json({"ok": False, "error": "Ingresa un mensaje para analizar."}, status=400)
                    return

                result = service.predict_lstm(texto)
                status = 200 if result.get("ok") else 503
                self._send_json(result, status=status)
                return

            self.send_error(404, "Ruta no encontrada")

    return OrderUIHandler


def launch_web_ui(
    host: str = "127.0.0.1",
    port: int = 8000,
    device: str = "cpu",
    open_browser: bool = True,
) -> None:
    service = PipelineService(device=device)
    server = ThreadingHTTPServer((host, port), create_handler(service))
    url = f"http://{host}:{port}"

    print("=" * 65)
    print("  INTERFAZ WEB · SISTEMA DE SÍNTESIS DE PEDIDOS (Bi-GRU + CRF)")
    print("=" * 65)
    print(f"  URL local   : {url}")
    print(f"  Motor       : {'Bi-GRU + CRF activo' if service.pipeline else 'Pendiente (instalar torch)'}")
    print("  Presiona Ctrl+C para detener el servidor.")
    print("=" * 65)

    if open_browser:
        webbrowser.open(url)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServidor detenido.")
    finally:
        server.server_close()


if __name__ == "__main__":
    launch_web_ui()
