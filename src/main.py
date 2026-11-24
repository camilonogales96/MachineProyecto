"""
Punto de entrada principal del Sistema de Extracción y Síntesis de Pedidos.
Por defecto abre la interfaz web local (Gestor de Pedidos).
También permite consola interactiva (--consola), inferencia directa (--texto)
o evaluación sobre test.json (--evaluar).
"""

import argparse
import json
import os
import sys
from typing import Dict, Any

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def mostrar_resultado(resultado: Dict[str, Any]) -> None:
    """Imprime la intención, orden sintetizada, entidades BIO y contrato JSON."""
    print("\n" + "=" * 55)
    print(f"INTENCIÓN DETECTADA : {resultado['intencion'].upper()}")
    print(f"ORDEN SINTETIZADA   : {resultado['comanda_cocina']}")
    print("=" * 55)

    if resultado["intencion"] == "pedido":
        entidades = [
            f"{tok} ({tag})"
            for tok, tag in zip(resultado["tokens"], resultado["bio_tags"])
            if tag != "O"
        ]
        print("\n[Entidades BIO]:")
        print("  " + (", ".join(entidades) if entidades else "(Sin entidades detectadas)"))

    print("\n[Contrato JSON]:")
    print(json.dumps(resultado["contrato_json"], ensure_ascii=False, indent=2))
    print("-" * 55)


def modo_interactivo(pipeline) -> None:
    """Inicia la consola interactiva conectada al pipeline principal."""
    print("=" * 65)
    print("  SISTEMA DE SÍNTESIS DE PEDIDOS DE RESTAURANTE (Bi-GRU + CRF)")
    print("=" * 65)
    print('Escribe un mensaje del cliente o "salir" para terminar.\n')

    while True:
        try:
            texto = input(">> Mensaje: ").strip()
            if not texto:
                continue
            if texto.lower() in {"salir", "exit", "quit", "q"}:
                print("Saliendo...")
                break
            resultado = pipeline.process(texto)
            mostrar_resultado(resultado)
        except KeyboardInterrupt:
            print("\nSaliendo...")
            break
        except Exception as exc:
            print(f"Error procesando el mensaje: {exc}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ejecuta el Sistema de Síntesis de Pedidos (Interfaz Web o Consola)."
    )
    parser.add_argument(
        "--consola",
        action="store_true",
        help="Ejecuta en modo consola interactiva en lugar de abrir la interfaz web.",
    )
    parser.add_argument(
        "--texto",
        type=str,
        help="Procesa un único mensaje en consola y termina.",
    )
    parser.add_argument(
        "--evaluar",
        action="store_true",
        help="Ejecuta la evaluación completa sobre data/dataset/test.json.",
    )
    parser.add_argument(
        "--puerto",
        type=int,
        default=8000,
        help="Puerto HTTP para la interfaz web (por defecto: 8000).",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="No abre el navegador automáticamente al iniciar la interfaz web.",
    )
    parser.add_argument(
        "--dispositivo",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Dispositivo de inferencia para PyTorch (por defecto: cpu).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.evaluar:
        try:
            from src.evaluate import evaluate_system
        except ModuleNotFoundError as exc:
            print(
                f"Falta la dependencia '{exc.name}' para evaluar. "
                "Instala las dependencias con: pip install -r requirements.txt"
            )
            sys.exit(1)

        test_path = os.path.join(PROJECT_ROOT, "data", "dataset", "test.json")
        evaluate_system(test_path)
        return

    if args.texto or args.consola:
        try:
            from src.pipeline import OrderExtractionPipeline
        except ModuleNotFoundError as exc:
            print(
                f"Falta la dependencia '{exc.name}' para cargar el modelo. "
                "Instala las dependencias con: pip install -r requirements.txt"
            )
            sys.exit(1)

        pipeline = OrderExtractionPipeline(device=args.dispositivo)
        if args.texto:
            mostrar_resultado(pipeline.process(args.texto))
        else:
            modo_interactivo(pipeline)
        return

    from src.web_ui import launch_web_ui

    launch_web_ui(
        port=args.puerto,
        device=args.dispositivo,
        open_browser=not args.no_browser,
    )


if __name__ == "__main__":
    main()
