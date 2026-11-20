"""
DEMO INTERACTIVA DE PRUEBA EN VIVO
Permite ingresar pedidos en lenguaje natural libre por teclado y ver en tiempo real:
- Clasificación de Intención (Bi-GRU)
- Comanda sintetizada para cocina
- Contrato JSON estructurado
- Etiquetas BIO de NER detectadas
"""

import os
import sys
import json

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from src.pipeline import OrderExtractionPipeline


def main():
    print("=" * 70)
    print("  SISTEMA DE SÍNTESIS DE PEDIDOS DE RESTAURANTE (DEEP LEARNING)")
    print("  Modelos Bi-GRU + CRF entrenados 100% From Scratch")
    print("=" * 70)
    print("Cargando pipeline y modelos entrenados...")
    
    pipeline = OrderExtractionPipeline()
    print("¡Sistema listo!\n")
    print("Puedes escribir cualquier mensaje de prueba en español.")
    print("Ejemplos sugeridos:")
    print('  - "Hola buenas tardes, quisiera pedir 2 hamburguesas dobles sin cebolla y papas"')
    print('  - "Buenas, me pones una burger simple y cámbiame el tomate por tocino porfa"')
    print('  - "Hola, ¿a qué hora cierran hoy?"')
    print('  - "Llevo esperando una hora y la comida llegó fría"')
    print('Escribe "salir" para terminar.\n' + "-" * 70)

    while True:
        try:
            texto = input("\n>> Ingresa un mensaje de cliente: ").strip()
            if not texto:
                continue
            if texto.lower() in ("salir", "exit", "quit", "q"):
                print("¡Hasta luego!")
                break

            resultado = pipeline.process(texto)

            print("\n" + "=" * 50)
            print(f"INTENCIÓN DETECTADA : {resultado['intencion'].upper()}")
            print(f"COMANDA PARA COCINA : {resultado['comanda_cocina']}")
            print("=" * 50)
            
            # Mostrar tokens con sus etiquetas BIO si es pedido
            if resultado["intencion"] == "pedido":
                print("\n[Entidades BIO Reconocidas]:")
                entidades_detectadas = [
                    f"{tok} ({tag})" for tok, tag in zip(resultado["tokens"], resultado["bio_tags"])
                    if tag != "O"
                ]
                if entidades_detectadas:
                    print("  " + ", ".join(entidades_detectadas))
                else:
                    print("  (Sin entidades específicas)")

            print("\n[Contrato JSON (Fuente de Verdad)]:")
            print(json.dumps(resultado["contrato_json"], ensure_ascii=False, indent=2))
            print("-" * 50)

        except KeyboardInterrupt:
            print("\nSaliendo...")
            break
        except Exception as e:
            print(f"Error procesando la entrada: {e}")


if __name__ == "__main__":
    main()
