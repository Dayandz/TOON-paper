"""
inference.py
Envía payloads JSON y TOON a Ollama y captura las respuestas.
Usa la API REST local de Ollama (no requiere API key).
"""

import requests
import json
import time
from typing import Optional

OLLAMA_URL  = "http://localhost:11434/api/generate"
MODEL       = "llama3.1:8b"
TEMPERATURE = 0.0          # Determinista para reproducibilidad
TIMEOUT_SEC = 120


# ── Prompt base del experimento ───────────────────────────────────────────────

SYSTEM_PROMPT = """Eres un sistema de análisis de datos estructurados.
Recibirás un payload de datos en un formato estructurado.
Tu tarea es responder la pregunta incluida en el payload de forma concisa y precisa.
Responde ÚNICAMENTE con la respuesta a la pregunta, sin comentarios adicionales."""

QUESTION_SUFFIX = "\n\nPregunta: ¿Qué información contiene este payload y cuál es su propósito principal? Responde en 2-3 oraciones."


def call_ollama(prompt: str, model: str = MODEL) -> Optional[str]:
    """
    Llama a Ollama con un prompt y retorna el texto de respuesta.
    Retorna None si hay error.
    """
    payload = {
        "model":  model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": TEMPERATURE,
            "num_predict": 200,    # Máximo de tokens a generar
            "top_p":       1.0,
        },
    }
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=TIMEOUT_SEC)
        resp.raise_for_status()
        data = resp.json()
        return data.get("response", "").strip()
    except requests.exceptions.ConnectionError:
        print("  [ERROR] Ollama no está corriendo. Ejecuta: ollama serve")
        return None
    except requests.exceptions.Timeout:
        print(f"  [ERROR] Timeout después de {TIMEOUT_SEC}s")
        return None
    except Exception as e:
        print(f"  [ERROR] {e}")
        return None


def run_pair(json_str: str, toon_str: str) -> dict:
    """
    Envía el mismo contenido en JSON y en TOON a Ollama.
    Retorna un dict con ambas respuestas y tiempos de latencia.
    """
    prompt_json = f"{SYSTEM_PROMPT}\n\nDATOS (formato JSON):\n{json_str}{QUESTION_SUFFIX}"
    prompt_toon = f"{SYSTEM_PROMPT}\n\nDATOS (formato TOON):\n{toon_str}{QUESTION_SUFFIX}"

    # Llamada JSON
    t0 = time.time()
    resp_json = call_ollama(prompt_json)
    latency_json = round(time.time() - t0, 3)

    # Pausa breve para no saturar Ollama
    time.sleep(0.5)

    # Llamada TOON
    t0 = time.time()
    resp_toon = call_ollama(prompt_toon)
    latency_toon = round(time.time() - t0, 3)

    return {
        "response_json":  resp_json  or "",
        "response_toon":  resp_toon  or "",
        "latency_json_s": latency_json,
        "latency_toon_s": latency_toon,
        "ok":             (resp_json is not None) and (resp_toon is not None),
    }


def check_ollama() -> bool:
    """Verifica que Ollama esté disponible antes de iniciar el experimento."""
    try:
        resp = requests.get("http://localhost:11434/api/tags", timeout=5)
        models = [m["name"] for m in resp.json().get("models", [])]
        if not any(MODEL in m for m in models):
            print(f"  [WARN] Modelo '{MODEL}' no encontrado. Modelos disponibles: {models}")
            return False
        return True
    except Exception:
        print("  [ERROR] No se puede conectar a Ollama en localhost:11434")
        print("          Asegúrate de que Ollama esté corriendo.")
        return False


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Verificando conexión con Ollama...")
    if not check_ollama():
        exit(1)

    # Prueba rápida con un payload mínimo
    sample_json = json.dumps({
        "session_id": "test_01",
        "model": "llama3.1:8b",
        "temperature": 0.7,
        "stream": True,
        "messages": [
            {"role": "system",  "content": "Eres un asistente experto en IA."},
            {"role": "user",    "content": "¿Qué es el overfitting?"},
        ]
    }, ensure_ascii=False)

    sample_toon = (
        'session_id:"test_01"\n'
        'model:"llama3.1:8b"\n'
        'temperature:0.7\n'
        'stream:+\n'
        'messages:[\n'
        '  <role:"system" content:"Eres un asistente experto en IA.">\n'
        '  <role:"user" content:"¿Qué es el overfitting?">\n'
        ']'
    )

    print("Enviando prueba a Ollama...\n")
    result = run_pair(sample_json, sample_toon)

    print(f"Respuesta JSON ({result['latency_json_s']}s):\n  {result['response_json'][:120]}...")
    print(f"\nRespuesta TOON ({result['latency_toon_s']}s):\n  {result['response_toon'][:120]}...")
    print(f"\nEstado: {'OK' if result['ok'] else 'ERROR'}")
