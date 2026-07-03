"""
converter.py  —  TOON v1.2  (serialización compacta)
=====================================================
Convierte un dict Python/JSON al formato TOON optimizado para tokens.

Reglas de conversión
──────────────────────────────────────────────────────────
  Tipo          JSON                TOON
  ──────────    ──────────────────  ──────────────────────
  objeto {}     {"k":"v","n":1}     <k:"v" n:1>
  true          true                +
  false         false               -
  null / None   null                ~
  string        "hola"              "hola"        (sin cambio)
  número        42 / 3.14           42 / 3.14     (sin cambio)
  lista         [1,2,3]             [1 2 3]       (espacio como separador)
  clave         "nombre"            nombre        (sin comillas)

Estrategia de compactación
──────────────────────────────────────────────────────────
  • Objetos simples (sin listas ni subobjetos) → una sola línea: <k1:v1 k2:v2>
  • Listas cortas de escalares             → una sola línea: [v1 v2 v3]
  • Listas de objetos o listas largas      → una entrada por línea con sangría
  • Nivel raíz                             → pares clave:valor separados por \n

El resultado ocupa sistemáticamente menos tokens que JSON porque:
  1. Elimina comillas de todas las claves
  2. Comprime booleanos y null a 1 carácter
  3. Usa < > en lugar de { } (1 carácter cada uno, igual que JSON)
  4. Elimina comas separadoras entre campos de objeto
  5. Evita saltos de línea innecesarios en estructuras simples
"""

import json
import re
from pathlib import Path

# ── Configuración ─────────────────────────────────────────────────────────────

# Número máximo de ítems escalares en una lista para mantenerla en 1 línea
_MAX_INLINE_LIST = 8
# Número máximo de campos en un objeto para mantenerlo en 1 línea
_MAX_INLINE_FIELDS = 6


# ── Helpers ───────────────────────────────────────────────────────────────────

def _is_scalar(v) -> bool:
    """True si el valor es primitivo (no contiene subestructuras)."""
    return v is None or isinstance(v, (bool, int, float, str))


def _all_scalar(lst: list) -> bool:
    return all(_is_scalar(v) for v in lst)


def _is_simple_dict(d: dict) -> bool:
    """True si todos los valores del dict son escalares y cabe en 1 línea."""
    return (
        len(d) <= _MAX_INLINE_FIELDS
        and all(_is_scalar(v) for v in d.values())
    )


# ── Núcleo de conversión ──────────────────────────────────────────────────────

def _scalar_to_toon(v) -> str:
    """Convierte un valor escalar a su representación TOON."""
    if v is True:
        return "+"
    if v is False:
        return "-"
    if v is None:
        return "~"
    if isinstance(v, str):
        escaped = v.replace('"', '\\"')
        return f'"{escaped}"'
    # int o float
    return str(v)


def _list_to_toon(lst: list, indent: int) -> str:
    """Convierte una lista a TOON."""
    if not lst:
        return "[]"

    # Lista corta de escalares → una línea
    if _all_scalar(lst) and len(lst) <= _MAX_INLINE_LIST:
        inner = " ".join(_scalar_to_toon(v) for v in lst)
        return f"[{inner}]"

    # Lista con objetos o larga → una entrada por línea
    pad  = "  " * indent
    pad1 = "  " * (indent + 1)
    items = []
    for item in lst:
        items.append(f"{pad1}{_value_to_toon(item, indent + 1)}")
    return "[\n" + "\n".join(items) + f"\n{pad}]"


def _dict_to_toon(d: dict, indent: int) -> str:
    """Convierte un dict a TOON."""
    if not d:
        return "<>"

    # Objeto simple → todo en una línea: <k1:v1 k2:v2>
    if _is_simple_dict(d):
        pairs = " ".join(f"{k}:{_scalar_to_toon(v)}" for k, v in d.items())
        return f"<{pairs}>"

    # Objeto complejo → un campo por línea
    pad  = "  " * indent
    pad1 = "  " * (indent + 1)
    lines = []
    for k, v in d.items():
        lines.append(f"{pad1}{k}:{_value_to_toon(v, indent + 1)}")
    return "<\n" + "\n".join(lines) + f"\n{pad}>"


def _value_to_toon(v, indent: int = 0) -> str:
    """Despacha al convertidor correcto según el tipo."""
    if _is_scalar(v):
        return _scalar_to_toon(v)
    if isinstance(v, list):
        return _list_to_toon(v, indent)
    if isinstance(v, dict):
        return _dict_to_toon(v, indent)
    # Fallback
    return str(v)


def json_to_toon(data: dict) -> str:
    """
    Punto de entrada principal.
    Convierte un dict Python a string TOON.
    El nivel raíz se serializa como pares clave:valor separados por \\n.
    """
    lines = []
    for k, v in data.items():
        lines.append(f"{k}:{_value_to_toon(v, indent=0)}")
    return "\n".join(lines)


def json_str_to_toon(json_string: str) -> str:
    """Convierte un string JSON a string TOON."""
    return json_to_toon(json.loads(json_string))


def load_and_convert(json_path: str) -> tuple:
    """
    Lee un archivo .json y retorna:
      (dict_original, json_minificado_string, toon_string)

    Nota: json_minificado_string usa separators=(',',':') para medir
    tokens de forma justa (sin espacios decorativos).
    """
    path = Path(json_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    json_min = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    toon = json_to_toon(data)
    return data, json_min, toon


# ── CLI / demo ────────────────────────────────────────────────────────────────

def _demo():
    import sys
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        def tok(s): return len(enc.encode(s))
        has_tiktoken = True
    except ImportError:
        has_tiktoken = False

    samples = [
        # 1. Objeto simple con booleanos y null
        {
            "session_id": "a3f9c1",
            "model": "llama3.1:8b",
            "temperature": 0.7,
            "stream": True,
            "memory": None,
            "active": False,
        },
        # 2. Objeto con mensajes anidados (chat)
        {
            "session_id": "sess_xyz",
            "model": "llama3.1:8b",
            "temperature": 0.7,
            "stream": True,
            "messages": [
                {"role": "system",    "content": "Eres un asistente experto en IA."},
                {"role": "user",      "content": "¿Qué es el overfitting?"},
            ],
        },
        # 3. RAG con chunks
        {
            "query": "¿Como funciona LSTM?",
            "top_k": 3,
            "rerank": True,
            "include_metadata": False,
            "chunks": [
                {"id": "doc_42_p3", "score": 0.91, "source": "deep_learning.pdf",
                 "text": "LSTM usa puertas de entrada, olvido y salida."},
                {"id": "doc_07_p1", "score": 0.87, "source": "rnn_survey.pdf",
                 "text": "La celda de memoria preserva gradientes a largo plazo."},
            ],
        },
    ]

    names = ["Objeto simple", "Chat con mensajes", "RAG con chunks"]
    print("=" * 58)
    print("  DEMO — converter.py  (modo compacto)")
    print("=" * 58)

    for name, data in zip(names, samples):
        json_min = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
        toon     = json_to_toon(data)

        print(f"\n── {name} ──")
        print(f"\n  JSON minificado:\n  {json_min}")
        print(f"\n  TOON:\n")
        for line in toon.splitlines():
            print(f"  {line}")

        if has_tiktoken:
            tj = tok(json_min)
            tt = tok(toon)
            delta = tt - tj
            sign  = "+" if delta >= 0 else ""
            pct   = (tt / tj - 1) * 100
            print(f"\n  Tokens → JSON: {tj}  |  TOON: {tt}  |  Δ {sign}{delta} ({pct:+.1f}%)")
        print()

    if not has_tiktoken:
        print("  (instala tiktoken para ver conteo de tokens)")


if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 2:
        data, json_min, toon = load_and_convert(sys.argv[1])
        print("=== JSON (minificado) ===")
        print(json_min)
        print("\n=== TOON ===")
        print(toon)
    else:
        _demo()