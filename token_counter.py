"""
token_counter.py
Cuenta tokens usando tiktoken (cl100k_base).
No requiere API — corre 100% local.
"""

import tiktoken
import json
from pathlib import Path

# cl100k_base es el tokenizador de GPT-4, Claude 3+ y modelos Llama modernos
_ENCODING = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """Retorna el número de tokens de un string."""
    return len(_ENCODING.encode(text))


def count_file(json_path: str) -> dict:
    """
    Lee un .json, lo convierte a TOON y retorna un dict con:
      - json_tokens
      - toon_tokens
      - reduction_pct
      - json_chars
      - toon_chars
    """
    from converter import load_and_convert

    _, json_str, toon_str = load_and_convert(json_path)

    # JSON minificado (sin indentación) — simula payload real de API
    data = json.loads(json_str)
    json_min = json.dumps(data, ensure_ascii=False, separators=(",", ":"))

    j_tokens = count_tokens(json_min)
    t_tokens  = count_tokens(toon_str)
    reduction = round((1 - t_tokens / j_tokens) * 100, 2) if j_tokens > 0 else 0.0

    return {
        "json_tokens":    j_tokens,
        "toon_tokens":    t_tokens,
        "reduction_pct":  reduction,
        "json_chars":     len(json_min),
        "toon_chars":     len(toon_str),
    }


def count_batch(category_dir: str) -> list[dict]:
    """
    Procesa todos los .json de un directorio y retorna lista de resultados.
    """
    results = []
    paths = sorted(Path(category_dir).glob("*.json"))
    if not paths:
        print(f"  [WARN] No se encontraron archivos en {category_dir}")
        return results
    for path in paths:
        metrics = count_file(str(path))
        metrics["file"] = path.name
        results.append(metrics)
    return results


def summarize(results: list[dict]) -> dict:
    """Calcula estadísticas agregadas de una lista de resultados."""
    if not results:
        return {}
    n = len(results)
    avg = lambda key: round(sum(r[key] for r in results) / n, 2)
    return {
        "n":                   n,
        "avg_json_tokens":     avg("json_tokens"),
        "avg_toon_tokens":     avg("toon_tokens"),
        "avg_reduction_pct":   avg("reduction_pct"),
        "total_json_tokens":   sum(r["json_tokens"] for r in results),
        "total_toon_tokens":   sum(r["toon_tokens"] for r in results),
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) >= 2:
        # Contar un archivo específico
        m = count_file(sys.argv[1])
        print(f"JSON tokens : {m['json_tokens']}")
        print(f"TOON tokens : {m['toon_tokens']}")
        print(f"Reducción   : {m['reduction_pct']}%")
        print(f"JSON chars  : {m['json_chars']}")
        print(f"TOON chars  : {m['toon_chars']}")
    else:
        # Demo rápido con texto inline
        sample = '{"active":true,"count":42,"name":"test","data":null}'
        toon   = 'active:+\ncount:42\nname:"test"\ndata:~'
        print(f"Demo JSON   : {sample}")
        print(f"Demo TOON   : {toon}")
        print(f"JSON tokens : {count_tokens(sample)}")
        print(f"TOON tokens : {count_tokens(toon)}")
