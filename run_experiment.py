"""
run_experiment.py
Orquestador principal del experimento TOON vs JSON.

Modos de ejecución:
  python run_experiment.py --mode tokens    → solo conteo de tokens (rápido, sin LLM)
  python run_experiment.py --mode full      → tokens + inferencia + métricas
  python run_experiment.py --mode sample N  → full sobre N instancias por categoría
"""

import json
import csv
import argparse
import time
from pathlib import Path

from generator      import main as generate_data
from converter      import load_and_convert
from token_counter  import count_file, summarize
from inference      import run_pair, check_ollama
from evaluator      import evaluate_pair, summarize_metrics

DATA_DIR    = Path("data/synthetic")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

CATEGORIES  = ["chat", "rag", "agent", "finetune", "prompt"]

CSV_FIELDS  = [
    "category", "file",
    "json_tokens", "toon_tokens", "reduction_pct",
    "json_chars",  "toon_chars",
    "latency_json_s", "latency_toon_s",
    "bleu4", "rouge_l",
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def progress(current, total, label=""):
    pct  = int(current / total * 40)
    bar  = "█" * pct + "░" * (40 - pct)
    print(f"\r  [{bar}] {current}/{total} {label}", end="", flush=True)


# ── Paso 1: generar datos si no existen ──────────────────────────────────────

def ensure_data():
    sample = DATA_DIR / "chat" / "chat_000.json"
    if not sample.exists():
        print("Generando datos sintéticos...")
        generate_data()
    else:
        print("Datos sintéticos ya existen, omitiendo generación.")


# ── Paso 2: procesar una categoría ───────────────────────────────────────────

def process_category(cat: str, max_instances: int, do_inference: bool) -> list[dict]:
    cat_dir = DATA_DIR / cat
    files   = sorted(cat_dir.glob("*.json"))[:max_instances]
    rows    = []

    for i, path in enumerate(files):
        progress(i + 1, len(files), cat)

        # Tokens
        tok = count_file(str(path))

        row = {
            "category":       cat,
            "file":           path.name,
            "json_tokens":    tok["json_tokens"],
            "toon_tokens":    tok["toon_tokens"],
            "reduction_pct":  tok["reduction_pct"],
            "json_chars":     tok["json_chars"],
            "toon_chars":     tok["toon_chars"],
            "latency_json_s": None,
            "latency_toon_s": None,
            "bleu4":          None,
            "rouge_l":        None,
        }

        # Inferencia + métricas (opcional)
        if do_inference:
            _, json_str, toon_str = load_and_convert(str(path))
            inf = run_pair(json_str, toon_str)

            if inf["ok"]:
                metrics = evaluate_pair(inf["response_json"], inf["response_toon"])
                row.update({
                    "latency_json_s": inf["latency_json_s"],
                    "latency_toon_s": inf["latency_toon_s"],
                    "bleu4":          metrics["bleu4"],
                    "rouge_l":        metrics["rouge_l"],
                })

        rows.append(row)

    print()  # nueva línea tras la barra de progreso
    return rows


# ── Paso 3: guardar CSV ───────────────────────────────────────────────────────

def save_csv(all_rows: list[dict]) -> Path:
    out = RESULTS_DIR / "results.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(all_rows)
    return out


# ── Paso 4: resumen en consola ────────────────────────────────────────────────

def print_summary(all_rows: list[dict], do_inference: bool):
    print("\n" + "═" * 62)
    print(f"  {'CATEGORÍA':<12} {'JSON tok':>8} {'TOON tok':>8} {'REDUCCIÓN':>10}", end="")
    if do_inference:
        print(f"  {'BLEU-4':>7}  {'ROUGE-L':>7}", end="")
    print()
    print("─" * 62)

    for cat in CATEGORIES:
        rows = [r for r in all_rows if r["category"] == cat]
        if not rows:
            continue
        tok_s = summarize(rows)
        line = (
            f"  {cat:<12}"
            f" {tok_s['avg_json_tokens']:>8.0f}"
            f" {tok_s['avg_toon_tokens']:>8.0f}"
            f" {tok_s['avg_reduction_pct']:>9.1f}%"
        )
        if do_inference:
            valid = [r for r in rows if r["bleu4"] is not None]
            if valid:
                ms = summarize_metrics([{"bleu4": r["bleu4"], "rouge_l": r["rouge_l"],
                                          "len_ref": 0, "len_hyp": 0} for r in valid])
                line += f"  {ms['avg_bleu4']:>7.4f}  {ms['avg_rouge_l']:>7.4f}"
        print(line)

    # Total global
    tok_g = summarize(all_rows)
    print("─" * 62)
    total_saved = tok_g["total_json_tokens"] - tok_g["total_toon_tokens"]
    print(f"  {'GLOBAL':<12}"
          f" {tok_g['avg_json_tokens']:>8.0f}"
          f" {tok_g['avg_toon_tokens']:>8.0f}"
          f" {tok_g['avg_reduction_pct']:>9.1f}%")
    print(f"\n  Tokens totales ahorrados : {total_saved:,}")
    print(f"  Instancias procesadas    : {tok_g['n']}")
    print("═" * 62)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Experimento TOON vs JSON")
    parser.add_argument("--mode", choices=["tokens", "full", "sample"],
                        default="tokens", help="Modo de ejecución")
    parser.add_argument("--n", type=int, default=10,
                        help="Instancias por categoría en modo sample")
    args = parser.parse_args()

    do_inference   = args.mode in ("full", "sample")
    max_instances  = args.n if args.mode == "sample" else 100

    print("\n╔══════════════════════════════════╗")
    print("║  Experimento TOON vs JSON v1.0   ║")
    print("╚══════════════════════════════════╝\n")
    print(f"  Modo          : {args.mode}")
    print(f"  Inferencia LLM: {'Sí — llama3.1:8b' if do_inference else 'No (solo tokens)'}")
    print(f"  Instancias    : {max_instances} por categoría\n")

    # Verificar Ollama si hace falta
    if do_inference:
        print("Verificando Ollama...")
        if not check_ollama():
            print("\n[ABORT] Ollama no disponible. Usa --mode tokens para continuar sin LLM.")
            return
        print("  Ollama OK\n")

    # Generar datos
    ensure_data()
    print()

    # Procesar
    all_rows = []
    t_start  = time.time()

    for cat in CATEGORIES:
        print(f"Procesando: {cat}")
        rows = process_category(cat, max_instances, do_inference)
        all_rows.extend(rows)

    elapsed = round(time.time() - t_start, 1)

    # Guardar resultados
    csv_path = save_csv(all_rows)

    # Resumen
    print_summary(all_rows, do_inference)
    print(f"\n  Tiempo total : {elapsed}s")
    print(f"  CSV guardado : {csv_path}\n")


if __name__ == "__main__":
    main()
