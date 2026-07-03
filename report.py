"""
report.py
Lee results/results.csv y genera visualizaciones del experimento.
Requiere: pip install matplotlib pandas
"""

import csv
import json
from pathlib import Path
from collections import defaultdict

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("[WARN] matplotlib no instalado. Solo se mostrará el reporte en texto.")
    print("       Instala con: pip install matplotlib pandas")

RESULTS_CSV = Path("results/results.csv")
CHARTS_DIR  = Path("results/charts")
CATEGORIES  = ["chat", "rag", "agent", "finetune", "prompt"]
CAT_LABELS  = {
    "chat":     "Chat simple",
    "rag":      "RAG pipeline",
    "agent":    "Agente multi-tool",
    "finetune": "Fine-tuning",
    "prompt":   "Prompt eng.",
}
COLORS = {
    "json": "#D85A30",
    "toon": "#1D9E75",
    "save": "#534AB7",
}


# ── Carga de datos ────────────────────────────────────────────────────────────

def load_results() -> list[dict]:
    if not RESULTS_CSV.exists():
        print(f"[ERROR] No se encontró {RESULTS_CSV}")
        print("        Ejecuta primero: python run_experiment.py --mode tokens")
        return []
    rows = []
    with open(RESULTS_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append({
                "category":      row["category"],
                "json_tokens":   int(row["json_tokens"]),
                "toon_tokens":   int(row["toon_tokens"]),
                "reduction_pct": float(row["reduction_pct"]),
                "json_chars":    int(row["json_chars"]),
                "toon_chars":    int(row["toon_chars"]),
                "latency_json":  float(row["latency_json_s"]) if row["latency_json_s"] else None,
                "latency_toon":  float(row["latency_toon_s"]) if row["latency_toon_s"] else None,
                "bleu4":         float(row["bleu4"])   if row["bleu4"]   else None,
                "rouge_l":       float(row["rouge_l"]) if row["rouge_l"] else None,
            })
    return rows


def group_by_category(rows: list[dict]) -> dict:
    groups = defaultdict(list)
    for r in rows:
        groups[r["category"]].append(r)
    return dict(groups)


def avg(values):
    v = [x for x in values if x is not None]
    return sum(v) / len(v) if v else 0.0


# ── Reporte de texto ──────────────────────────────────────────────────────────

def print_report(rows: list[dict]):
    groups = group_by_category(rows)
    has_inference = any(r["bleu4"] is not None for r in rows)

    print("\n╔══════════════════════════════════════════════════════════╗")
    print("║          REPORTE FINAL — TOON vs JSON                   ║")
    print("╚══════════════════════════════════════════════════════════╝\n")

    header = f"  {'Categoría':<16} {'JSON':>7} {'TOON':>7} {'Reduc.':>8} {'Chars↓':>7}"
    if has_inference:
        header += f"  {'BLEU-4':>7}  {'ROUGE-L':>7}  {'Lat↓%':>7}"
    print(header)
    print("─" * (len(header) + 2))

    for cat in CATEGORIES:
        if cat not in groups:
            continue
        g = groups[cat]
        avg_j   = avg([r["json_tokens"]   for r in g])
        avg_t   = avg([r["toon_tokens"]   for r in g])
        avg_r   = avg([r["reduction_pct"] for r in g])
        avg_cj  = avg([r["json_chars"]    for r in g])
        avg_ct  = avg([r["toon_chars"]    for r in g])
        char_r  = round((1 - avg_ct / avg_cj) * 100, 1) if avg_cj > 0 else 0

        line = f"  {CAT_LABELS[cat]:<16} {avg_j:>7.0f} {avg_t:>7.0f} {avg_r:>7.1f}% {char_r:>6.1f}%"

        if has_inference:
            avg_b = avg([r["bleu4"]   for r in g if r["bleu4"]   is not None])
            avg_rl= avg([r["rouge_l"] for r in g if r["rouge_l"] is not None])
            avg_lj= avg([r["latency_json"] for r in g if r["latency_json"] is not None])
            avg_lt= avg([r["latency_toon"] for r in g if r["latency_toon"] is not None])
            lat_r = round((1 - avg_lt / avg_lj) * 100, 1) if avg_lj > 0 else 0
            line += f"  {avg_b:>7.4f}  {avg_rl:>7.4f}  {lat_r:>6.1f}%"

        print(line)

    # Global
    print("─" * (len(header) + 2))
    total_j = sum(r["json_tokens"] for r in rows)
    total_t = sum(r["toon_tokens"] for r in rows)
    saved   = total_j - total_t
    avg_red = avg([r["reduction_pct"] for r in rows])

    print(f"\n  Total JSON tokens  : {total_j:,}")
    print(f"  Total TOON tokens  : {total_t:,}")
    print(f"  Tokens ahorrados   : {saved:,}  ({avg_red:.1f}% promedio)")
    print(f"  Instancias totales : {len(rows)}")

    if has_inference:
        all_bleu = [r["bleu4"]   for r in rows if r["bleu4"]   is not None]
        all_rl   = [r["rouge_l"] for r in rows if r["rouge_l"] is not None]
        if all_bleu:
            print(f"\n  BLEU-4 promedio    : {avg(all_bleu):.4f}")
            print(f"  ROUGE-L promedio   : {avg(all_rl):.4f}")
            print(f"  {'✓' if avg(all_bleu) >= 0.80 else '✗'} Hipótesis H2 ({'confirmada' if avg(all_bleu) >= 0.80 else 'no confirmada'}) — umbral BLEU ≥ 0.80")

    h1_ok = avg_red >= 40.0
    print(f"\n  {'✓' if h1_ok else '✗'} Hipótesis H1 ({'confirmada' if h1_ok else 'no confirmada'}) — reducción ≥ 40%")
    print()


# ── Gráficas ──────────────────────────────────────────────────────────────────

def make_charts(rows: list[dict]):
    if not HAS_MPL:
        return
    CHARTS_DIR.mkdir(exist_ok=True)
    groups = group_by_category(rows)
    cats   = [c for c in CATEGORIES if c in groups]
    labels = [CAT_LABELS[c] for c in cats]

    avg_j = [avg([r["json_tokens"]   for r in groups[c]]) for c in cats]
    avg_t = [avg([r["toon_tokens"]   for r in groups[c]]) for c in cats]
    avg_r = [avg([r["reduction_pct"] for r in groups[c]]) for c in cats]

    # ── Gráfica 1: Tokens JSON vs TOON por categoría ──────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    x = range(len(cats))
    w = 0.35
    bars_j = ax.bar([i - w/2 for i in x], avg_j, w,
                    color=COLORS["json"], alpha=0.85, label="JSON")
    bars_t = ax.bar([i + w/2 for i in x], avg_t, w,
                    color=COLORS["toon"], alpha=0.85, label="TOON")

    for bar, val in zip(bars_j, avg_j):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 4,
                f"{val:.0f}", ha="center", va="bottom", fontsize=9)
    for bar, val in zip(bars_t, avg_t):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 4,
                f"{val:.0f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Tokens promedio", fontsize=11)
    ax.set_title("Comparación de tokens: JSON vs TOON", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    out1 = CHARTS_DIR / "tokens_comparison.png"
    plt.savefig(out1, dpi=150)
    plt.close()
    print(f"  Gráfica guardada: {out1}")

    # ── Gráfica 2: % reducción por categoría ─────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = [COLORS["save"]] * len(cats)
    bars   = ax.barh(labels, avg_r, color=colors, alpha=0.85)
    for bar, val in zip(bars, avg_r):
        ax.text(val + 0.3, bar.get_y() + bar.get_height()/2,
                f"{val:.1f}%", va="center", fontsize=10)
    ax.axvline(40, color="gray", linestyle="--", linewidth=1, label="Umbral H1 (40%)")
    ax.set_xlabel("Reducción de tokens (%)", fontsize=11)
    ax.set_title("Reducción de tokens por categoría", fontsize=13, fontweight="bold")
    ax.set_xlim(0, max(avg_r) * 1.18)
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    out2 = CHARTS_DIR / "reduction_by_category.png"
    plt.savefig(out2, dpi=150)
    plt.close()
    print(f"  Gráfica guardada: {out2}")

    # ── Gráfica 3: BLEU-4 y ROUGE-L (solo si hay datos de inferencia) ─────────
    has_inf = any(r["bleu4"] is not None for r in rows)
    if has_inf:
        avg_b  = [avg([r["bleu4"]   for r in groups[c] if r["bleu4"]   is not None]) for c in cats]
        avg_rl = [avg([r["rouge_l"] for r in groups[c] if r["rouge_l"] is not None]) for c in cats]

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(labels, avg_b,  "o-", color=COLORS["json"], linewidth=2, label="BLEU-4")
        ax.plot(labels, avg_rl, "s-", color=COLORS["toon"], linewidth=2, label="ROUGE-L")
        ax.axhline(0.80, color="gray", linestyle="--", linewidth=1, label="Umbral mínimo (0.80)")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Puntuación", fontsize=11)
        ax.set_title("Calidad de respuesta: BLEU-4 y ROUGE-L", fontsize=13, fontweight="bold")
        ax.legend(fontsize=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        out3 = CHARTS_DIR / "quality_metrics.png"
        plt.savefig(out3, dpi=150)
        plt.close()
        print(f"  Gráfica guardada: {out3}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Cargando resultados...")
    rows = load_results()
    if not rows:
        exit(1)

    print_report(rows)

    if HAS_MPL:
        print("Generando gráficas...")
        make_charts(rows)
        print(f"\nGráficas en: {CHARTS_DIR}/")
    else:
        print("Instala matplotlib para generar gráficas:")
        print("  pip install matplotlib pandas")
