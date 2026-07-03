"""
evaluator.py
Calcula métricas BLEU-4 y ROUGE-L entre respuestas JSON y TOON.
"""

import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

# Descarga silenciosa si no existe
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)

try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)

_SMOOTHER = SmoothingFunction().method1
_ROUGE    = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)


def evaluate_pair(reference: str, hypothesis: str) -> dict:
    """
    Compara dos respuestas de texto.
    - reference  : respuesta generada con JSON (baseline)
    - hypothesis : respuesta generada con TOON (candidato)

    Retorna:
      bleu4      — puntuación BLEU-4 (0.0 - 1.0)
      rouge_l    — F1 de ROUGE-L    (0.0 - 1.0)
      len_ref    — palabras en referencia
      len_hyp    — palabras en hipótesis
    """
    if not reference or not hypothesis:
        return {"bleu4": 0.0, "rouge_l": 0.0, "len_ref": 0, "len_hyp": 0}

    ref_tokens  = reference.lower().split()
    hyp_tokens  = hypothesis.lower().split()

    bleu = sentence_bleu(
        [ref_tokens],
        hyp_tokens,
        weights=(0.25, 0.25, 0.25, 0.25),   # BLEU-4
        smoothing_function=_SMOOTHER,
    )

    rouge = _ROUGE.score(reference, hypothesis)

    return {
        "bleu4":   round(bleu, 4),
        "rouge_l": round(rouge["rougeL"].fmeasure, 4),
        "len_ref": len(ref_tokens),
        "len_hyp": len(hyp_tokens),
    }


def summarize_metrics(metrics_list: list[dict]) -> dict:
    """Estadísticas agregadas de una lista de resultados de evaluate_pair."""
    n = len(metrics_list)
    if n == 0:
        return {}
    avg = lambda k: round(sum(m[k] for m in metrics_list) / n, 4)
    return {
        "n":            n,
        "avg_bleu4":    avg("bleu4"),
        "avg_rouge_l":  avg("rouge_l"),
        "avg_len_ref":  avg("len_ref"),
        "avg_len_hyp":  avg("len_hyp"),
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ref  = "El payload contiene una sesión de chat con mensajes de sistema y usuario, y su propósito es enviar una consulta al modelo de lenguaje."
    hyp1 = "El payload describe una sesión de conversación con mensajes de sistema y usuario para enviar al modelo."
    hyp2 = "El documento tiene imágenes de gatos y perros en un zoológico de Buenos Aires."

    print("=== Prueba de métricas ===\n")
    m1 = evaluate_pair(ref, hyp1)
    m2 = evaluate_pair(ref, hyp2)

    print(f"Respuesta similar  → BLEU-4: {m1['bleu4']:.4f}  |  ROUGE-L: {m1['rouge_l']:.4f}")
    print(f"Respuesta distinta → BLEU-4: {m2['bleu4']:.4f}  |  ROUGE-L: {m2['rouge_l']:.4f}")
    print("\n(BLEU/ROUGE cercanos a 1.0 = respuestas equivalentes)")
