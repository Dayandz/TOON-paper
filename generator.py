"""
generator.py
Genera 500 payloads JSON sintéticos: 100 instancias x 5 categorías.
Salida: data/synthetic/<categoria>_<id>.json
"""

import json
import random
import os
import string
from pathlib import Path

random.seed(42)
OUTPUT_DIR = Path("data/synthetic")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Utilidades ──────────────────────────────────────────────────────────────

def rand_id(prefix="", length=6):
    chars = string.ascii_lowercase + string.digits
    return prefix + "".join(random.choices(chars, k=length))

def rand_score():
    return round(random.uniform(0.70, 0.99), 2)

def rand_bool():
    return random.choice([True, False])

TOPICS = [
    "overfitting", "backpropagation", "transformers", "LSTM", "embeddings",
    "attention mechanism", "gradient descent", "regularización", "dropout",
    "batch normalization", "transfer learning", "fine-tuning", "tokenización",
    "redes convolucionales", "reinforcement learning", "RAG", "LLM", "RLHF",
]

ROLES_SYS = [
    "Eres un asistente experto en inteligencia artificial.",
    "Eres un tutor de machine learning para estudiantes universitarios.",
    "Eres un asistente técnico especializado en NLP.",
    "Responde siempre de forma concisa y con ejemplos prácticos.",
]

QUESTIONS = [
    "¿Qué es el {}?",
    "Explica {} con un ejemplo simple.",
    "¿Cuál es la diferencia entre {} y {}?",
    "¿Cómo se implementa {} en Python?",
    "¿Por qué es importante {} en deep learning?",
]

SOURCES = [
    "deep_learning_book.pdf", "rnn_survey.pdf", "attention_paper.pdf",
    "transformer_arxiv.pdf", "ml_fundamentals.pdf", "nlp_handbook.pdf",
]

TOOLS = [
    ("read_file",    3000),
    ("web_search",   5000),
    ("run_code",     8000),
    ("send_email",   2000),
    ("query_db",     4000),
    ("summarize",    3500),
    ("translate",    2500),
]

TECHNIQUES = [
    "chain_of_thought", "few_shot", "zero_shot",
    "tree_of_thought", "self_consistency", "react",
]

LANGUAGES = ["es", "en", "pt", "fr"]
LEVELS     = ["beginner", "intermediate", "advanced"]
SPLITS     = ["train", "val", "test"]
ML_LABELS  = ["ml", "nlp", "education", "research", "production", "cv", "rl"]

# ── Categoría 1: Chat simple ─────────────────────────────────────────────────

def gen_chat(idx):
    n_user_turns = random.randint(1, 4)
    messages = [{"role": "system", "content": random.choice(ROLES_SYS)}]
    for _ in range(n_user_turns):
        topic = random.choice(TOPICS)
        q_template = random.choice(QUESTIONS)
        if "{}" in q_template:
            try:
                content = q_template.format(topic, random.choice(TOPICS))
            except IndexError:
                content = q_template.format(topic)
        else:
            content = q_template
        messages.append({"role": "user", "content": content})
        if random.random() > 0.4:
            messages.append({
                "role": "assistant",
                "content": f"El concepto de {topic} se refiere a..."
            })
    return {
        "session_id":   rand_id("sess_"),
        "model":        "llama3.1:8b",
        "temperature":  round(random.uniform(0.0, 1.0), 1),
        "max_tokens":   random.choice([256, 512, 1024]),
        "stream":       rand_bool(),
        "messages":     messages,
    }

# ── Categoría 2: RAG pipeline ────────────────────────────────────────────────

def gen_rag(idx):
    n_chunks = random.randint(2, 5)
    topic = random.choice(TOPICS)
    chunks = []
    for i in range(n_chunks):
        chunks.append({
            "id":      rand_id("doc_") + f"_p{random.randint(1,10)}",
            "score":   rand_score(),
            "source":  random.choice(SOURCES),
            "page":    random.randint(1, 120),
            "text":    f"Fragmento sobre {topic}: definición y aplicaciones en contexto práctico.",
        })
    chunks.sort(key=lambda c: c["score"], reverse=True)
    return {
        "query":            f"¿Cómo funciona {topic}?",
        "top_k":            n_chunks,
        "rerank":           rand_bool(),
        "include_metadata": rand_bool(),
        "similarity_threshold": round(random.uniform(0.60, 0.90), 2),
        "chunks":           chunks,
        "cache_hit":        rand_bool(),
        "request_id":       rand_id("req_"),
    }

# ── Categoría 3: Agente multi-tool ───────────────────────────────────────────

def gen_agent(idx):
    n_tools = random.randint(2, 5)
    selected_tools = random.sample(TOOLS, n_tools)
    tools_list = []
    for name, timeout in selected_tools:
        tools_list.append({
            "name":       name,
            "enabled":    rand_bool(),
            "timeout_ms": timeout,
            "retries":    random.randint(1, 3),
        })
    topic = random.choice(TOPICS)
    return {
        "agent_id":   rand_id("agent_"),
        "goal":       f"Analizar y resumir información sobre {topic}",
        "max_steps":  random.randint(3, 8),
        "verbose":    rand_bool(),
        "memory":     None if random.random() > 0.5 else rand_id("mem_"),
        "tools_available": tools_list,
        "allow_parallel":  rand_bool(),
        "abort_on_error":  rand_bool(),
    }

# ── Categoría 4: Fine-tuning dataset ─────────────────────────────────────────

def gen_finetune(idx):
    n_turns = random.randint(1, 3)
    topic = random.choice(TOPICS)
    conversation = []
    for _ in range(n_turns):
        conversation.append({
            "role":    "user",
            "content": f"Define {topic} con un ejemplo práctico.",
        })
        conversation.append({
            "role":    "assistant",
            "content": f"El concepto de {topic} se puede entender como...",
        })
    n_labels = random.randint(1, 3)
    return {
        "example_id":      rand_id("ft_"),
        "split":           random.choice(SPLITS),
        "quality_score":   rand_score(),
        "flagged":         False,
        "human_reviewed":  rand_bool(),
        "conversation":    conversation,
        "labels":          random.sample(ML_LABELS, n_labels),
        "source_model":    random.choice(["gpt4", "claude3", "llama3", "human"]),
        "token_count":     random.randint(80, 600),
    }

# ── Categoría 5: Prompt engineering ──────────────────────────────────────────

def gen_prompt(idx):
    topic = random.choice(TOPICS)
    return {
        "prompt_id":   rand_id("p_"),
        "technique":   random.choice(TECHNIQUES),
        "use_examples": rand_bool(),
        "n_examples":  random.randint(0, 5),
        "variables": {
            "topic":    topic,
            "level":    random.choice(LEVELS),
            "language": random.choice(LANGUAGES),
            "audience": random.choice(["students", "engineers", "researchers"]),
        },
        "constraints": {
            "max_tokens":    random.choice([256, 512, 1024]),
            "bullet_points": rand_bool(),
            "citations":     None if random.random() > 0.5 else rand_bool(),
            "language":      random.choice(LANGUAGES),
        },
        "version": round(random.uniform(1.0, 3.0), 1),
        "active":  rand_bool(),
    }

# ── Generación principal ──────────────────────────────────────────────────────

CATEGORIES = {
    "chat":     gen_chat,
    "rag":      gen_rag,
    "agent":    gen_agent,
    "finetune": gen_finetune,
    "prompt":   gen_prompt,
}

def main():
    total = 0
    for cat_name, gen_fn in CATEGORIES.items():
        cat_dir = OUTPUT_DIR / cat_name
        cat_dir.mkdir(exist_ok=True)
        for i in range(100):
            payload = gen_fn(i)
            path = cat_dir / f"{cat_name}_{i:03d}.json"
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            total += 1
        print(f"  [OK] {cat_name:10s} — 100 instancias generadas")
    print(f"\nTotal: {total} archivos en {OUTPUT_DIR}/")

if __name__ == "__main__":
    print("Generando datos sintéticos...\n")
    main()
