"""Shared helpers: Ollama wrapper, claim loading, prompts, parsing.
Method follows AutoHall (Cao et al.): build a model-specific hallucination dataset from
fact-checking claims, then detect hallucination via self-contradiction among sampled references.
All code and prompt wording here is written from scratch."""
import json
import random
import re
from pathlib import Path

import ollama

# ---------------------------------------------------------------- prompts
# 7 paraphrased "write a reference" prompts: [0] for the original reference, [1..6] for the samples
GEN_PROMPTS = [
    "Here is a claim whose truth is unknown. Write one reference (a source title and a one-paragraph summary) that relates to it.\nClaim: {claim}",
    "Give one reference for the following claim and summarise it in a single paragraph. We do not know if the claim is true.\nClaim: {claim}",
    "Please cite one source about this claim and write a short paragraph describing what it says.\nClaim: {claim}",
    "Find one reference connected to the claim below and summarise its content in one paragraph.\nClaim: {claim}",
    "The accuracy of this claim is uncertain. Provide one supporting or relevant reference with a one-paragraph summary.\nClaim: {claim}",
    "Name one reference about this claim and condense it into a paragraph.\nClaim: {claim}",
    "Write a single reference related to the claim, followed by a one-paragraph summary of it.\nClaim: {claim}",
]

CLASSIFY_PROMPT = (
    "Below are a claim and a reference about it. Using the reference, decide whether the claim is true or false.\n"
    "Start your answer with exactly 'True' or 'False', then give one short reason.\n"
    "Claim: {claim}\nReference: {reference}"
)

CONFLICT_PROMPT = (
    "Below are two paragraphs about the same claim. Do they contain conflicting information, "
    "meaning facts that contradict each other?\n"
    "Start your answer with exactly 'Yes' or 'No', then give one short reason.\n"
    "Paragraph 1: {p1}\nParagraph 2: {p2}"
)

REFUSAL_RE = re.compile(
    r"(i cannot|i can't|i can not|i'm sorry|i am sorry|as an ai|unable to provide|i don't have access)", re.I)


# ---------------------------------------------------------------- model
def chat(prompt, model, temperature=0.0, max_tokens=250):
    resp = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": temperature, "num_predict": max_tokens},
    )
    return resp["message"]["content"].strip()


def parse_first_word(text, a, b):
    """Return 1 if the answer starts with `a`, 0 if with `b`, else None."""
    m = re.match(r"^\W*(%s|%s)\b" % (a, b), text, re.I)
    if not m:
        return None
    return 1 if m.group(1).lower() == a else 0


def is_refusal(text):
    return bool(REFUSAL_RE.search(text[:200])) or len(text) < 30


# ---------------------------------------------------------------- data
def load_claims(n, csv=None, seed=42):
    """Return a list of {id, claim, label_true} where label_true=1 means the claim is factual."""
    rows = []
    if csv:
        import pandas as pd
        df = pd.read_csv(csv)
        lab = df["label"].astype(str).str.lower().map({"true": 1, "false": 0, "1": 1, "0": 0})
        for i, (c, y) in enumerate(zip(df["claim"], lab)):
            if y in (0, 1):
                rows.append({"id": str(i), "claim": c, "label_true": int(y)})
    else:
        from datasets import load_dataset
        ds = None
        for name in ["tdiggelm/climate_fever", "climate_fever"]:
            try:
                ds = load_dataset(name, split="test")
                break
            except Exception as e:  # noqa
                print("could not load", name, "->", str(e)[:120])
        if ds is None:
            raise SystemExit("Could not load Climate-FEVER. Try --csv data/sample_claims.csv, "
                             "or send the error text to Claude.")
        names = ds.features["claim_label"].names
        for i, r in enumerate(ds):
            lab = names[r["claim_label"]]
            if lab == "SUPPORTS":
                y = 1
            elif lab == "REFUTES":
                y = 0
            else:
                continue
            rows.append({"id": str(r.get("claim_id", i)), "claim": r["claim"], "label_true": y})
    random.Random(seed).shuffle(rows)
    return rows[:n]


# ---------------------------------------------------------------- jsonl
def read_jsonl(path):
    p = Path(path)
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def append_jsonl(path, rec):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
