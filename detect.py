"""Stage 2: self-contradiction detection.
Balance the dataset, then for each reference sample K new references (different prompts, higher temperature)
and ask the model whether each sample conflicts with the original. Saves the full conflict vector."""
import argparse
import random
import time

from common import (GEN_PROMPTS, CONFLICT_PROMPT, chat, parse_first_word, read_jsonl, append_jsonl)

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="mistral")
ap.add_argument("--inp", default="data/built.jsonl")
ap.add_argument("--out", default="data/detected.jsonl")
ap.add_argument("--k", type=int, default=6)
ap.add_argument("--temp", type=float, default=0.7)
ap.add_argument("--max-items", type=int, default=0, help="optional cap on balanced set size (for quick tests)")
ap.add_argument("--seed", type=int, default=42)
args = ap.parse_args()
assert args.k <= len(GEN_PROMPTS) - 1, "k must be <= 6"

rows = [r for r in read_jsonl(args.inp) if "hallucinated" in r]
hall = [r for r in rows if r["hallucinated"] == 1]
fact = [r for r in rows if r["hallucinated"] == 0]
m = min(len(hall), len(fact))
if args.max_items:
    m = min(m, args.max_items // 2)
if m < 3:
    raise SystemExit(f"Only {len(hall)} hallucinated / {len(fact)} factual references. Run build_dataset.py with a larger --n.")
rng = random.Random(args.seed)
items = rng.sample(hall, m) + rng.sample(fact, m)
rng.shuffle(items)
print(f"balanced set: {m} hallucinated + {m} factual = {len(items)} items")

done = {r["id"] for r in read_jsonl(args.out)}
for it in items:
    if it["id"] in done:
        continue
    t0 = time.time()
    samples = [chat(GEN_PROMPTS[i + 1].format(claim=it["claim"]), args.model, args.temp) for i in range(args.k)]
    conflicts = []
    for s in samples:
        ans = chat(CONFLICT_PROMPT.format(p1=it["reference"], p2=s), args.model, 0.0, 80)
        v = parse_first_word(ans, "yes", "no")
        conflicts.append(0 if v is None else v)  # unparsable counts as no conflict
    append_jsonl(args.out, {"id": it["id"], "claim": it["claim"], "label_true": it["label_true"],
                            "reference": it["reference"], "hallucinated": it["hallucinated"],
                            "samples": samples, "conflicts": conflicts})
    print(f"[{it['id']}] {time.time() - t0:.1f}s  conflicts={sum(conflicts)}/{args.k}  true_label={it['hallucinated']}")
