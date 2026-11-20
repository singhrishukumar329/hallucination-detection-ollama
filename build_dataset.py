"""Stage 1: build a model-specific hallucination dataset.
For each claim: model writes a reference -> model classifies the claim using that reference ->
if its verdict disagrees with the ground-truth label, the reference is labelled 'hallucinated'."""
import argparse
import time

from common import (GEN_PROMPTS, CLASSIFY_PROMPT, chat, parse_first_word, is_refusal,
                    load_claims, read_jsonl, append_jsonl)

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="mistral")
ap.add_argument("--n", type=int, default=80, help="number of claims to try")
ap.add_argument("--csv", default=None, help="optional csv with columns claim,label(true/false)")
ap.add_argument("--out", default="data/built.jsonl")
ap.add_argument("--seed", type=int, default=42)
args = ap.parse_args()

claims = load_claims(args.n, args.csv, args.seed)
done = {r["id"] for r in read_jsonl(args.out)}
print(f"{len(claims)} claims, {len(done)} already done")

for c in claims:
    if c["id"] in done:
        continue
    t0 = time.time()
    ref = chat(GEN_PROMPTS[0].format(claim=c["claim"]), args.model, temperature=0.1)
    rec = dict(c, reference=ref)
    if is_refusal(ref):
        rec["skipped"] = "refusal"
    else:
        verdict = chat(CLASSIFY_PROMPT.format(claim=c["claim"], reference=ref), args.model, 0.0, 120)
        pred_true = parse_first_word(verdict, "true", "false")
        if pred_true is None:
            rec["skipped"] = "unparsable"
        else:
            rec["verdict_true"] = pred_true
            rec["hallucinated"] = int(pred_true != c["label_true"])
    append_jsonl(args.out, rec)
    print(f"[{c['id']}] {time.time() - t0:.1f}s  {rec.get('skipped', 'hallucinated=' + str(rec['hallucinated']))}")

rows = read_jsonl(args.out)
ok = [r for r in rows if "hallucinated" in r]
h = sum(r["hallucinated"] for r in ok)
print(f"\nkept {len(ok)} / {len(rows)}; hallucinated {h} ({100 * h / max(len(ok), 1):.1f}%)")
