# Self-contradiction hallucination detection with a local LLM (Ollama)

Status: in progress

## Question
With a small local model, how well does self-contradiction detect the model's own factual hallucinations,
and is the "any conflict" rule the best decision rule?

## Method
Idea from AutoHall (Cao et al., arXiv:2310.00259). I re-implemented it from scratch (own code, own prompts, Ollama instead of
ChatGPT/Llama-2); I did not copy the authors' code.
1. `build_dataset.py`: for each Climate-FEVER claim (SUPPORTS/REFUTES only) the model writes a reference, then classifies the
   claim using that reference. If the verdict disagrees with the ground-truth label, the reference is labelled hallucinated.
2. `detect.py`: balance the classes, sample K=6 more references with different prompts, and ask the model whether each
   conflicts with the original. The conflict vector is saved.
3. `analyze.py`: accuracy / precision / recall / F1 / AUROC, a threshold sweep over the conflict count, and a K ablation.

## My extensions
- Conflict-count thresholds and AUROC instead of only the "at least one conflict" rule
- K ablation (cost vs accuracy)
- Manual check of automatic label quality (N examples)

## How to run
```
ollama pull mistral
pip install -r requirements.txt
python build_dataset.py --model mistral --n 80
python detect.py --model mistral
python analyze.py
```
Quick smoke test without Climate-FEVER: add `--csv data/sample_claims.csv` to `build_dataset.py`.

## Results
(fill from results/*.csv after running; report N, model name, and the manual label-check percentage)

## Limitations
Small N, one model, labels produced by the same small model (noisy), one dataset, factuality only.

## Notes
Code scaffold written with help from an AI assistant; I ran the experiments and wrote the analysis.

## References
- Cao et al., AutoHall: Automated Factuality Hallucination Dataset Generation for LLMs (arXiv:2310.00259)
- Diggelmann et al., Climate-FEVER (2020)
- Manakul et al., SelfCheckGPT (2023)
