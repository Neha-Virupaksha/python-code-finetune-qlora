# QLoRA Fine-tuning of Qwen2.5-Coder-1.5B for Pandas/NumPy/Scikit-learn Code Generation

Fine-tunes [Qwen2.5-Coder-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct) with QLoRA on a filtered subset of Python instruction data, targeting data-analysis code generation (pandas / numpy / sklearn / matplotlib). Trained on a free-tier Google Colab T4 GPU.

## Results

Evaluated on a held-out split (never seen during training) using three complementary methods:

| Eval | What it measures | Base | Fine-tuned | Change |
|---|---|---|---|---|
| Held-out perplexity | How well the model predicts reference completions | 2.166 | 1.731 | **-20.0%** (lower is better) |
| Code execution success (n=40, sandboxed) | Whether generated code actually runs | 55.0% (22/40) | 62.5% (25/40) | **+7.5pp** |
| — of which, syntax errors | Code that doesn't even compile | 6/40 | **0/40** | **eliminated** |

**Summary:** the fine-tune reliably improved output *format* — the base model wraps code in markdown fences with trailing explanatory prose and sometimes truncates mid-thought; the fine-tuned model emits clean, direct code matching the target style. This shows up concretely as a 20% perplexity reduction and the complete elimination of syntax errors in sandboxed execution testing. Actual code-execution success also improved, though more modestly (+7.5 percentage points) and with some individual examples flipping in each direction — this is a real but measured gain in runnable correctness, not a dramatic capability jump. See [Limitations](#limitations) below for what this evaluation does and doesn't establish.

## Setup

```bash
pip install -q transformers peft bitsandbytes accelerate datasets trl huggingface_hub
```

Designed to run in Google Colab on a free T4 GPU (`Runtime > Change runtime type > T4 GPU`).

## Training

```bash
python train_qlora.py
```

- **Base model:** Qwen2.5-Coder-1.5B-Instruct, loaded in 4-bit NF4 (QLoRA)
- **Data:** [`iamtarun/python_code_instructions_18k_alpaca`](https://huggingface.co/datasets/iamtarun/python_code_instructions_18k_alpaca), keyword-filtered down to ~3,300 data-analysis-flavored examples (pandas, numpy, sklearn, matplotlib, etc.), capped at 4,000 for a fast, free-tier-friendly run
- **LoRA config:** r=32, alpha=64, targeting `q_proj/k_proj/v_proj/o_proj`
- **Precision:** fp16 throughout (T4 has no native bf16 tensor-core support — see [Notes on T4 + fp16](#notes-on-t4--fp16) below)
- **2 epochs**, batch size 2 with gradient accumulation of 4 (effective batch size 8)

A 10% held-out split is written to `eval_holdout.jsonl` before training and used by every eval script below.

## Evaluation

Three scripts, each testing a different question:

| Script | Question it answers |
|---|---|
| `evaluate.py` | How much did the model's predictions shift toward the target data distribution? (perplexity) |
| `evaluate_code_execution.py` + `sandbox_worker.py` | Does the generated code actually run? (sandboxed execution, HumanEval-style) |
| `evaluate_similarity.py` | *(included for completeness — see caveat below)* token-overlap similarity to reference code |

```bash
python evaluate.py                     # perplexity comparison + 5 qualitative samples -> eval_report.md
python evaluate_code_execution.py      # sandboxed execution comparison -> eval_code_execution_report.md
```

`evaluate_code_execution.py` requires `sandbox_worker.py` in the same working directory — it's launched as a subprocess for isolation, not imported directly.

## Limitations

Being upfront about what this evaluation does and doesn't show:

- **Execution success ≠ correctness.** Code can run without error and still get the wrong answer. The sandboxed eval catches crashes, syntax errors, and hallucinated file/API references — it does not verify output against the reference solution.
- **Small sample size (n=40)** for the execution eval. A handful of examples flip in either direction; the net +7.5pp is a real but not overwhelming signal, and shouldn't be read as "dramatically better," just measurably better.
- **Shared dataset artifacts affect both models equally.** A meaningful chunk of failures (for both base and fine-tuned) are `FileNotFoundError`s on hallucinated CSV filenames — an artifact of the source dataset's instructions assuming an unprovided data file, not something fine-tuning could reasonably fix.
- **The token-overlap similarity metric (`evaluate_similarity.py`) is included but not featured in results above.** It scores raw vocabulary overlap with the reference and structurally rewards verbose output — since the fine-tune was trained toward terser, cleaner completions, this metric shows a *negative* "improvement" that reflects a scoring bias (fewer words = fewer chances to overlap), not an actual regression. Kept in the repo for transparency, not as a headline number.
- **Single training run, greedy decoding, one dataset filter.** No hyperparameter sweep, no multiple seeds. Numbers above describe this specific run.

## Notes on T4 + fp16

Qwen2.5-Coder's config declares `torch_dtype: bfloat16` by default, but T4 GPUs (Turing architecture) have no native bf16 tensor-core support. Training scripts here force fp16 consistently and include a dtype-correction pass (`cast_bf16_to_fp16` in `train_qlora.py`) re-applied every training step, because certain internal `transformers` operations (tokenizer/model special-token alignment) can silently reintroduce bf16 tensors at unpredictable points during setup — a callback re-applying the fix once wasn't sufficient; it needed to run before every step. If adapting this for an A100/H100 or similar Ampere+ GPU, bf16 can likely be used natively and this workaround dropped.

## Files

| File | Purpose |
|---|---|
| `train_qlora.py` | QLoRA fine-tuning script |
| `eval_holdout.jsonl` | Held-out eval split (written by training) |
| `evaluate.py` | Perplexity comparison + qualitative samples |
| `evaluate_code_execution.py` | Sandboxed code-execution comparison |
| `sandbox_worker.py` | Isolated subprocess worker used by the execution eval |
| `evaluate_similarity.py` | Token-overlap similarity metric (see limitations) |
| `eval_report.md` | Generated perplexity/qualitative report |
| `eval_code_execution_report.md` | Generated execution-eval report |
