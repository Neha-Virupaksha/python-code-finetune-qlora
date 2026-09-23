"""
Evaluate the QLoRA fine-tune from train_qlora.py: compares the base model
against the fine-tuned (base + LoRA adapter) model on the held-out eval
split that train_qlora.py saved to eval_holdout.jsonl.

Run this in the same Colab session/runtime, after train_qlora.py has
finished and produced its output directory (OUTPUT_DIR below), so the base
model doesn't need to be re-downloaded:

  !python evaluate.py

Produces two things:
  1. A quantitative comparison: mean per-token loss / perplexity on the
     held-out completions, computed for the base model and the fine-tuned
     model on the exact same examples. Lower loss / perplexity on the
     fine-tuned model = it has measurably adapted to this data distribution.
  2. A qualitative side-by-side: for a handful of held-out prompts, the
     actual generated completions from both models next to the reference
     output, so you can eyeball the difference in style/correctness.

Both are written to eval_report.md as well as printed to stdout.
"""

import json
import math
import random

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# ---------------------------------------------------------------------------
# Config -- must match train_qlora.py
# ---------------------------------------------------------------------------
BASE_MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
OUTPUT_DIR = "qwen2.5-coder-pandas-lora-v2"  # where train_qlora.py saved the adapter
EVAL_FILE = "eval_holdout.jsonl"             # written by train_qlora.py
MAX_LENGTH = 512                             # same as training
NUM_QUALITATIVE_SAMPLES = 5
MAX_NEW_TOKENS = 200
SEED = 42

random.seed(SEED)
torch.manual_seed(SEED)


# ---------------------------------------------------------------------------
# 1. Load held-out eval examples
# ---------------------------------------------------------------------------
def load_eval_examples(path):
    examples = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


print(f"Loading eval set from {EVAL_FILE}...")
eval_examples = load_eval_examples(EVAL_FILE)
print(f"Loaded {len(eval_examples)} held-out examples.")


# ---------------------------------------------------------------------------
# 2. Prompt formatting -- must match train_qlora.py's format_example, but
#    kept as two pieces (prompt-only vs prompt+response) so we can mask the
#    prompt tokens out of the loss and know where to cut for generation.
# ---------------------------------------------------------------------------
def build_prompt(instruction, input_text):
    if input_text:
        return (
            f"### Instruction:\n{instruction}\n\n"
            f"### Input:\n{input_text}\n\n"
            f"### Response:\n"
        )
    return f"### Instruction:\n{instruction}\n\n### Response:\n"


# ---------------------------------------------------------------------------
# 3. Load base model in 4-bit + tokenizer, then attach the LoRA adapter.
#    We load the base weights ONCE. peft.PeftModel wraps them and exposes a
#    disable_adapter() context manager, which lets us evaluate "base" and
#    "fine-tuned" behavior on the same in-memory model without a second,
#    memory-expensive load of the base weights.
# ---------------------------------------------------------------------------
print("Loading base model in 4-bit...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
    dtype=torch.float16,
)

print(f"Attaching LoRA adapter from {OUTPUT_DIR}...")
model = PeftModel.from_pretrained(base_model, OUTPUT_DIR)
model.eval()


# ---------------------------------------------------------------------------
# 4. Quantitative comparison: mean per-token loss + perplexity on the
#    reference completions, with and without the adapter enabled.
# ---------------------------------------------------------------------------
def compute_mean_loss(model, examples, adapter_enabled):
    """Weighted-average per-token cross-entropy loss over completion tokens
    only (prompt tokens are masked out with label = -100), so the metric
    reflects how well the model predicts the *response*, not the
    instruction. Weighted by token count per example for a correct overall
    average across examples of different lengths."""
    total_loss_x_tokens = 0.0
    total_tokens = 0

    context = torch.no_grad()
    adapter_ctx = (
        model.disable_adapter() if not adapter_enabled else _NullContext()
    )

    with context, adapter_ctx:
        for ex in examples:
            prompt = build_prompt(ex["instruction"], ex.get("input", ""))
            full_text = prompt + ex["output"] + tokenizer.eos_token

            prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            full = tokenizer(
                full_text,
                add_special_tokens=False,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            )
            input_ids = full["input_ids"].to(model.device)
            attention_mask = full["attention_mask"].to(model.device)

            labels = input_ids.clone()
            prompt_len = min(len(prompt_ids), input_ids.shape[1])
            labels[:, :prompt_len] = -100

            n_completion_tokens = int((labels != -100).sum().item())
            if n_completion_tokens == 0:
                continue

            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            # out.loss is already the mean CE over non-masked (completion)
            # tokens for this example; weight it back up by token count so
            # we can correctly average across examples of different lengths.
            total_loss_x_tokens += out.loss.item() * n_completion_tokens
            total_tokens += n_completion_tokens

    mean_loss = total_loss_x_tokens / total_tokens
    perplexity = math.exp(mean_loss)
    return mean_loss, perplexity, total_tokens


class _NullContext:
    def __enter__(self):
        return None

    def __exit__(self, *args):
        return False


print(f"\nScoring base model on {len(eval_examples)} held-out examples...")
base_loss, base_ppl, n_tok = compute_mean_loss(model, eval_examples, adapter_enabled=False)
print(f"  base model: loss={base_loss:.4f}  perplexity={base_ppl:.3f}  ({n_tok} completion tokens)")

print(f"Scoring fine-tuned model on {len(eval_examples)} held-out examples...")
ft_loss, ft_ppl, _ = compute_mean_loss(model, eval_examples, adapter_enabled=True)
print(f"  fine-tuned model: loss={ft_loss:.4f}  perplexity={ft_ppl:.3f}")

loss_delta = base_loss - ft_loss
ppl_rel_improvement = (base_ppl - ft_ppl) / base_ppl * 100

print(f"\nLoss reduction: {loss_delta:.4f} (lower is better)")
print(f"Perplexity improvement: {ppl_rel_improvement:.1f}% relative to base")


# ---------------------------------------------------------------------------
# 5. Qualitative side-by-side generations
# ---------------------------------------------------------------------------
def generate(model, prompt, adapter_enabled):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_LENGTH).to(
        model.device
    )
    adapter_ctx = model.disable_adapter() if not adapter_enabled else _NullContext()
    with torch.no_grad(), adapter_ctx:
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)


print(f"\nGenerating {NUM_QUALITATIVE_SAMPLES} qualitative comparisons...")
sample_indices = random.sample(range(len(eval_examples)), min(NUM_QUALITATIVE_SAMPLES, len(eval_examples)))

qualitative_results = []
for i, idx in enumerate(sample_indices):
    ex = eval_examples[idx]
    prompt = build_prompt(ex["instruction"], ex.get("input", ""))
    print(f"  [{i + 1}/{len(sample_indices)}] generating...")
    base_gen = generate(model, prompt, adapter_enabled=False)
    ft_gen = generate(model, prompt, adapter_enabled=True)
    qualitative_results.append(
        {
            "instruction": ex["instruction"],
            "input": ex.get("input", ""),
            "reference": ex["output"],
            "base_generation": base_gen,
            "finetuned_generation": ft_gen,
        }
    )


# ---------------------------------------------------------------------------
# 6. Write report
# ---------------------------------------------------------------------------
report_lines = []
report_lines.append("# QLoRA Fine-tune Evaluation Report\n")
report_lines.append(f"Base model: `{BASE_MODEL}`  \nAdapter: `{OUTPUT_DIR}`  \nEval set: `{EVAL_FILE}` ({len(eval_examples)} examples)\n")

report_lines.append("## Quantitative: held-out completion loss / perplexity\n")
report_lines.append("| Model | Mean loss | Perplexity |")
report_lines.append("|---|---|---|")
report_lines.append(f"| Base | {base_loss:.4f} | {base_ppl:.3f} |")
report_lines.append(f"| Fine-tuned | {ft_loss:.4f} | {ft_ppl:.3f} |")
report_lines.append(f"\n**Loss reduction:** {loss_delta:.4f}  \n**Perplexity improvement:** {ppl_rel_improvement:.1f}% relative to base\n")
if loss_delta > 0:
    report_lines.append(
        "The fine-tuned model assigns higher probability to the reference "
        "completions than the base model does on held-out data -- i.e. it "
        "has measurably adapted toward this dataset's style/content.\n"
    )
else:
    report_lines.append(
        "**Warning:** the fine-tuned model does not show a loss improvement "
        "over the base model on held-out data. This can mean the adapter "
        "didn't train effectively, training was too short, the learning "
        "rate was off, or the adapter directory being loaded doesn't match "
        "the run you expect -- worth double-checking before relying on "
        "this checkpoint.\n"
    )

report_lines.append("## Qualitative: sample generations\n")
for i, r in enumerate(qualitative_results):
    report_lines.append(f"### Sample {i + 1}\n")
    report_lines.append(f"**Instruction:** {r['instruction']}\n")
    if r["input"]:
        report_lines.append(f"**Input:** {r['input']}\n")
    report_lines.append(f"**Reference output:**\n```\n{r['reference']}\n```\n")
    report_lines.append(f"**Base model output:**\n```\n{r['base_generation']}\n```\n")
    report_lines.append(f"**Fine-tuned model output:**\n```\n{r['finetuned_generation']}\n```\n")
    report_lines.append("---\n")

report_text = "\n".join(report_lines)
with open("eval_report.md", "w") as f:
    f.write(report_text)

print("\nDone. Full report written to eval_report.md")
