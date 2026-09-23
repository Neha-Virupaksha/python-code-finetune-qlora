"""
Follow-up eval: does the fine-tune actually produce *better working code*,
not just better-formatted code?

evaluate.py (the first eval script) showed the fine-tuned model matches the
target output *format* much more closely (no markdown fences, no trailing
prose) and scores meaningfully lower perplexity on held-out completions.
That's a real result, but it doesn't tell you whether the code itself is
more correct -- a model can learn "stop after the code block" without
getting any better at writing pandas/sklearn.

This script tests that directly: it generates completions from both models
on a batch of held-out prompts and *actually executes* the generated code
in a sandboxed subprocess (timeout-limited, dangerous builtins/imports
blocked -- similar in spirit to how HumanEval-style code benchmarks work),
then compares execution success rate between base and fine-tuned.

It also checks a cheap secondary signal: whether the generated code
actually uses the library the instruction called for (pandas/numpy/sklearn/
etc), since "runs without crashing" and "uses the right tool" are both part
of "did the model get better at this domain."

Run in the same Colab session as before:
  !python evaluate_code_execution.py
"""

import json
import random
import re
import subprocess
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE_MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
OUTPUT_DIR = "qwen2.5-coder-pandas-lora-v2"
EVAL_FILE = "eval_holdout.jsonl"
N_SAMPLES = 40          # how many held-out examples to test (raise for more confidence, costs more time)
MAX_NEW_TOKENS = 300    # generous, to avoid penalizing either model for truncation
EXEC_TIMEOUT_SECONDS = 8
SEED = 42

# Keywords used to check whether generated code actually engages the
# library the instruction called for (cheap heuristic, not a proof of
# correctness -- just "did it reach for the right tool").
LIBRARY_KEYWORDS = {
    "pandas": ["pandas", "pd.", "dataframe", "read_csv"],
    "numpy": ["numpy", "np."],
    "sklearn": ["sklearn", "scikit"],
    "matplotlib": ["matplotlib", "plt.", "pyplot"],
}

random.seed(SEED)
torch.manual_seed(SEED)


# ---------------------------------------------------------------------------
# 1. Load held-out eval examples, subsample
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
all_examples = load_eval_examples(EVAL_FILE)
n = min(N_SAMPLES, len(all_examples))
eval_examples = random.sample(all_examples, n)
print(f"Testing on {n} randomly sampled held-out examples.")


# ---------------------------------------------------------------------------
# 2. Prompting + generation (same template as training/evaluate.py)
# ---------------------------------------------------------------------------
def build_prompt(instruction, input_text):
    if input_text:
        return (
            f"### Instruction:\n{instruction}\n\n"
            f"### Input:\n{input_text}\n\n"
            f"### Response:\n"
        )
    return f"### Instruction:\n{instruction}\n\n### Response:\n"


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


class _NullContext:
    def __enter__(self):
        return None

    def __exit__(self, *args):
        return False


def generate(prompt, adapter_enabled):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(model.device)
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


# ---------------------------------------------------------------------------
# 3. Extract code from a generation (strip markdown fences / trailing prose)
# ---------------------------------------------------------------------------
FENCE_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)


def extract_code(text):
    match = FENCE_RE.search(text)
    if match:
        return match.group(1).strip()
    # No fence found (this is the common case for the fine-tuned model,
    # which was trained to emit code directly). Trailing prose after the
    # code block is harder to strip reliably without fences, so we just use
    # the raw text -- worst case this slightly disadvantages generations
    # that ramble after valid code, which affects both models equally.
    return text.strip()


# ---------------------------------------------------------------------------
# 4. Sandboxed execution
#
# Generated code is executed by sandbox_worker.py, launched as a completely
# separate OS process via subprocess.run() -- not via Python's
# multiprocessing module. This process already has an initialized CUDA
# context (from loading the model), and fork()-ing a process that owns a
# CUDA context is unsafe; multiprocessing's "spawn" alternative would
# instead re-execute this script's top-level code (including the model
# load) in every child, since none of it is guarded by
# `if __name__ == "__main__":`. A plain subprocess sidesteps both: a fresh
# interpreter that only imports sandbox_worker.py, fully independent of
# this process's CUDA/model state.
#
# Within the worker: dangerous builtins are removed (open, eval, exec,
# compile, input, exit), dangerous modules are blocked at import time (os,
# sys, subprocess, shutil, socket, requests, urllib, ctypes, pathlib), and
# matplotlib is forced to a non-interactive backend with plt.show() patched
# to a no-op so plotting code doesn't hang. This mirrors how standard
# code-generation benchmarks (e.g. HumanEval) execute untrusted model
# output: isolate it, bound its runtime, only allow the compute libraries
# the task is actually about.
# ---------------------------------------------------------------------------
SANDBOX_SCRIPT = "sandbox_worker.py"
RESULT_MARKER = "###SANDBOX_RESULT###"


def run_sandboxed(code, timeout=EXEC_TIMEOUT_SECONDS):
    try:
        proc = subprocess.run(
            [sys.executable, SANDBOX_SCRIPT],
            input=code,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return "timeout", f"exceeded {timeout}s"

    # The executed code may itself print to stdout; our result is always
    # the (last) line carrying RESULT_MARKER, so search for that
    # specifically rather than trying to parse arbitrary lines as JSON.
    for line in reversed(proc.stdout.splitlines()):
        if line.startswith(RESULT_MARKER):
            try:
                result = json.loads(line[len(RESULT_MARKER):])
                return result["status"], result["error"]
            except (json.JSONDecodeError, KeyError):
                break

    # No parseable result line found -- the worker crashed before printing
    # its result. Report what we can rather than silently miscounting it.
    stderr_snip = proc.stderr.strip()[-300:] if proc.stderr else "(no stderr)"
    return "runtime_error", f"sandbox produced no result; exit code {proc.returncode}; stderr: {stderr_snip}"


# ---------------------------------------------------------------------------
# 5. Library-usage heuristic
# ---------------------------------------------------------------------------
def mentioned_libraries(instruction_and_input):
    text = instruction_and_input.lower()
    mentioned = set()
    for lib, keywords in LIBRARY_KEYWORDS.items():
        if any(kw.split(".")[0] in text for kw in keywords):
            mentioned.add(lib)
    return mentioned


def used_libraries(code):
    text = code.lower()
    used = set()
    for lib, keywords in LIBRARY_KEYWORDS.items():
        if any(kw in text for kw in keywords):
            used.add(lib)
    return used


# ---------------------------------------------------------------------------
# 6. Run the eval
# ---------------------------------------------------------------------------
def run_eval(adapter_enabled, label):
    print(f"\nGenerating + executing {label} model completions on {n} examples...")
    results = []
    for i, ex in enumerate(eval_examples):
        prompt = build_prompt(ex["instruction"], ex.get("input", ""))
        gen_text = generate(prompt, adapter_enabled=adapter_enabled)
        code = extract_code(gen_text)
        status, error = run_sandboxed(code)

        req_libs = mentioned_libraries(ex["instruction"] + " " + ex.get("input", ""))
        used_libs = used_libraries(code)
        lib_match = req_libs.issubset(used_libs) if req_libs else None

        results.append(
            {
                "instruction": ex["instruction"],
                "generated_code": code,
                "status": status,
                "error": error,
                "required_libraries": sorted(req_libs),
                "used_libraries": sorted(used_libs),
                "library_match": lib_match,
            }
        )
        print(f"  [{i + 1}/{n}] {status}" + (f" ({error})" if error and status != "success" else ""))
    return results


base_results = run_eval(adapter_enabled=False, label="base")
ft_results = run_eval(adapter_enabled=True, label="fine-tuned")


# ---------------------------------------------------------------------------
# 7. Aggregate + report
# ---------------------------------------------------------------------------
def summarize(results):
    total = len(results)
    success = sum(1 for r in results if r["status"] == "success")
    syntax_err = sum(1 for r in results if r["status"] == "syntax_error")
    runtime_err = sum(1 for r in results if r["status"] == "runtime_error")
    timeout = sum(1 for r in results if r["status"] == "timeout")

    lib_checked = [r for r in results if r["library_match"] is not None]
    lib_match_rate = (
        sum(1 for r in lib_checked if r["library_match"]) / len(lib_checked)
        if lib_checked
        else None
    )

    return {
        "total": total,
        "success_rate": success / total,
        "success": success,
        "syntax_error": syntax_err,
        "runtime_error": runtime_err,
        "timeout": timeout,
        "library_match_rate": lib_match_rate,
        "library_checked_n": len(lib_checked),
    }


base_summary = summarize(base_results)
ft_summary = summarize(ft_results)

print("\n" + "=" * 60)
print("EXECUTION SUCCESS RATE")
print("=" * 60)
print(f"  Base:       {base_summary['success']}/{base_summary['total']} "
      f"({base_summary['success_rate'] * 100:.1f}%)")
print(f"  Fine-tuned: {ft_summary['success']}/{ft_summary['total']} "
      f"({ft_summary['success_rate'] * 100:.1f}%)")

if base_summary["library_match_rate"] is not None:
    print(f"\n  Base library-match rate:       {base_summary['library_match_rate'] * 100:.1f}% "
          f"(n={base_summary['library_checked_n']})")
    print(f"  Fine-tuned library-match rate: {ft_summary['library_match_rate'] * 100:.1f}% "
          f"(n={ft_summary['library_checked_n']})")

# Write full report
report = []
report.append("# Code Execution Eval: Base vs Fine-tuned\n")
report.append(f"Base model: `{BASE_MODEL}`  \nAdapter: `{OUTPUT_DIR}`  \nSample size: {n} held-out examples (seed={SEED})\n")

report.append("## Execution success rate\n")
report.append("| Model | Success | Syntax error | Runtime error | Timeout | Success rate |")
report.append("|---|---|---|---|---|---|")
report.append(
    f"| Base | {base_summary['success']} | {base_summary['syntax_error']} | "
    f"{base_summary['runtime_error']} | {base_summary['timeout']} | {base_summary['success_rate'] * 100:.1f}% |"
)
report.append(
    f"| Fine-tuned | {ft_summary['success']} | {ft_summary['syntax_error']} | "
    f"{ft_summary['runtime_error']} | {ft_summary['timeout']} | {ft_summary['success_rate'] * 100:.1f}% |"
)

if base_summary["library_match_rate"] is not None:
    report.append("\n## Library-usage match rate\n")
    report.append(
        "Of the examples whose instruction named a specific library "
        "(pandas/numpy/sklearn/matplotlib), the fraction where the "
        "generated code actually used that library:\n"
    )
    report.append(
        f"- Base: {base_summary['library_match_rate'] * 100:.1f}% "
        f"(n={base_summary['library_checked_n']})\n"
        f"- Fine-tuned: {ft_summary['library_match_rate'] * 100:.1f}% "
        f"(n={ft_summary['library_checked_n']})\n"
    )

delta = ft_summary["success_rate"] - base_summary["success_rate"]
report.append("\n## Interpretation\n")
if delta > 0:
    report.append(
        f"The fine-tuned model's generated code executed successfully "
        f"{delta * 100:.1f} percentage points more often than the base "
        f"model's, on the same {n} held-out prompts. Combined with the "
        f"perplexity result from evaluate.py, this suggests the fine-tune "
        f"improved actual code correctness, not just output formatting.\n"
    )
elif delta < 0:
    report.append(
        f"The fine-tuned model's generated code executed successfully "
        f"{abs(delta) * 100:.1f} percentage points *less* often than the "
        f"base model's on these {n} prompts. This is an important, honest "
        f"result: the earlier perplexity/format improvement does not "
        f"appear to translate into more correct code here. Worth reporting "
        f"alongside the format result rather than instead of it.\n"
    )
else:
    report.append(
        "No difference in execution success rate between base and "
        "fine-tuned on this sample. The earlier improvement looks to be "
        "primarily about output formatting rather than code correctness.\n"
    )

report.append(
    "\n*Caveat: execution success is necessary but not sufficient for "
    "correctness -- code can run without error and still not solve the "
    "task properly (e.g. wrong logic, wrong output). This eval catches "
    "crashes, syntax errors, and hallucinated APIs/files; it does not "
    "verify semantic correctness against the reference output.*\n"
)

report.append("\n## Per-example detail\n")
for i, (b, f_) in enumerate(zip(base_results, ft_results)):
    report.append(f"### Example {i + 1}\n")
    report.append(f"**Instruction:** {b['instruction']}\n")
    report.append(f"**Base:** `{b['status']}`" + (f" -- {b['error']}" if b["error"] else "") + "  ")
    report.append(f"**Fine-tuned:** `{f_['status']}`" + (f" -- {f_['error']}" if f_["error"] else "") + "\n")

report_text = "\n".join(report)
with open("eval_code_execution_report.md", "w") as f:
    f.write(report_text)

# Also dump raw results as JSON for anyone who wants to dig into specifics.
with open("eval_code_execution_results.json", "w") as f:
    json.dump({"base": base_results, "finetuned": ft_results, "base_summary": base_summary, "finetuned_summary": ft_summary}, f, indent=2)

print("\nDone. Full report written to eval_code_execution_report.md")
print("Raw per-example results written to eval_code_execution_results.json")
