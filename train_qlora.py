"""
QLoRA fine-tuning of Qwen2.5-Coder-1.5B-Instruct for Python data-analysis
code generation (pandas / numpy / sklearn).

Run this in Google Colab with a free T4 GPU:
  Runtime > Change runtime type > T4 GPU

Install dependencies first (run in a Colab cell):
  !pip install -q transformers peft bitsandbytes accelerate datasets trl huggingface_hub
"""

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
from transformers import TrainerCallback

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE_MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
DATASET_NAME = "iamtarun/python_code_instructions_18k_alpaca"
OUTPUT_DIR = "qwen2.5-coder-pandas-lora-v2"
HF_REPO_ID = "NehaVirupaksha/qwen2.5-coder-pandas-lora-v2"

# Keywords used to filter the dataset down to data-analysis-flavored examples
DATA_ANALYSIS_KEYWORDS = [
    "pandas", "numpy", "dataframe", "sklearn", "scikit-learn",
    "matplotlib", "seaborn", "csv", "dataset", "plot", "array",
]

# ---------------------------------------------------------------------------
# 1. Load and filter dataset
# ---------------------------------------------------------------------------
print("Loading dataset...")
raw_dataset = load_dataset(DATASET_NAME, split="train")


def is_data_analysis_example(example):
    text = (example.get("instruction", "") + " " + example.get("input", "") +
            " " + example.get("output", "")).lower()
    return any(kw in text for kw in DATA_ANALYSIS_KEYWORDS)


filtered_dataset = raw_dataset.filter(is_data_analysis_example)
print(f"Filtered dataset size: {len(filtered_dataset)} / {len(raw_dataset)}")

# Keep it small for a fast, free-tier-friendly training run.
# Increase this once you've validated the pipeline works end to end.
filtered_dataset = filtered_dataset.shuffle(seed=42).select(
    range(min(4000, len(filtered_dataset)))
)

# Hold out a small eval split for before/after comparison later
split_dataset = filtered_dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = split_dataset["train"]
eval_dataset = split_dataset["test"]
eval_dataset.to_json("eval_holdout.jsonl")  # used later by evaluate.py
print(f"Train: {len(train_dataset)} | Eval holdout: {len(eval_dataset)}")


def format_example(example):
    instruction = example["instruction"]
    input_text = example.get("input", "")
    output = example["output"]
    if input_text:
        prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}"
    else:
        prompt = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"
    return {"text": prompt}


train_dataset = train_dataset.map(
    format_example, remove_columns=train_dataset.column_names
)

# ---------------------------------------------------------------------------
# 2. Load base model in 4-bit (QLoRA)
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

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
    dtype=torch.float16,
)
model = prepare_model_for_kbit_training(model)

# Qwen2.5-Coder's config declares torch_dtype=bfloat16 by default, even though
# we've loaded the actual weights in float16. Several internal transformers
# code paths (e.g. resize_token_embeddings, used when the tokenizer's
# pad/bos/eos ids don't match the model config) fall back to
# model.config.torch_dtype when they can't infer a dtype from the model's
# parameters directly -- which for a 4-bit quantized model is ambiguous,
# since most params are packed as uint8. Forcing this to float16 makes sure
# any such fallback stays consistent with the rest of the model.
model.config.torch_dtype = torch.float16

# ---------------------------------------------------------------------------
# 3. Apply LoRA adapters
# ---------------------------------------------------------------------------
lora_config = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

def cast_bf16_to_fp16(m):
    """Fix up parameter/buffer dtypes so fp16 mixed-precision training works
    on a T4 (no native bf16 support).

    Two separate rules apply here:
      - Frozen/base-model params and buffers just need to not be bfloat16 --
        float16 is fine for them, since they're not touched by the
        GradScaler. Some layers (e.g. RMSNorm, rotary embedding caches)
        register bf16 tensors as buffers rather than parameters.
      - Trainable parameters (the LoRA adapter weights) must stay float32.
        torch's fp16 GradScaler requires trainable ("master") weights to be
        fp32 -- if a trainable param is fp16, its .grad will be fp16 too, and
        GradScaler.unscale_ explicitly raises
        ValueError("Attempting to unscale FP16 gradients") rather than risk
        silent underflow. Casting LoRA params to fp16 (as an earlier version
        of this helper did) breaks that invariant.
    """
    for param in m.parameters():
        if param.requires_grad:
            if param.dtype != torch.float32:
                param.data = param.data.to(torch.float32)
        elif param.dtype == torch.bfloat16:
            param.data = param.data.to(torch.float16)
    for name, buf in m.named_buffers():
        if buf.dtype == torch.bfloat16:
            buf.data = buf.data.to(torch.float16)


# Run once now, up front.
cast_bf16_to_fp16(model)


class CastBf16ToFp16Callback(TrainerCallback):
    """Re-run the dtype sweep before every training step.

    We initially only ran this once, in on_train_begin, on the theory that
    whatever reintroduces bad dtypes (tokenizer/model special-token
    alignment, embedding resizing, etc.) happens once during trainer setup.
    In practice something is reintroducing an fp16 *trainable* parameter
    later than that -- likely lazily, on/around the first forward pass --
    which on_train_begin fires too early to catch, since it runs before the
    first step's forward/backward rather than between backward and
    clip_grad_norm (there's no hook available at that exact point).

    Rather than keep chasing the exact trigger, on_step_begin fires
    immediately before each step's forward/backward, so re-running the cheap
    dtype check there catches drift regardless of when or why it happens.
    The check itself is just a dtype comparison per parameter -- essentially
    free -- so doing it every step is not a meaningful cost.
    """

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        if model is not None:
            cast_bf16_to_fp16(model)

    def on_step_begin(self, args, state, control, model=None, **kwargs):
        if model is not None:
            cast_bf16_to_fp16(model)

# ---------------------------------------------------------------------------
# 4. Train
# ---------------------------------------------------------------------------
training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=2,
    learning_rate=2e-4,
    fp16=True,
    bf16=False,
    logging_steps=10,
    save_strategy="epoch",
    report_to="none",
    dataset_text_field="text",
    max_length=512,
)

trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    args=training_args,
    callbacks=[CastBf16ToFp16Callback()],
)

print("Starting training...")
trainer.train()

# ---------------------------------------------------------------------------
# 5. Save + push adapter weights to Hugging Face Hub
# ---------------------------------------------------------------------------
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

# To push to the Hub, run in a separate Colab cell first:
#   from huggingface_hub import notebook_login
#   notebook_login()
# Then:
#   model.push_to_hub(HF_REPO_ID)
#   tokenizer.push_to_hub(HF_REPO_ID)

print(f"Done. Adapter saved to ./{OUTPUT_DIR}")
print("Next: push to Hugging Face Hub, then run evaluate.py to compare base vs. fine-tuned.")
