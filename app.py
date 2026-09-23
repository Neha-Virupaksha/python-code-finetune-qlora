"""
Gradio demo comparing base vs. fine-tuned Qwen2.5-Coder on Python
data-analysis code generation. Deploy this file to Hugging Face Spaces
(free tier) as app.py, with a requirements.txt alongside it.

To deploy:
  1. Create a new Space on huggingface.co/new-space, SDK = Gradio
  2. Upload this file as app.py, and requirements.txt
  3. Space builds automatically and gives you a public link
"""

import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
ADAPTER_PATH = "Neha2virupaksha/qwen2.5-coder-pandas-lora-v2"

print("Loading tokenizer and models (this runs once at Space startup)...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, torch_dtype=torch.float32, device_map="cpu"
)
finetuned_model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)


def generate(model, instruction):
    prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        output_ids = model.generate(
            **inputs, max_new_tokens=200, do_sample=False, temperature=0.0
        )
    output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return output.split("### Response:")[-1].strip()


def compare(instruction):
    base_output = generate(base_model, instruction)
    ft_output = generate(finetuned_model, instruction)
    return base_output, ft_output


demo = gr.Interface(
    fn=compare,
    inputs=gr.Textbox(
        label="Instruction",
        placeholder="e.g. Write a pandas function to remove duplicate rows and fill missing values with the column mean",
        lines=3,
    ),
    outputs=[
        gr.Code(label="Base Model Output", language="python"),
        gr.Code(label="Fine-Tuned Model Output", language="python"),
    ],
    title="Python Data-Analysis Code Assistant — Base vs. Fine-Tuned",
    description=(
        "Qwen2.5-Coder-1.5B-Instruct fine-tuned with QLoRA on pandas/numpy/"
        "sklearn code-generation examples. On held-out evaluation, the "
        "fine-tuned model showed a 20% perplexity improvement, eliminated "
        "syntax errors entirely (0/40 vs. 6/40 for base), and improved "
        "sandboxed code-execution success rate from 55% to 62.5%. "
        "Try a prompt below to compare outputs directly."
    ),
)

if __name__ == "__main__":
    demo.launch()
