# export_onnx.py
import torch
import os
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from optimum.exporters.onnx import main_export

# Ensure CUDA cache is cleared
torch.cuda.empty_cache()

# Set the model name and output path
model_name = "Qwen/Qwen2.5-Coder-1.5B-Instruct"  # Use the smaller model for testing
onnx_path = Path("./onnx_model")
os.makedirs(onnx_path, exist_ok=True)

# Export directly from hub to ONNX
main_export(
    model_name_or_path=model_name,
    output=onnx_path,
    task="text-generation",
    opset=14,
    device="cpu",  # Force CPU
    no_post_process=True,  # Skip post-processing to save memory
    trust_remote_code=True
)

print(f"Model successfully exported to {onnx_path}")