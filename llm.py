import os
from abc import ABC, abstractmethod
from typing import Literal

import torch
from dotenv import load_dotenv
from openai import OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import importlib.util
from optimum.onnxruntime import ORTModelForCausalLM
from optimum.exporters.onnx import main_export
from pathlib import Path
import tempfile
load_dotenv()


def get_model(model_name: Literal["qwen", "deepseek"] = "qwen") -> int:
    if model_name == "qwen":
        return Qwen()
    elif model_name == "deepseek":
        return Openai(
            api_key=os.getenv("LLM_API_KEY"), api_url=os.getenv("LLM_API_URL")
        )

    raise ValueError(f"Model {model_name} not available")


class LLMModel(ABC):

    def __init__(self):
        self.model_name = None
        self.model = None
        self.tokenizer = None
        self.bnb_config = None
        self.attn = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.do_sampling = False # use the model default as the default here
        self.onnx_path = Path("./onnx_model")

    def check_flash_att_compatibility(self) -> bool:
        """
        Check if flash_attention can be used, otherwise default to sdpa
        Args:
        Returns:
            bool value indicating if flash_attn can be used
        """

        if not torch.cuda.is_available():
            print("No GPU available, defaulting to CPU")
            return False

        if importlib.util.find_spec('flash_attn') is None:
            print(f"No package flash_attn available for import. Ensure it is installed and try again!")
            return False
        
        gpu_name = torch.cuda.get_device_name()
        gpu_idx = torch.cuda.current_device()
        # tuple value representing the minor and major capability of the gpu
        gpu_capability = torch.cuda.get_device_capability(gpu_idx)

        print(f"The following GPU is available: ")
        print(f"\tname: {gpu_name}")
        print(f"\tindex: {gpu_idx}")
        print(f"\tCapability: {gpu_capability[0]}.{gpu_capability[1]}")

        if gpu_capability[0] >= 8: # ampere=8
            print("gpu support flash_attn")
            return True
        else:
            print("gpu does not support flash_attn")
            return False
        
    def export_to_onnx(self) -> None:
        """
        Export the model to ONNX for faster inference
        Args:
            onnx_path: where do we store the ONNX exported model
        Return:
            None
        """
        
        
        os.makedirs(self.onnx_path, exist_ok=True)

        if self.model is not None and self.tokenizer is not None:
            torch.cuda.empty_cache()
            with tempfile.TemporaryDirectory() as tmp_dir:

                # perform onnx conversion on CPU as this is memory intensive
                # self.model = self.model.cpu()
                self.model.save_pretrained(tmp_dir)
                self.tokenizer.save_pretrained(tmp_dir)
                main_export(
                    model_name_or_path=tmp_dir,
                    output=self.onnx_path,
                    task='text-generation',
                    opset=14,
                    device="cpu",  # Use CPU for export
                    no_post_process=True,  # Skip post-processing to save memory
                    trust_remote_code=True
                )
        else:
            raise ValueError("Ensure there is a model and a tokenizer loaded, before attempting ONNX conversion")

    def load_model(self, device:str) -> None:
        """
        Load a specific LLM model

        Args:
            device: the device onto which the model should be loaded. For onnx it is cpu
        """

        if self.model_name is None:
            raise ValueError("No model name specified, please set attribute model_name")
        if self.bnb_config is None:
            print("No quantization mechanism is implemented. To change this set attribute bnb_config")
        if self.attn is None:
            print("attention is not specified defaulting to sdpa. To change this set attribute attn")
            
        self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map=device,
                quantization_config=None if self.bnb_config is None else self.bnb_config,
                use_sliding_window=False,
                do_sample=self.do_sampling, # but we need to look at accuracy
                attn_implementation="sdpa" if self.attn is None else self.attn
            )
        print(f"Model loaded on {device}")

    @abstractmethod
    def __call__(self, system_prompt: str, user_prompt: str) -> str:
        """
        Abstract method to process system and user prompts.

        Args:
            system_prompt (str): The system prompt.
            user_prompt (str): The user prompt.

        Returns:
            str: The processed result.
        """
        pass


class Openai(LLMModel):
    def __init__(self, api_key, api_url="https://api.deepseek.com"):
        self.client = OpenAI(api_key=api_key, base_url=api_url)

    def __call__(self, system_prompt, user_prompt) -> str:
        response = self.client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            stream=False,
        )

        return response.choices[0].message.content


class Qwen(LLMModel):
    def __init__(
        self,
        model_name: Literal[
        "Qwen/Qwen2.5-Coder-1.5B-Instruct", "Qwen/Qwen2.5-Coder-7B-Instruct", 
    ] = "Qwen/Qwen2.5-Coder-1.5B-Instruct",
        bit_config: Literal["8bit", "4bit", "none"] = "4bit",
        use_onnx: bool=False
    ):
        super().__init__()

        self.model_name = model_name
        if bit_config == "8bit":
            self.bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        elif bit_config == "4bit":
            self.bnb_config = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16
            )

        if self.check_flash_att_compatibility():
            print("Flash attention will be used as the attention mechanism")
            self.attn = "flash_attention_2"
        else:
            print("Unable to run on GPU using flash attention, will run on CPU using sdpa attention mechanism")
            self.attn = "sdpa"

        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        print(f"self.model = {self.model}")
        if use_onnx:
            # load the model directly to CPU to avoid having to move it when converting to onnx
            

            print("creating a onnx version of the model for inference")
            
            # check to see if a onnx directory exists
            if not os.path.exists(self.onnx_path) and not os.path.exists(os.path.join(self.onnx_path, "model.onnx")):
                print("here")
                self.load_model(device="cpu")
                self.export_to_onnx()
                self.model.config.save_pretrained(self.onnx_path)
                
            else:    
                # load onnx model onto device
                print(f"onnx_path = {self.onnx_path}")
                self.model = ORTModelForCausalLM.from_pretrained(
                    self.onnx_path,
                    provider=["CUDAExecutionProvider"] if self.device=='cuda' else ["CPUExecutionProvider"],
                    library_name="transformers" 
                )

        else:
            self.load_model(device=self.device)
        

    def __call__(self, system_prompt, user_prompt) -> str:
        print(user_prompt)
        messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {"role": "user", "content": user_prompt},
        ]

        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            generated_ids = self.model.generate(**model_inputs, max_new_tokens=1500)

            generated_ids = [
                output_ids[len(input_ids) :]
                for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]

            response = self.tokenizer.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0]

        return response
