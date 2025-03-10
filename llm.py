import os
from abc import ABC, abstractmethod
from typing import Literal

import torch
from dotenv import load_dotenv
from openai import OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import importlib.util

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

    def check_flash_att_compatibility(self) -> bool:
        """
        Check if flash_attention can be used, otherwise default to sdpa
        Args:
            None
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
            return True
        else:
            return False

    def load_model(self, device:str, vllm: bool) -> None:
        """
        Load a specific LLM model

        Args:
            device: the device onto which the model should be loaded. For onnx it is cpu
            vllm: use of vllm as inference mechanism
        """

        if self.model_name is None:
            raise ValueError("No model name specified, please set attribute model_name")
        if self.attn is None:
            print("attention is not specified. To change this set attribute attn")

        if not vllm:
            if self.bnb_config is None:
                print("No quantization mechanism is implemented. To change this set attribute bnb_config")
         
                
            self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16,
                    device_map=device,
                    quantization_config=None if self.bnb_config is None else self.bnb_config,
                    use_sliding_window=False,
                    do_sample=self.do_sampling, # but we need to look at accuracy
                    attn_implementation="sdpa" if self.attn is None else self.attn
                )
            
        else:
            print(f"using vllm as inference mechanism")
            # no parameter to pass to vLLM and thus set it in the environment
            if self.attn != "flash_attention2":
                os.environ["VLLM_ATTENTION_BACKEND"] = "TORCH_SDPA"
            from vllm import LLM as vLLM, SamplingParams
            self.model = vLLM(model=self.model_name, 
                              tensor_parallel_size=1,
                              device=self.device,
                              quantization="awq", # Using AWQ (aactivation aware weight quantization) 4-bit quantization
                              dtype="half",
                              trust_remote_code=True)
            self.vllm_sampling = SamplingParams(
                temperature=0.7 if hasattr(self, 'do_sampling') and self.do_sampling else 0.0,
                top_p=0.95 if hasattr(self, 'do_sampling') and self.do_sampling else 1.0,
                max_tokens=1500
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
        vllm: bool=True
    ):
        super().__init__()
        self.vllm = vllm
        self.model_name = model_name
        if bit_config == "8bit":
            self.bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        elif bit_config == "4bit":
            self.bnb_config = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16
            )
        if not self.vllm:
            if self.check_flash_att_compatibility():
                print("Flash attention will be used as the attention mechanism")
                self.attn = "flash_attention_2"
            else:
                print(f"Unable to run on GPU using flash attention, will run on {self.device} using sdpa attention mechanism")
                self.attn = "sdpa"

        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.load_model(device=self.device, vllm=self.vllm)
        

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
        if not self.vllm:
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
        else:
            outputs = self.model.generate(text, self.vllm_sampling)
            response = outputs[0].outputs[0].text
        return response
