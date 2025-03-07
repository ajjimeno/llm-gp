import os
from abc import ABC, abstractmethod
from typing import Literal

import torch
from dotenv import load_dotenv
import importlib
from openai import OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from logger_config import getLogger

load_dotenv()

logger = getLogger(__name__)


def get_model(model_name: Literal["qwen", "deepseek"] = "qwen") -> int:
    if model_name == "qwen":
        return (
            Qwen(os.getenv("LLM_MODEL_NAME"))
            if os.getenv("LLM_MODEL_NAME", None)
            else Qwen()
        )
    elif model_name == "deepseek":
        return Openai(
            api_key=os.getenv("LLM_API_KEY"), api_url=os.getenv("LLM_API_URL")
        )

    raise ValueError(f"Model {model_name} not available")


class LLMModel(ABC):
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
            "Qwen/Qwen2.5-Coder-1.5B-Instruct", "Qwen/Qwen2.5-Coder-7B-Instruct"
        ] = "Qwen/Qwen2.5-Coder-1.5B-Instruct",
        bit_config: Literal["8bit", "4bit", "none"] = "4bit",
    ):
        if bit_config == "8bit":
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        elif bit_config == "4bit":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16
            )

        logger.info(f"Qwen|{model_name}|{bnb_config}")

        if self.check_flash_att_compatibility():
            logger.info("Flash attention will be used as the attention mechanism")
            self.attn = "flash_attention_2"
        else:
            logger.info("Unable to run on GPU using flash attention, will run on CPU using sdpa attention mechanism")
            self.attn = "sdpa"

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="cuda",
            quantization_config=bnb_config,
            use_sliding_window=False,
            attn_implementation=self.attn
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

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
