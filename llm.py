from abc import ABC, abstractmethod
from typing import Literal

import torch

# from openai import OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def get_model(model_name: Literal["qwen", "deepseek"] = "qwen") -> int:
    if model_name == "qwen":
        return Qwen()
    elif model_name == "deepseek":
        pass

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
            "Qwen/Qwen2.5-Coder-7B-Instruct"
        ] = "Qwen/Qwen2.5-Coder-7B-Instruct",
        bit_config: Literal["8bit", "4bit", "none"] = "4bit",
    ):
        if bit_config == "8bit":
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        elif bit_config == "4bit":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16
            )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="cuda",
            quantization_config=bnb_config,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def __call__(self, system_prompt, user_prompt) -> str:
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
