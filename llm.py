import importlib
import os
from abc import ABC, abstractmethod
from typing import Literal

import ollama
import torch
from dotenv import load_dotenv
from openai import OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from logger_config import getLogger

load_dotenv()

logger = getLogger(__name__)


def get_model(
    model_name: Literal["qwen", "deepseek", "ollama", "openrouter"] = "qwen"
) -> int:
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
    elif model_name == "ollama":
        return Ollama()
    elif model_name == "openrouter":
        return OpenRouter()

    raise ValueError(f"Model {model_name} not available")


class LLMModel(ABC):

    def __init__(self):
        self.model_name = None
        self.model = None
        self.tokenizer = None
        self.bnb_config = None
        self.attn = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.do_sampling = False  # use the model default as the default here
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

        if importlib.util.find_spec("flash_attn") is None:
            print(
                f"No package flash_attn available for import. Ensure it is installed and try again!"
            )
            return False

        gpu_name = torch.cuda.get_device_name()
        gpu_idx = torch.cuda.current_device()
        # tuple value representing the minor and major capability of the gpu
        gpu_capability = torch.cuda.get_device_capability(gpu_idx)

        print(f"The following GPU is available: ")
        print(f"\tname: {gpu_name}")
        print(f"\tindex: {gpu_idx}")
        print(f"\tCapability: {gpu_capability[0]}.{gpu_capability[1]}")

        if gpu_capability[0] >= 8:  # ampere=8
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
            # Make sure the model is completely moved to CPU
            # if hasattr(self, 'model'):
            #     # First unload from GPU
            # self.model = self.model.cpu()

            # # Force all parameters to CPU
            # for param in self.model.parameters():
            #     param.data = param.data.cpu()

            # # Force all buffers to CPU
            # for buffer in self.model.buffers():
            #     buffer.data = buffer.data.cpu()

            # Clear CUDA cache
            torch.cuda.empty_cache()
            with tempfile.TemporaryDirectory() as tmp_dir:

                # perform onnx conversion on CPU as this is memory intensive
                # self.model = self.model.cpu()
                self.model.save_pretrained(tmp_dir)
                self.tokenizer.save_pretrained(tmp_dir)
                main_export(
                    model_name_or_path=tmp_dir,
                    output=self.onnx_path,
                    task="text-generation",
                    opset=14,
                    device="cpu",  # Use CPU for export
                    no_post_process=True,  # Skip post-processing to save memory
                    trust_remote_code=True,
                )
        else:
            raise ValueError(
                "Ensure there is a model and a tokenizer loaded, before attempting ONNX conversion"
            )

    def load_model(self, device: str) -> None:
        """
        Load a specific LLM model

        Args:
            device: the device onto which the model should be loaded. For onnx it is cpu
        """

        if self.model_name is None:
            raise ValueError("No model name specified, please set attribute model_name")
        if self.bnb_config is None:
            print(
                "No quantization mechanism is implemented. To change this set attribute bnb_config"
            )
        if self.attn is None:
            print(
                "attention is not specified defaulting to sdpa. To change this set attribute attn"
            )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map=device,
            quantization_config=None if self.bnb_config is None else self.bnb_config,
            use_sliding_window=False,
            do_sample=self.do_sampling,  # but we need to look at accuracy
            attn_implementation="sdpa" if self.attn is None else self.attn,
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

    def check_flash_att_compatibility(self) -> bool:
        """
        Check if flash_attention can be used, otherwise default to sdpa
        Args:
        Returns:
            bool value indicating if flash_attn can be used
        """

        if not torch.cuda.is_available():
            logger.info("No GPU available, defaulting to CPU")
            return False

        if importlib.util.find_spec("flash_attn") is None:
            logger.info(
                f"No package flash_attn available for import. Ensure it is installed and try again!"
            )
            return False

        gpu_name = torch.cuda.get_device_name()
        gpu_idx = torch.cuda.current_device()
        # tuple value representing the minor and major capability of the gpu
        gpu_capability = torch.cuda.get_device_capability(gpu_idx)

        logger.info(f"The following GPU is available: ")
        logger.info(f"\tname: {gpu_name}")
        logger.info(f"\tindex: {gpu_idx}")
        logger.info(f"\tCapability: {gpu_capability[0]}.{gpu_capability[1]}")

        if gpu_capability[0] >= 8:  # ampere=8
            logger.info("gpu support flash_attn")
            return True
        else:
            logger.info("gpu does not support flash_attn")
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


class OpenRouter(LLMModel):
    def __init__(self):
        # print(os.getenv("OPENROUTER_API_KEY"))
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
        )

    def __call__(self, system_prompt, user_prompt) -> str:
        print(system_prompt)
        print(user_prompt)
        response = self.client.chat.completions.create(
            model="qwen/qwen-2.5-coder-32b-instruct",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )

        print(response)
        logger.info(response.choices[0].message.content)

        return response.choices[0].message.content


# Parameter reference
# https://github.com/ollama/ollama/blob/main/docs/modelfile.md#valid-parameters-and-values
class Ollama(LLMModel):
    def __call__(self, system_prompt, user_prompt) -> str:
        logger.info(user_prompt)
        response = ollama.chat(
            "qwen2.5-coder:32b-instruct-q4_0",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            options={
                "num_ctx": 32768,
                "num_predict": 3500,
                "temperature": 0.0,
            },
        )

        return response["message"]["content"]


class Qwen(LLMModel):
    def __init__(
        self,
        model_name: Literal[
            "Qwen/Qwen2.5-Coder-1.5B-Instruct", "Qwen/Qwen2.5-Coder-7B-Instruct"
        ] = "Qwen/Qwen2.5-Coder-1.5B-Instruct",
        bit_config: Literal["8bit", "4bit", "none"] = "4bit",
        use_onnx: bool = True,
    ):
        super().__init__()

        self.model_name = model_name
        if bit_config == "8bit":
            self.bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        elif bit_config == "4bit":
            self.bnb_config = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16
            )

        logger.info(f"Qwen|{model_name}|{bnb_config}")

        if self.check_flash_att_compatibility():
            logger.info("Flash attention will be used as the attention mechanism")
            self.attn = "flash_attention_2"
        else:
            logger.info(
                "Unable to run on GPU using flash attention, will run on CPU using sdpa attention mechanism"
            )
            self.attn = "sdpa"

        if model_name.endswith("AWQ"):
            logger.info(f"Loading AWQ {model_name}")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="cuda",
                use_sliding_window=False,
                attn_implementation=self.attn,
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="cuda",
                quantization_config=bnb_config,
                use_sliding_window=False,
                attn_implementation=self.attn,
            )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def __call__(self, system_prompt, user_prompt) -> str:
        logger.info(user_prompt)
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
