import os
import warnings
from typing import Optional, Dict, Any, List, Union

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GenerationConfig

path = os.getcwd()
model_path = os.path.join(path, "saved_models")

class SelectModel:
    def __init__(
            self,
            model_name: str,
            provider: str,
            load_in_4bit: bool = True,
            load_in_8bit: bool = False,
            set_gpu_limit: bool = False,
            max_gpu_memory: str = "8GiB",
            context_window: int = 4096):
        """
        Initialize a model selector that can work with various API providers or local models.

        Args:
            model_name: The name of the model to use
            provider: The provider of the model (openai, anthropic, together, or hugging_face)
            load_in_4bit: Whether to load the model in 4-bit precision (for local models)
            load_in_8bit: Whether to load the model in 8-bit precision (for local models)
            set_gpu_limit: Whether to set a GPU memory limit (for local models)
            max_gpu_memory: The maximum GPU memory to use (for local models)
            context_window: The maximum context window size for the model
        """
        self.provider = provider
        self.model_name = model_name
        self.full_model_name = model_name
        self.context_window = context_window
        self._setup_provider()

        if not self.local_model:
            return

        self._setup_local_model(load_in_4bit, load_in_8bit, set_gpu_limit, max_gpu_memory)
        self.clear_cache()

    def _setup_provider(self):
        """Set up the appropriate client based on the provider."""
        if self.provider == "openai":
            print(f"Using OpenAI API with model: {self.model_name}")
            from openai import OpenAI
            self.client = OpenAI()
            self.local_model = False
        elif self.provider == "anthropic":
            print(f"Using Anthropic API with model: {self.model_name}")
            import anthropic
            self.client = anthropic.Anthropic()
            self.local_model = False
        elif self.provider == "together":
            print(f"Using Together API with model: {self.model_name}")
            from together import Together
            self.client = Together()
            self.local_model = False
        elif self.provider == "hugging_face":
            print(f"Running local model: {self.model_name}")
            self.local_model = True
            self._access_token = os.getenv("HUGGING_FACE_HUB_TOKEN")
            os.makedirs(model_path, exist_ok=True)
            os.environ['TRANSFORMERS_CACHE'] = model_path
            self.model_family = self.model_name.split("/")[0]
            self.model_name = self.model_name.split("/")[1]
        else:
            raise ValueError(f"Invalid provider: {self.provider}")

    def _setup_local_model(self, load_in_4bit, load_in_8bit, set_gpu_limit, max_gpu_memory):
        """Set up a local Hugging Face model."""
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.full_model_name,
            token=self._access_token,
            trust_remote_code=True,
            padding_side="left",
            model_max_length=self.context_window,
            cache_dir=model_path
        )
        if not self._tokenizer.pad_token:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        if not self._tokenizer.bos_token:
            self._tokenizer.bos_token = self._tokenizer.eos_token
        quantization_config = None
        if load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        elif load_in_8bit:
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True
            )
        kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
            "low_cpu_mem_usage": True,
            "cache_dir": model_path
        }

        if quantization_config:
            kwargs["quantization_config"] = quantization_config
            quantization_config.llm_int8_enable_fp32_cpu_offload = True
        if torch.cuda.is_available():
            kwargs["device_map"] = "auto"
            if set_gpu_limit:
                kwargs["max_memory"] = {0: max_gpu_memory}
            self._device = "cuda"
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        else:
            self._device = "cpu"
            if quantization_config:
                quantization_config.llm_int8_enable_fp32_cpu_offload = True
            kwargs.update({
                "device_map": "balanced_low_0",
                "llm_int8_enable_fp32_cpu_offload": True
            })
            warnings.warn("Running on CPU. This will be significantly slower.")

        # Load the model
        self._model = AutoModelForCausalLM.from_pretrained(
            self.full_model_name,
            **kwargs
        )
        self._model.eval()

        # Set up default generation parameters
        self._default_generation_config = {
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 50,
            "num_beams": 1,
            "repetition_penalty": 1.1,
            "do_sample": True,
            "pad_token_id": self._tokenizer.pad_token_id,
            "bos_token_id": self._tokenizer.bos_token_id,
            "eos_token_id": self._tokenizer.eos_token_id,
            "max_new_tokens": 512,
            "use_cache": True
        }

    def generate_text(
            self,
            prompt: str,
            system_prompt: Optional[str] = None,
            max_new_tokens: int = 512,
            temperature: float = 0.7,
            top_p: float = 0.9,
            top_k: int = 50,
            repetition_penalty: float = 1.1,
            **kwargs: Dict[str, Any]
    ) -> str:
        """
        Generate text from the model based on the given prompt.

        Args:
            prompt: The text prompt to generate from
            system_prompt: An optional system prompt for API-based models
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Controls randomness (higher = more random)
            top_p: Nucleus sampling parameter (1.0 = no nucleus sampling)
            top_k: Top-k sampling parameter (0 = no top-k sampling)
            repetition_penalty: Penalty for repeating tokens (1.0 = no penalty)
            **kwargs: Additional parameters to pass to the generation config

        Returns:
            The generated text as a string
        """
        if self.local_model:
            return self._generate_local(prompt, system_prompt, max_new_tokens, temperature,
                                        top_p, top_k, repetition_penalty, **kwargs)
        else:
            return self._generate_api(prompt, system_prompt, max_new_tokens, temperature,
                                      top_p, top_k, repetition_penalty, **kwargs)

    def _generate_api(self, prompt, system_prompt, max_new_tokens, temperature, top_p, top_k, repetition_penalty,
                      **kwargs):
        """Generate text using API-based models."""
        if system_prompt:
            messages = [
                {"role": "system", "content": system_prompt.strip()},
                {"role": "user", "content": prompt.strip()}
            ]
        else:
            messages = [{"role": "user", "content": prompt.strip()}]

        try:
            if  self.provider == "openai":
                completion = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_new_tokens,
                    frequency_penalty=repetition_penalty - 1.0 if repetition_penalty > 1.0 else 0,
                    presence_penalty=0.2 if repetition_penalty > 1.0 else 0,
                    store=False
                )
                return completion.choices[0].message.content

            elif self.provider == "anthropic":
                completion = self.client.messages.create(
                    model=self.model_name,
                    messages=messages,
                    max_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p
                )
                return completion.content[0].text

            elif self.provider == "together":
                completion = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    max_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    repetition_penalty=repetition_penalty
                )
                return completion.choices[0].message.content

            else:
                raise ValueError(f"Unknown provider: {self.provider}")

        except Exception as e:
            print(f"API request failed: {str(e)}")
            return f"Error generating response: {str(e)}"

    def _generate_local(self, prompt, system_prompt, max_new_tokens, temperature, top_p, top_k, repetition_penalty,
                        **kwargs):
        """Generate text using a local Hugging Face model."""
        if system_prompt:
            full_prompt = f"{system_prompt.strip()}\n\n{prompt.strip()}"
        else:
            full_prompt = prompt.strip()

        max_input_length = self.context_window - max_new_tokens - 10

        # Tokenize input
        inputs = self._tokenizer(
            full_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_input_length,
            padding=True
        )
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

        # Set up generation config
        current_config = self._default_generation_config.copy()
        current_config.update({
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "repetition_penalty": repetition_penalty,
            **kwargs
        })
        generation_config = GenerationConfig(**current_config)

        try:
            with torch.inference_mode(), torch.amp.autocast(self._device, dtype=torch.float16):
                outputs = self._model.generate(
                    **inputs,
                    generation_config=generation_config
                )
            generated_text = self._tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            return generated_text

        except Exception as e:
            print(f"Local generation failed: {str(e)}")
            return f"Error generating response: {str(e)}"

    def clear_cache(self):
        """Clear CUDA cache to free up memory"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def unload_model(self):
        """Unload the model from memory"""
        if self.local_model and hasattr(self, '_model'):
            del self._model
            del self._tokenizer
            self.clear_cache()
            print("Model unloaded from memory")