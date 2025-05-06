import openai
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GenerationConfig
from typing import Optional, Dict, Any
import warnings
import os
path = os.getcwd()
model_path = os.path.join(path, r"saved_models")

class SelectModel:
    def __init__(
            self,
            model_name: str,
            provider: str,
            load_in_4bit: bool = True,
            set_gpu_limit: bool = False,
            max_gpu_memory: str = "8GiB"):
        self.provider = provider.split("/")[0]
        self.model_name = model_name
        self.full_model_name = model_name
        if self.provider == "openai":
            print("Using openai API")
            from openai import OpenAI
            self.client = OpenAI()
            self.local_model = False
        elif self.provider == "anthropic":
            print("Using anthropic API")
            import anthropic
            self.client = anthropic.Anthropic()
            self.local_model = False
        elif self.provider == "together":
            print("Using together API ")
            from together import Together
            self.client = Together()
            self.local_model = False
        elif self.provider == "hugging_face":
            print("Running local model")
            self.local_model = True
            self._access_token = os.getenv("HUGGING_FACE_HUB_TOKEN")
            os.makedirs(model_path, exist_ok=True)
            os.environ['TRANSFORMERS_CACHE'] = model_path
            self.model_family = provider.split("/")[1]
            self.full_model_name = self.model_family + "/" + self.model_name
        else:
            print("Not a valid provider")

        if not self.local_model:
            return

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.full_model_name,
            token=self._access_token,
            trust_remote_code=True,
            padding_side="left",
            model_max_length=4096,
            cache_dir=model_path
        )
        if not self._tokenizer.pad_token:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        if not self._tokenizer.bos_token:
            self._tokenizer.bos_token = self._tokenizer.eos_token
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=load_in_4bit,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4")
        kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
            "quantization_config": quantization_config,
            "low_cpu_mem_usage": True}
        if torch.cuda.is_available():
            kwargs.update({"device_map": "auto"})
            if set_gpu_limit:
                kwargs.update({"max_memory": {0: max_gpu_memory}})
            self._device = "cuda"
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        else:
            self._device = "cpu"
            kwargs.update({
                "device_map": "balanced_low_0",
                "llm_int8_enable_fp32_cpu_offload": True})
            warnings.warn("Running on CPU. This will be significantly slower.")
        self._model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=model_path,
            **kwargs)
        self._model.eval()
        self._default_generation_config = {
            "temperature": 0.6,
            "top_p": 0.7,
            "top_k": 50,
            "num_beams": 1,
            "repetition_penalty": 1,
            "do_sample": True,
            "pad_token_id": self._tokenizer.pad_token_id,
            "bos_token_id": self._tokenizer.bos_token_id,
            "eos_token_id": self._tokenizer.eos_token_id,
            "max_new_tokens": 512,
            "use_cache": True}
        self.clear_cache()

    def generate_text(
            self,
            prompt: str,
            system_prompt: Optional[str] = None,
            max_new_tokens: int = 512,
            temperature: float = 0.6,
            top_p: float = 0.7,
            top_k: int = 50,
            repetition_penalty: float = 1,
            **kwargs: Dict[str, Any]
    ) -> str:
        if system_prompt:
            full_prompt = f"{system_prompt.strip()}\n\n{prompt.strip()}\n\n"
        else:
            full_prompt = prompt.strip() + "\n\n"
        full_prompt += f"You have {str(max_new_tokens)} tokens for your response.\n\n"
        full_prompt += "Response:\n"

        if not self.local_model:
            messages =  [{"role": "system", "content": full_prompt}]
            match self.provider:
                case "openai":
                    try:
                        completion = self.client.chat.completions.create( model=self.model_name,
                                                                          messages=messages,
                                                                          # temperature=temperature,
                                                                          # top_p=top_p,
                                                                          store=False,
                                                                          max_completion_tokens=max_new_tokens)
                        return completion.choices[0].message.content
                    except openai.BadRequestError:
                        return ""
                case "anthropic":
                    completion = self.client.messages.create(model=self.model_name,
                                                             messages=messages,
                                                             max_tokens=max_new_tokens,
                                                             temperature=temperature)
                    return completion.content.text
                case "together":
                    completion = self.client.chat.completions.create(model=self.model_name,
                                                                     messages=messages,
                                                                     max_tokens=max_new_tokens,
                                                                     temperature=temperature,
                                                                     top_p=top_p,
                                                                     repetition_penalty=repetition_penalty)
                    return completion.choices[0].message.content
                case _:
                    raise ValueError("could not find provider for non local model")

        inputs = self._tokenizer(
            full_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self._tokenizer.model_max_length - max_new_tokens,
            padding=True)
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
        current_config = self._default_generation_config.copy()
        current_config.update({
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "repetition_penalty": repetition_penalty,
            **kwargs})
        generation_config = GenerationConfig(**current_config)
        try:
            with torch.inference_mode(), torch.amp.autocast(self._device, dtype=torch.float16):
                outputs = self._model.generate(
                    **inputs,
                    generation_config=generation_config)
            generated_text = self._tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True)
            print(generated_text)
            return generated_text

        except Exception as e:
            raise RuntimeError(f"Text generation failed: {str(e)}")

    def clear_cache(self):
        """Clear CUDA cache to free up memory"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
