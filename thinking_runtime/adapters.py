"""Optional real model integrations. No mock answers or automatic model substitution."""
from __future__ import annotations

import json
import math
import os
import time
import urllib.request
from pathlib import Path


def expression_from_text(text):
    text = text.strip()
    if text.startswith("```"):
        text = "\n".join(text.splitlines()[1:-1]).strip()
    try:
        value = json.loads(text)
        return value["expression"] if isinstance(value, dict) and isinstance(value.get("expression"), str) else ""
    except json.JSONDecodeError:
        return text  # The independent verifier still rejects prose or executable code.


def prompt_for(task, feedback):
    return ("Solve this exact arithmetic puzzle. Output only a JSON object with one key, expression. "
            "Do not include explanations. " + json.dumps(task) +
            "\nPrevious attempts and verifier feedback: " + json.dumps(feedback[-8:]))


class HTTPGenerator:
    """OpenAI-compatible chat endpoint, for local Gemma OR a real 120B baseline."""
    def __init__(self, base_url, model, max_tokens=256, seed=None, llama_no_thinking=False):
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model = model
        self.max_tokens = max_tokens
        self.seed = seed
        self.llama_no_thinking = llama_no_thinking

    def generate(self, task, feedback, timeout):
        payload = {"model": self.model, "messages": [{"role": "user", "content": prompt_for(task, feedback)}],
                   "max_tokens": self.max_tokens, "temperature": 0.7}
        if self.seed is not None:
            payload["seed"] = self.seed
        if self.llama_no_thinking:
            payload["chat_template_kwargs"] = {"enable_thinking": False}
            payload["cache_prompt"] = False
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "arithmetic_solution", "strict": True,
                    "schema": {"type": "object", "properties": {
                        "expression": {"type": "string", "pattern": "^[0-9+*/() \\-]{1,256}$"}},
                        "required": ["expression"], "additionalProperties": False}}}
        headers = {"Content-Type": "application/json"}
        if os.getenv("MODEL_API_KEY"):
            headers["Authorization"] = "Bearer " + os.environ["MODEL_API_KEY"]
        request = urllib.request.Request(self.url, json.dumps(payload).encode(), headers)
        start = time.perf_counter()
        with urllib.request.urlopen(request, timeout=max(0.01, timeout)) as response:
            body = response.read(2_000_001)
        if len(body) > 2_000_000:
            raise ValueError("Model response exceeds 2 MB")
        data = json.loads(body)
        text = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        return {"expression": expression_from_text(text), "seconds": time.perf_counter() - start,
                "input_tokens": usage.get("prompt_tokens"), "output_tokens": usage.get("completion_tokens")}


class GemmaGenerator:
    def __init__(self, path, max_tokens=128, device="cpu"):
        import torch
        from transformers import AutoProcessor, AutoModelForMultimodalLM
        if not Path(path).is_dir():
            raise FileNotFoundError(f"Download Gemma first: {path}")
        self.torch = torch
        self.max_tokens = max_tokens
        self.processor = AutoProcessor.from_pretrained(str(path), local_files_only=True)
        dtype = torch.bfloat16 if device.startswith("cuda") and torch.cuda.is_bf16_supported() else torch.float32
        self.model = AutoModelForMultimodalLM.from_pretrained(
            str(path), local_files_only=True, torch_dtype=dtype).to(device).eval()

    def generate(self, task, feedback, timeout):
        start = time.perf_counter()
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt_for(task, feedback)}]}]
        inputs = self.processor.apply_chat_template(messages, add_generation_prompt=True,
                    tokenize=True, return_dict=True, return_tensors="pt", enable_thinking=False)
        inputs = inputs.to(self.model.device)
        with self.torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=self.max_tokens,
                       do_sample=True, temperature=0.7, max_time=max(0.01, timeout))
        count = inputs["input_ids"].shape[-1]
        generated = output[0, count:]
        return {"expression": expression_from_text(self.processor.decode(generated, skip_special_tokens=True)),
                "seconds": time.perf_counter() - start, "input_tokens": count,
                "output_tokens": len(generated)}


class LayaRanker:
    def __init__(self, path, device="cpu"):
        os.environ.setdefault("USE_TF", "0")
        import laya
        if not Path(path).is_dir():
            raise FileNotFoundError(f"Download Laya first: {path}")
        self.agent = laya.load(str(path), device=device)

    def rank(self, task, candidates):
        # Short explicit choices respect the English model's small context window.
        criteria = {f"c{i}": "; ".join(c) for i, c in enumerate(candidates)}
        questions = {"next": {"type": "choice", "instructions": "Choose the remaining arithmetic state most likely to reach the target by combining terms.",
                              "criteria": criteria}}
        result = self.agent.predict(json.dumps(task), questions)
        answer = result["answers"]["next"]
        choice = answer["choice"]
        if choice not in criteria:
            raise ValueError("Laya returned a choice outside the supplied set")
        selected = list(criteria).index(choice)
        # A typed answer orders search, never marks a candidate verified.
        return [selected] + [i for i in range(len(candidates)) if i != selected]


class EmbeddingMemory:
    def __init__(self, path, device="cpu"):
        import torch
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(str(path), device=device, local_files_only=True,
            config_kwargs={"vision_config": None, "audio_config": None},
            model_kwargs={"torch_dtype": torch.float32})

    def retrieve(self, query, documents, top_k=3):
        if not documents: return []
        q = self.model.encode([query], prompt_name="SearchQuery", normalize_embeddings=True)
        d = self.model.encode(documents, prompt_name="Document", normalize_embeddings=True)
        scores = (d @ q[0]).tolist()
        if not all(math.isfinite(x) for x in scores):
            raise ValueError("Non-finite embedding score")
        return [{"text": documents[i], "score": scores[i]} for i in
                sorted(range(len(documents)), key=lambda i: scores[i], reverse=True)[:top_k]]


class TimesFMForecaster:
    def __init__(self, path):
        import timesfm
        self.model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(str(path), torch_compile=False)
        self.model.compile(timesfm.ForecastConfig(max_context=256, max_horizon=32,
                           normalize_inputs=True, use_continuous_quantile_head=True,
                           force_flip_invariance=True, infer_is_positive=True,
                           fix_quantile_crossing=True))

    def forecast(self, history, horizon):
        import numpy as np
        if not history or not all(math.isfinite(float(x)) for x in history) or not 1 <= horizon <= 32:
            raise ValueError("Finite history and horizon 1..32 required")
        point, quantile = self.model.forecast(horizon=horizon, inputs=[np.asarray(history[-256:], dtype=np.float32)])
        result = {"point": point.tolist(), "quantiles": quantile.tolist()}
        if not np.isfinite(point).all() or not np.isfinite(quantile).all():
            raise ValueError("Non-finite forecast")
        return result
