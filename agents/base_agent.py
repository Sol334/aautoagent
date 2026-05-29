"""Shared Ollama inference base for all Galactic Capital agents."""

import re
import httpx


class BaseAgent:
    def __init__(self, model: str, host: str):
        self.model = model
        host = host.replace("0.0.0.0", "localhost")
        self.host = host if host.startswith("http") else f"http://{host}"

    def generate(self, prompt: str, timeout: int = 90) -> str:
        """Send prompt to Ollama, return response text. Raises on failure."""
        with httpx.Client(timeout=timeout) as client:
            resp = client.post(
                f"{self.host}/api/generate",
                json={"model": self.model, "prompt": prompt, "stream": False},
            )
            resp.raise_for_status()
            raw = resp.json().get("response", "")
        # Strip DeepSeek R1 chain-of-thought tags before returning
        return re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()

    def analyze(self, data: object) -> object:
        raise NotImplementedError
