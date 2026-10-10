"""Ollama client wrapper for local LLM inference and service status inspection."""
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import requests

from src.config import OLLAMA_BASE_URL, OLLAMA_MODEL, OLLAMA_TIMEOUT


@dataclass
class OllamaStatus:
    """Status details of Ollama service and configured model."""
    is_running: bool
    installed_models: List[str]
    configured_model: str
    is_configured_model_available: bool
    status_message: str
    recommended_commands: List[str]


class OllamaClient:
    """Interacts with local Ollama REST API with diagnostic helpers."""

    def __init__(
        self,
        base_url: str = OLLAMA_BASE_URL,
        model_name: str = OLLAMA_MODEL,
        timeout: int = OLLAMA_TIMEOUT
    ):
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.timeout = timeout

    def check_status(self) -> OllamaStatus:
        """Inspect whether Ollama daemon is reachable and if target model exists."""
        tags_url = f"{self.base_url}/api/tags"
        try:
            resp = requests.get(tags_url, timeout=3)
            if resp.status_code != 200:
                return OllamaStatus(
                    is_running=False,
                    installed_models=[],
                    configured_model=self.model_name,
                    is_configured_model_available=False,
                    status_message=f"Ollama returned HTTP status {resp.status_code}.",
                    recommended_commands=[
                        "Ensure Ollama daemon is running: 'ollama serve'"
                    ]
                )

            data = resp.json()
            models_list = [m.get("name", "") for m in data.get("models", []) if m.get("name")]
            # Strip tags for matching (e.g. 'llama3.2:latest' vs 'llama3.2')
            model_names_normalized = [m.split(":")[0] for m in models_list]

            is_available = (
                self.model_name in models_list or
                f"{self.model_name}:latest" in models_list or
                self.model_name in model_names_normalized
            )

            if is_available:
                status_msg = f"Ollama is running. Model '{self.model_name}' is ready locally."
                recommended = []
            else:
                if models_list:
                    status_msg = (
                        f"Ollama is running, but configured model '{self.model_name}' is not found. "
                        f"Locally installed models: {', '.join(models_list)}."
                    )
                else:
                    status_msg = (
                        f"Ollama is running, but no models are currently installed. "
                        f"Please pull model '{self.model_name}'."
                    )
                recommended = [
                    f"ollama pull {self.model_name}",
                    "ollama list",
                    f"ollama run {self.model_name}"
                ]

            return OllamaStatus(
                is_running=True,
                installed_models=models_list,
                configured_model=self.model_name,
                is_configured_model_available=is_available,
                status_message=status_msg,
                recommended_commands=recommended
            )

        except requests.exceptions.ConnectionError:
            return OllamaStatus(
                is_running=False,
                installed_models=[],
                configured_model=self.model_name,
                is_configured_model_available=False,
                status_message=(
                    f"Could not connect to Ollama at {self.base_url} (Connection refused). "
                    "Make sure the Ollama application or service is started."
                ),
                recommended_commands=[
                    "ollama serve",
                    f"ollama pull {self.model_name}",
                    "ollama list"
                ]
            )
        except Exception as e:
            return OllamaStatus(
                is_running=False,
                installed_models=[],
                configured_model=self.model_name,
                is_configured_model_available=False,
                status_message=f"Error checking Ollama status: {str(e)}",
                recommended_commands=[
                    "ollama list"
                ]
            )

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model_override: Optional[str] = None
    ) -> Tuple[bool, str]:
        """
        Send a generation request to Ollama /api/generate.
        Returns (success: bool, response_or_error_text: str).
        """
        model = model_override or self.model_name
        payload: Dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.2,  # Low temperature for strict factual grounding
                "top_p": 0.9
            }
        }
        if system_prompt:
            payload["system"] = system_prompt

        url = f"{self.base_url}/api/generate"
        try:
            resp = requests.post(url, json=payload, timeout=self.timeout)
            if resp.status_code == 200:
                result = resp.json()
                return True, result.get("response", "").strip()
            elif resp.status_code == 404:
                return False, (
                    f"Model '{model}' not found in Ollama (HTTP 404). "
                    f"To install this model, run in terminal: `ollama pull {model}`"
                )
            else:
                return False, f"Ollama API returned error HTTP {resp.status_code}: {resp.text}"
        except requests.exceptions.ConnectionError:
            return False, (
                f"Connection refused to Ollama at {self.base_url}. "
                "Ensure Ollama is started locally (`ollama serve`)."
            )
        except requests.exceptions.Timeout:
            return False, (
                f"Ollama request timed out after {self.timeout} seconds. "
                "The local model may still be loading into memory or the computer is under heavy load."
            )
        except Exception as e:
            return False, f"Unexpected error communicating with Ollama: {str(e)}"
