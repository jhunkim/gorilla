import json
import os
import time
from typing import Any

from bfcl_eval.model_handler.api_inference.openai_completion import OpenAICompletionsHandler

class GaussO4Handler(OpenAICompletionsHandler):
    def __init__(
        self,
        model_name,
        temperature,
        registry_name,
        is_fc_model,
        **kwargs,
    ) -> None:
        super().__init__(model_name, temperature, registry_name, is_fc_model, **kwargs)

    def _build_client_kwargs(self):
        kwargs = super()._build_client_kwargs()
        
        # Add custom base URL and headers for Gauss O4
        if base_url := os.getenv("GAUSS_BASE_URL"):
            kwargs["base_url"] = base_url
            
        custom_headers = {}
        if auth_key := os.getenv("GAUSS_AUTH_KEY_B64"):
            custom_headers["X-Custom-Auth"] = auth_key
            
        if "default_headers" in kwargs:
            kwargs["default_headers"].update(custom_headers)
        else:
            kwargs["default_headers"] = custom_headers
            
        return kwargs

    def generate_with_backoff(self, **kwargs):
        # Add intentional delay to respect 10 RPM limit
        time.sleep(6) # 60s / 10 = 6s
        return super().generate_with_backoff(**kwargs)
