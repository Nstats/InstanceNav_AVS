"""Compatibility interface for graphs, filters and landmark selection."""
from concurrent.futures import ThreadPoolExecutor
from utils.model_client import get_model_client


class FlaskFunc:
    def __init__(self, server_port=None):
        self.model_client = get_model_client(server_port)

    def get_llm_response(self, prompt):
        return self.model_client.complete(prompt)

    def get_vlm_response(self, prompt, image):
        return self.model_client.complete(prompt, image)

    def get_vlm_responses(self, requests):
        """Independent image requests; preserve order and propagate failures."""
        if not requests:
            return []
        with ThreadPoolExecutor(max_workers=self.model_client.concurrency) as pool:
            return list(pool.map(lambda item: self.get_vlm_response(*item), requests))

    def get_llm_responses(self, prompts, max_workers=None):
        if not prompts:
            return []
        # Default stays the shared API concurrency. Landmark ranking can pass
        # its own worker count without changing other callers.
        workers = self.model_client.concurrency if not max_workers else max_workers
        with ThreadPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(self.get_llm_response, prompts))
