import os
import random

from locust import HttpUser, between, task

API_KEY = os.getenv("API_KEY")

QUERIES = [
    "What is the rate limit for Standard tier API?",
    "How do I enable MFA on my account?",
    "What is the RTO for disaster recovery?",
    "What is the data retention for Premium tier?",
    "How do I configure webhooks?",
]

PROMPTS = [
    "Explain what RAG is in one sentence.",
    "Summarize the benefits of rate limiting.",
    "What does MFA stand for?",
]


def _headers() -> dict[str, str]:
    return {"x-api-key": API_KEY} if API_KEY else {}


class RagUser(HttpUser):
    wait_time = between(0.2, 1.0)

    @task(3)
    def query(self) -> None:
        payload = {"query": random.choice(QUERIES), "use_rag": True}
        self.client.post("/query", json=payload, headers=_headers())

    @task(1)
    def generate(self) -> None:
        payload = {"prompt": random.choice(PROMPTS), "max_new_tokens": 80}
        self.client.post("/generate", json=payload, headers=_headers())
