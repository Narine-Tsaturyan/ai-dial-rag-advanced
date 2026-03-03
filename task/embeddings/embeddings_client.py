import json

import requests

DIAL_EMBEDDINGS = 'https://ai-proxy.lab.epam.com/openai/deployments/{model}/embeddings'

class DialEmbeddingsClient:
    def __init__(self, deployment: str, api_key: str):
        self.deployment = deployment
        self.api_key = api_key
        self.url = DIAL_EMBEDDINGS.format(model=self.deployment)
        self.headers = {
            "Content-Type": "application/json",
            "api-key": self.api_key
        }

    def get_embeddings(self, texts):
        payload = {
            "input": texts
        }
        response = requests.post(self.url, headers=self.headers, data=json.dumps(payload))
        response.raise_for_status()
        data = response.json()
        embeddings = {}
        for item in data.get("data", []):
            idx = item["index"]
            emb = item["embedding"]
            embeddings[idx] = emb
        return embeddings