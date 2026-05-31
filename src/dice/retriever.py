"""Demo-pool retrieval for dynamic in-context example selection."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class RetrievedDemo:
    document: dict
    distance: float


class DemoRetriever:
    """Embed high-level demo knowledge and retrieve nearest trajectories."""

    def __init__(
        self,
        documents: list[dict],
        encode_field: str = "question_knowledge",
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        normalize: bool = False,
    ):
        if not documents:
            raise ValueError("DemoRetriever requires at least one document.")
        if any(encode_field not in doc for doc in documents):
            raise KeyError(f"Every demo document must include the field {encode_field!r}.")

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError("Install sentence-transformers to use DemoRetriever.") from exc

        self.documents = documents
        self.encode_field = encode_field
        self.normalize = normalize
        self.model = SentenceTransformer(model_name)
        self.embeddings = self._encode([doc[encode_field] for doc in documents])

    @classmethod
    def from_json(
        cls,
        path: Path,
        encode_field: str = "question_knowledge",
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        normalize: bool = False,
    ) -> "DemoRetriever":
        with path.open() as f:
            documents = json.load(f)
        return cls(documents, encode_field=encode_field, model_name=model_name, normalize=normalize)

    def _encode(self, texts: list[str]) -> np.ndarray:
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        embeddings = np.asarray(embeddings, dtype=np.float32)
        if self.normalize:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / np.maximum(norms, 1e-12)
        return embeddings

    def retrieve(self, query: str, top_k: int = 6) -> list[RetrievedDemo]:
        query_embedding = self._encode([query])
        distances = np.sum((self.embeddings - query_embedding) ** 2, axis=1)
        top_indices = np.argsort(distances)[:top_k]
        return [
            RetrievedDemo(document=self.documents[int(idx)], distance=float(distances[int(idx)]))
            for idx in top_indices
        ]


def format_demo_trajectory(document: dict, field: str = "traj_full", drop_final_line: bool = True) -> str:
    """Format a retrieved demo trajectory for insertion into the ReAct prompt."""

    trajectory = document[field].strip("\n")
    lines = trajectory.split("\n")
    if drop_final_line and len(lines) > 1:
        lines = lines[:-1]
    return "\n".join(lines).strip() + "\n"

