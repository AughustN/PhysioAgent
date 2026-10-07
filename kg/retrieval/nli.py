#!/usr/bin/env python
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

DEFAULT_MODEL = "pritamdeka/PubMedBERT-MNLI-MedNLI"
MAX_LENGTH = 128
BATCH = 128

RISK_HYPOTHESIS = ("This worsens fluid overload, reduces the response to a loop diuretic, "
                   "or increases the diuretic dose the patient needs.")
PROTECTIVE_HYPOTHESIS = ("This improves diuresis, preserves kidney function, or reduces "
                         "the diuretic dose the patient needs.")


@dataclass
class NLIRefiner:
    model_name: str = DEFAULT_MODEL
    device: Optional[str] = None
    batch: int = BATCH
    _tokenizer: Any = field(default=None, init=False, repr=False)
    _model: Any = field(default=None, init=False, repr=False)
    _labels: dict[str, int] = field(default_factory=dict, init=False, repr=False)

    def load(self) -> "NLIRefiner":
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        model.eval().to(self.device)
        self._model = model
        self._labels = {name.lower(): index for index, name in model.config.id2label.items()}
        for needed in ("entailment", "contradiction"):
            if needed not in self._labels:
                raise RuntimeError(
                    f"{self.model_name} has labels {model.config.id2label}, with no "
                    f"'{needed}' among them; this is not a 3-way NLI checkpoint.")
        return self

    def probabilities(self, premises: Sequence[str], hypothesis: str,
                      max_length: int = MAX_LENGTH, truncation: Any = True
                      ) -> list[tuple[float, float]]:
        import torch

        if self._model is None:
            self.load()
        entail_id = self._labels["entailment"]
        contra_id = self._labels["contradiction"]
        out: list[tuple[float, float]] = []
        for start in range(0, len(premises), self.batch):
            chunk = list(premises[start:start + self.batch])
            encoded = self._tokenizer(chunk, [hypothesis] * len(chunk),
                                      return_tensors="pt", padding=True,
                                      truncation=truncation, max_length=max_length)
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            with torch.no_grad():
                probabilities = self._model(**encoded).logits.softmax(-1)
            out += [(float(row[entail_id]), float(row[contra_id]))
                    for row in probabilities]
        return out

    def refine(self, paths: Sequence[Any]) -> Sequence[Any]:
        if not paths:
            return paths
        if self._model is None:
            self.load()
        premises = [path.text() for path in paths]
        risk = self.probabilities(premises, RISK_HYPOTHESIS)
        protective = self.probabilities(premises, PROTECTIVE_HYPOTHESIS)
        for path, (risk_entail, _), (prot_entail, _) in zip(paths, risk, protective):
            path.risk = risk_entail
            path.protective = prot_entail
        return paths
