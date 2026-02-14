"""RAG-based supply chain copilot with embeddings and optional LLM generation."""

from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class RetrievalResult:
    rank: int
    score: float
    source: str
    text: str
    metadata: dict[str, Any]


class EmbeddingIndex:
    """Simple vector index based on TF-IDF embeddings."""

    def __init__(self, vectorizer: TfidfVectorizer, matrix: Any, documents: list[dict[str, Any]]) -> None:
        self.vectorizer = vectorizer
        self.matrix = matrix
        self.documents = documents

    @classmethod
    def from_documents(cls, documents: list[dict[str, Any]]) -> "EmbeddingIndex":
        if not documents:
            raise ValueError("Cannot build embedding index with no documents.")
        texts = [doc["text"] for doc in documents]
        vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=12000)
        matrix = vectorizer.fit_transform(texts)
        return cls(vectorizer=vectorizer, matrix=matrix, documents=documents)

    def query(self, question: str, top_k: int = 4) -> list[RetrievalResult]:
        if not question.strip():
            return []
        q_vec = self.vectorizer.transform([question])
        sims = cosine_similarity(q_vec, self.matrix).flatten()
        top_idx = np.argsort(sims)[::-1][:top_k]

        results: list[RetrievalResult] = []
        for rank, idx in enumerate(top_idx, start=1):
            doc = self.documents[int(idx)]
            results.append(
                RetrievalResult(
                    rank=rank,
                    score=float(sims[int(idx)]),
                    source=str(doc.get("source", "unknown")),
                    text=str(doc.get("text", "")),
                    metadata=dict(doc.get("metadata", {})),
                )
            )
        return results

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("wb") as handle:
            pickle.dump({"vectorizer": self.vectorizer, "matrix": self.matrix, "documents": self.documents}, handle)

    @classmethod
    def load(cls, path: str | Path) -> "EmbeddingIndex":
        with Path(path).open("rb") as handle:
            obj = pickle.load(handle)
        return cls(vectorizer=obj["vectorizer"], matrix=obj["matrix"], documents=obj["documents"])


class SupplyChainCopilot:
    """RAG orchestrator for natural language business explanations."""

    def __init__(self, index: EmbeddingIndex, openai_model: str = "gpt-4o-mini") -> None:
        self.index = index
        self.openai_model = openai_model
        self._openai_client = None

        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            try:
                from openai import OpenAI

                self._openai_client = OpenAI(api_key=api_key)
            except Exception:
                self._openai_client = None

    @staticmethod
    def _format_context(results: list[RetrievalResult]) -> str:
        lines = []
        for item in results:
            lines.append(f"[{item.rank}] source={item.source} score={item.score:.4f}\n{item.text}")
        return "\n\n".join(lines)

    @staticmethod
    def _local_recommendation(question: str, retrieved: list[RetrievalResult]) -> str:
        lower_q = question.lower()
        recommendations: list[str] = []
        if "stockout" in lower_q:
            recommendations.append("Rebalance inventory and increase safety stock in regions with elevated stockout probability.")
        if "supplier" in lower_q or "risk" in lower_q:
            recommendations.append("Prioritize mitigation plans for high and critical risk suppliers, including dual sourcing.")
        if "forecast" in lower_q or "error" in lower_q:
            recommendations.append("Review demand drivers and recalibrate lag/seasonality features for high-error segments.")
        if "delay" in lower_q:
            recommendations.append("Investigate carrier performance and renegotiate lead-time SLAs for delayed lanes.")
        if not recommendations:
            recommendations.append("Track the listed anomalies and KPIs weekly, then trigger root-cause analysis for adverse trends.")

        evidence = "\n".join(f"- {r.source}: {r.text}" for r in retrieved[:3])
        recs = "\n".join(f"- {item}" for item in recommendations)
        return (
            "Local RAG summary (no external LLM API key configured).\n"
            "Top evidence:\n"
            f"{evidence}\n\n"
            "Recommended actions:\n"
            f"{recs}"
        )

    def _llm_answer(self, question: str, retrieved: list[RetrievalResult]) -> str:
        context = self._format_context(retrieved)
        prompt = (
            "You are an AI supply chain decision support copilot.\n"
            "Use ONLY the context below. If evidence is incomplete, state that explicitly.\n"
            "Return: (1) concise explanation, (2) likely drivers, (3) recommended actions.\n\n"
            f"Question: {question}\n\nContext:\n{context}"
        )
        if self._openai_client is None:
            return self._local_recommendation(question=question, retrieved=retrieved)

        response = self._openai_client.responses.create(
            model=self.openai_model,
            temperature=0.2,
            input=prompt,
        )
        return str(getattr(response, "output_text", "")).strip() or self._local_recommendation(question, retrieved)

    def ask(self, question: str, top_k: int = 4) -> dict[str, Any]:
        retrieved = self.index.query(question=question, top_k=top_k)
        answer = self._llm_answer(question=question, retrieved=retrieved)
        return {
            "question": question,
            "answer": answer,
            "retrieved": [
                {
                    "rank": item.rank,
                    "score": round(item.score, 4),
                    "source": item.source,
                    "text": item.text,
                    "metadata": item.metadata,
                }
                for item in retrieved
            ],
        }

    def evaluate(self, qa_set: list[dict[str, Any]], top_k: int = 4) -> pd.DataFrame:
        """
        Lightweight LLM evaluation via expected-keyword recall.

        Each QA item should include:
            {"question": "...", "expected_keywords": ["stockout", "west", ...]}
        """

        rows: list[dict[str, Any]] = []
        for item in qa_set:
            question = str(item["question"])
            expected = [str(token).lower() for token in item.get("expected_keywords", [])]
            response = self.ask(question=question, top_k=top_k)
            answer_text = response["answer"].lower()
            if expected:
                hits = sum(1 for token in expected if token in answer_text)
                keyword_recall = hits / len(expected)
            else:
                keyword_recall = np.nan
            rows.append(
                {
                    "question": question,
                    "keyword_recall": keyword_recall,
                    "retrieval_sources": ", ".join([r["source"] for r in response["retrieved"]]),
                }
            )
        return pd.DataFrame(rows)

    @classmethod
    def from_tables(
        cls,
        kpi_summary: pd.DataFrame,
        stockout_probability: pd.DataFrame,
        supplier_risk_scores: pd.DataFrame,
        anomaly_alerts: pd.DataFrame,
        forecast_model_metrics: pd.DataFrame,
    ) -> "SupplyChainCopilot":
        documents: list[dict[str, Any]] = []

        for _, row in kpi_summary.iterrows():
            documents.append(
                {
                    "source": "kpi_summary",
                    "text": f"KPI {row['kpi']} has value {row['value']}.",
                    "metadata": {"kpi": row["kpi"]},
                }
            )

        for _, row in stockout_probability.iterrows():
            documents.append(
                {
                    "source": "stockout_probability",
                    "text": f"Region {row['region']} stockout probability is {row['stockout_probability']}%.",
                    "metadata": {"region": row["region"]},
                }
            )

        top_suppliers = supplier_risk_scores.head(20)
        for _, row in top_suppliers.iterrows():
            documents.append(
                {
                    "source": "supplier_risk",
                    "text": (
                        f"Supplier {row['supplier_id']} has risk score {row['supplier_risk_score']} "
                        f"and tier {row['risk_tier']}. On-time rate {row['on_time_rate']:.3f}, "
                        f"average delay {row['average_delay_days']:.2f} days."
                    ),
                    "metadata": {"supplier_id": row["supplier_id"], "tier": row["risk_tier"]},
                }
            )

        top_anomalies = anomaly_alerts.head(40)
        for _, row in top_anomalies.iterrows():
            text = (
                f"Anomaly at {row.get('date')} in region {row.get('region')} for {row.get('product_id')}: "
                f"severity {row.get('alert_severity')}, reason {row.get('anomaly_reason')}, "
                f"delay {row.get('delivery_delay_days')}, transport_cost {row.get('transport_cost')}."
            )
            documents.append(
                {
                    "source": "anomaly_alert",
                    "text": text,
                    "metadata": {
                        "region": row.get("region"),
                        "product_id": row.get("product_id"),
                        "severity": row.get("alert_severity"),
                    },
                }
            )

        for _, row in forecast_model_metrics.iterrows():
            documents.append(
                {
                    "source": "forecast_model_metrics",
                    "text": (
                        f"Forecast model {row['model']} has test RMSE {row['test_rmse']:.3f}, "
                        f"CV RMSE mean {row['cv_rmse_mean']:.3f}, CV RMSE std {row['cv_rmse_std']:.3f}, "
                        f"bias proxy {row['bias_proxy_mean_error']:.3f}."
                    ),
                    "metadata": {"model": row["model"]},
                }
            )

        index = EmbeddingIndex.from_documents(documents=documents)
        return cls(index=index)
