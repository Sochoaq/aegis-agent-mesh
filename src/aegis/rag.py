"""Hybrid BM25 + TF-IDF Cosine Retriever with Reciprocal Rank Fusion (RRF)."""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict, Iterable, List, Tuple

from .schemas import RetrievalHit, RunbookChunk

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has",
    "in", "into", "is", "it", "of", "on", "or", "that", "the", "this", "to",
    "was", "when", "with",
}

DEFAULT_RUNBOOKS: List[RunbookChunk] = [
    RunbookChunk(
        runbook_id="RB-101",
        title="PostgreSQL & PgBouncer Connection Pool Exhaustion",
        domain="database",
        keywords=[
            "postgres", "postgresql", "pgbouncer", "hikaricp", "connection",
            "pool", "exhaustion", "idle", "transaction", "lock", "deadlock",
            "waiting", "sql", "rds", "connections",
        ],
        content=(
            "Database saturation caused by idle-in-transaction sessions exhausting "
            "PgBouncer or HikariCP connection pools, lock contention on hot tables, "
            "and acquiring connection timeouts under bursty writes. Drain idle pooled "
            "connections safely via drain_db_connection_pool before restarting workers."
        ),
        recommended_tool="drain_db_connection_pool",
        default_args={"idle_timeout_sec": 30},
    ),
    RunbookChunk(
        runbook_id="RB-102",
        title="Kubernetes Pod Memory Leak, Heap Growth & OOMKill Regression",
        domain="compute",
        keywords=[
            "kubernetes", "k8s", "pod", "oomkill", "oomkilled", "memory", "leak",
            "heap", "rss", "cgroup", "restart", "canary", "deployment", "container",
            "exit_code_137",
        ],
        content=(
            "Monotonic RSS heap memory growth inside Kubernetes pods leading to "
            "cgroup OOMKilled (Exit Code 137) crash-loops immediately following a "
            "recent canary or deployment rollout. Revert the faulty revision using "
            "rollback_deployment after human operator approval."
        ),
        recommended_tool="rollback_deployment",
        default_args={"target_revision": "rev-previous-stable"},
    ),
    RunbookChunk(
        runbook_id="RB-103",
        title="Edge Gateway BLE/MQTT Telemetry Drift, CRC Errors & Packet Loss",
        domain="edge-iot",
        keywords=[
            "edge", "gateway", "ble", "mqtt", "esp32", "sensor", "packet",
            "loss", "rssi", "snr", "crc", "telemetry", "broker", "dropout",
            "reconnect", "acquisition",
        ],
        content=(
            "Industrial or field edge gateway exhibiting elevated BLE/MQTT packet loss, "
            "CRC frame corruption, watchdog reconnect storms, or corrupted sensor frames "
            "upstream of the ingestion broker. Isolate the failing gateway onto the "
            "quarantine topic using isolate_edge_gateway."
        ),
        recommended_tool="isolate_edge_gateway",
        default_args={"mode": "quarantine-failover"},
    ),
    RunbookChunk(
        runbook_id="RB-104",
        title="API Gateway P99 Latency Spike & Upstream Worker Saturation",
        domain="networking",
        keywords=[
            "api", "gateway", "latency", "p99", "timeout", "504", "502", "nginx",
            "envoy", "throughput", "rps", "cpu", "queue", "thread", "saturation",
            "upstream", "autoscaling",
        ],
        content=(
            "Sudden HTTP 502/504 upstream timeout surge and P99 response latency spike "
            "caused by traffic bursts saturating stateless API worker queues and CPU "
            "utilization without memory leaks. Scale stateless replicas horizontally "
            "using scale_service_replicas within bounded capacity."
        ),
        recommended_tool="scale_service_replicas",
        default_args={"target_replicas": 6},
    ),
    RunbookChunk(
        runbook_id="RB-105",
        title="Credential Stuffing, IAM Token Anomaly & Unauthorized Access",
        domain="security",
        keywords=[
            "iam", "token", "jwt", "auth", "credential", "stuffing", "brute",
            "unauthorized", "401", "403", "key", "compromise", "apikey",
            "secret", "security", "intrusion",
        ],
        content=(
            "Anomalous spike in HTTP 401/403 authentication failures, leaked or "
            "replayed API tokens, or credential stuffing targeting IAM auth endpoints. "
            "Immediately revoke and reissue signing keys via rotate_api_credentials "
            "under critical security approval."
        ),
        recommended_tool="rotate_api_credentials",
        default_args={"scope": "service-iam"},
    ),
    RunbookChunk(
        runbook_id="RB-106",
        title="Model Serving Concept Drift, Feature Skew & PSI Threshold Breach",
        domain="mlops",
        keywords=[
            "model", "inference", "drift", "psi", "kl", "divergence", "feature",
            "skew", "calibration", "embedding", "accuracy", "confidence",
            "pytorch", "serving", "mlops", "nan",
        ],
        content=(
            "Production ML inference endpoint triggering Population Stability Index "
            "(PSI > 0.25) or KL-divergence feature skew alerts accompanied by "
            "confidence degradation without infrastructure saturation. Launch shadow "
            "recalibration via trigger_model_recalibration."
        ),
        recommended_tool="trigger_model_recalibration",
        default_args={"pipeline": "shadow-recalibrate"},
    ),
]


def tokenize(text: str) -> List[str]:
    """Normalize text into informative lowercase lexical tokens."""
    tokens = re.findall(r"[a-z0-9_]+", text.lower())
    return [tok for tok in tokens if tok not in STOPWORDS and len(tok) > 1]


class HybridRAGEngine:
    """In-memory hybrid retriever combining BM25Okapi and TF-IDF Cosine with RRF."""

    def __init__(
        self,
        runbooks: Iterable[RunbookChunk] | None = None,
        k1: float = 1.5,
        b: float = 0.75,
        rrf_k: int = 60,
    ) -> None:
        self.runbooks: List[RunbookChunk] = list(runbooks or DEFAULT_RUNBOOKS)
        self.k1 = k1
        self.b = b
        self.rrf_k = rrf_k
        self.n_docs = len(self.runbooks)

        # Build weighted token representations (keywords weighted 2x for domain precision)
        self.doc_tokens: List[List[str]] = []
        self.doc_tf: List[Counter[str]] = []
        self.df: Counter[str] = Counter()

        for rb in self.runbooks:
            weighted_text = f"{rb.title} {' '.join(rb.keywords)} {' '.join(rb.keywords)} {rb.content}"
            toks = tokenize(weighted_text)
            self.doc_tokens.append(toks)
            tf = Counter(toks)
            self.doc_tf.append(tf)
            for term in tf:
                self.df[term] += 1

        self.avg_dl = (
            sum(len(t) for t in self.doc_tokens) / max(1, self.n_docs)
        )

        # Precompute IDF and L2-normalized TF-IDF document vectors
        self.idf: Dict[str, float] = {}
        for term, freq in self.df.items():
            # Smooth BM25-compatible strictly positive IDF
            self.idf[term] = math.log(1.0 + (self.n_docs - freq + 0.5) / (freq + 0.5))

        self.doc_tfidf_vecs: List[Dict[str, float]] = []
        for tf in self.doc_tf:
            vec = {
                term: (1.0 + math.log(count)) * self.idf[term]
                for term, count in tf.items()
            }
            norm = math.sqrt(sum(w * w for w in vec.values())) or 1.0
            self.doc_tfidf_vecs.append({term: w / norm for term, w in vec.items()})

    def _bm25_score(self, query_tokens: List[str], doc_idx: int) -> float:
        tf = self.doc_tf[doc_idx]
        dl = len(self.doc_tokens[doc_idx]) or 1
        score = 0.0
        for term in query_tokens:
            if term not in tf:
                continue
            f = tf[term]
            idf = self.idf.get(term, 0.0)
            denom = f + self.k1 * (1.0 - self.b + self.b * (dl / self.avg_dl))
            score += idf * ((f * (self.k1 + 1.0)) / max(1e-9, denom))
        return score

    def _cosine_score(self, query_tokens: List[str], doc_idx: int) -> float:
        if not query_tokens:
            return 0.0
        q_tf = Counter(query_tokens)
        q_vec = {
            term: (1.0 + math.log(cnt)) * self.idf.get(term, math.log(1.0 + self.n_docs))
            for term, cnt in q_tf.items()
        }
        q_norm = math.sqrt(sum(w * w for w in q_vec.values())) or 1.0
        q_unit = {term: w / q_norm for term, w in q_vec.items()}

        d_vec = self.doc_tfidf_vecs[doc_idx]
        return sum(w * d_vec.get(term, 0.0) for term, w in q_unit.items())

    def search(self, query: str, top_k: int = 3) -> List[RetrievalHit]:
        """Rank runbooks using BM25 + TF-IDF Cosine fused via Reciprocal Rank Fusion."""
        q_tokens = tokenize(query)

        bm25_raw: List[Tuple[int, float]] = [
            (i, self._bm25_score(q_tokens, i)) for i in range(self.n_docs)
        ]
        cos_raw: List[Tuple[int, float]] = [
            (i, self._cosine_score(q_tokens, i)) for i in range(self.n_docs)
        ]

        bm25_order = sorted(bm25_raw, key=lambda x: (-x[1], x[0]))
        cos_order = sorted(cos_raw, key=lambda x: (-x[1], x[0]))

        bm25_ranks = {doc_idx: rank + 1 for rank, (doc_idx, _) in enumerate(bm25_order)}
        cos_ranks = {doc_idx: rank + 1 for rank, (doc_idx, _) in enumerate(cos_order)}
        bm25_scores = dict(bm25_raw)
        cos_scores = dict(cos_raw)

        fused: List[Tuple[int, float]] = []
        for i in range(self.n_docs):
            rrf = (1.0 / (self.rrf_k + bm25_ranks[i])) + (1.0 / (self.rrf_k + cos_ranks[i]))
            # Tiny tie-breaker from normalized lexical + semantic overlap so exact hits win ties cleanly
            tie = 1e-4 * (bm25_scores[i] + cos_scores[i])
            fused.append((i, rrf + tie))

        fused.sort(key=lambda x: (-x[1], x[0]))

        hits: List[RetrievalHit] = []
        for doc_idx, _ in fused[:top_k]:
            rb = self.runbooks[doc_idx]
            rrf_exact = (1.0 / (self.rrf_k + bm25_ranks[doc_idx])) + (
                1.0 / (self.rrf_k + cos_ranks[doc_idx])
            )
            hits.append(
                RetrievalHit(
                    runbook_id=rb.runbook_id,
                    title=rb.title,
                    domain=rb.domain,
                    bm25_score=round(bm25_scores[doc_idx], 4),
                    cosine_score=round(cos_scores[doc_idx], 4),
                    rrf_score=round(rrf_exact, 6),
                    bm25_rank=bm25_ranks[doc_idx],
                    cosine_rank=cos_ranks[doc_idx],
                    snippet=rb.content,
                    recommended_tool=rb.recommended_tool,
                )
            )
        return hits
