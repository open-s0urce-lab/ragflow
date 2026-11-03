#
#  Copyright 2024 The InfiniFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
import json
import logging
import math
import os
import time
from abc import ABC
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urljoin
from typing import Tuple, List
from http import HTTPStatus

import numpy as np
import requests

from common.log_utils import log_exception
from common.token_utils import num_tokens_from_string, truncate, total_token_count_from_response, usage_from_response
from rag.llm.mws_utils import mws_api_url, require_mws_token
from rag.utils.url_utils import append_api_path, ensure_v1

MAX_RERANK_TOKEN = 8196
RERANK_TOKEN_LIMIT_MODE = os.getenv("RERANK_TOKEN_LIMIT_MODE", "truncate").strip().lower()
if RERANK_TOKEN_LIMIT_MODE not in {"truncate", "passthrough", "raise_error"}:
    raise ValueError(f"Invalid RERANK_TOKEN_LIMIT_MODE {RERANK_TOKEN_LIMIT_MODE!r}; expected 'truncate', 'passthrough', or 'raise_error'")

# Backends such as GPUStack (llama.cpp) count the query and *every* document
# against a single context window, while the upstream adapters only cap each
# document individually. Measured on the corporate GPUStack with Russian text:
# a document estimated at ~6.9k tiktoken tokens (~29.6k chars) still fits the
# 8192-token window while ~11.6k estimated tokens is rejected, i.e. the server
# tokenizer counts ~1.2x more than tiktoken. 0.8 keeps a whole batch under the
# real limit; RERANK_BATCH_BUDGET_RATIO can be lowered further per deployment.
RERANK_BATCH_BUDGET_RATIO = float(os.getenv("RERANK_BATCH_BUDGET_RATIO", "0.8"))
if not 0 < RERANK_BATCH_BUDGET_RATIO <= 1:
    raise ValueError(f"Invalid RERANK_BATCH_BUDGET_RATIO {RERANK_BATCH_BUDGET_RATIO!r}; expected (0, 1]")

# A rejected batch is split in half (and a rejected single document is
# shortened) this many times before the error is reported. The tokenizer
# estimate is only a hint; the server has the final say.
RERANK_MAX_SPLIT_DEPTH = int(os.getenv("RERANK_MAX_SPLIT_DEPTH", "6"))

# Cross-encoders only need enough text to judge relevance, and a 512-token
# chunk is ~1.4k chars of Russian: capping each document keeps 64 candidates
# at a few batches instead of dozens (normal chunks are unaffected, oversized
# parse artifacts are cut).
RERANK_DOC_TOKEN_CAP = int(os.getenv("RERANK_DOC_TOKEN_CAP", "1024"))

# Batch requests are independent, so a few run in parallel to keep the chat
# latency acceptable. Set to 1 to make the adapter fully sequential.
RERANK_MAX_WORKERS = max(1, int(os.getenv("RERANK_MAX_WORKERS", "4")))


def build_rerank_batches(query: str, texts: List[str], max_token: int = MAX_RERANK_TOKEN, budget_ratio: float = RERANK_BATCH_BUDGET_RATIO) -> List[Tuple[List[int], List[str]]]:
    """Pack ``texts`` into batches that fit one rerank context window.

    Returns ``[(original_indices, batch_texts), ...]`` preserving the input
    order. A document larger than the remaining budget is truncated to the
    per-request document budget so a batch of one always fits, and documents
    are never reordered: cross-encoder scores stay comparable across batches
    because ``Base.similarity`` normalizes the merged array afterwards.
    """
    if not texts:
        return []
    total_budget = max(int(max_token * budget_ratio), 1)
    doc_budget = max(total_budget - num_tokens_from_string(query), 1)

    batches: List[Tuple[List[int], List[str]]] = []
    indices: List[int] = []
    batch: List[str] = []
    used = 0
    for index, text in enumerate(texts):
        tokens = num_tokens_from_string(text)
        if tokens > doc_budget:
            text = truncate(text, doc_budget)
            tokens = num_tokens_from_string(text)
        if batch and used + tokens > doc_budget:
            batches.append((indices, batch))
            indices, batch, used = [], [], 0
        indices.append(index)
        batch.append(text)
        used += tokens
    if batch:
        batches.append((indices, batch))
    return batches


def _describe_rerank_usage(res) -> str:
    """Describe where a rerank response keeps its token usage, for diagnostics.

    Providers disagree on the usage layout (``usage.total_tokens``,
    ``usage.prompt_tokens``, ``meta.tokens``, ``meta.billed_units``, ...), and an
    unmapped shape silently degrades ``token_count`` to 0. Only structural
    fields are rendered here so the query and documents are never logged.
    """
    if not isinstance(res, dict):
        return f"response_type={type(res).__name__} (expected dict)"
    usage = res.get("usage")
    meta = res.get("meta")
    usage_keys = sorted(usage) if isinstance(usage, dict) else usage
    meta_keys = sorted(meta) if isinstance(meta, dict) else meta
    return f"top_level_keys={sorted(res)}, usage={usage_keys}, meta={meta_keys}"


class Base(ABC):
    def __init__(self, key, model_name, max_token=MAX_RERANK_TOKEN, **kwargs):
        self.max_token = max_token or MAX_RERANK_TOKEN

    def similarity(self, query: str, texts: List) -> Tuple[np.ndarray, int]:
        """Score ``texts`` against ``query`` and return ``(rank, token_count)``.

        This is the single public entry point shared by every reranker. It
        short-circuits empty input, applies the configured token-limit policy,
        and guarantees the returned scores are min-max normalized to ``[0, 1]``
        regardless of what the backend emits (relevance scores, cosine
        similarities or raw logits). Downstream
        hybrid scoring blends the reranker output with token similarity on a
        fixed ``[0, 1]`` scale, so an un-normalized provider (e.g. NVIDIA's
        unbounded logits) would otherwise corrupt the final ordering.

        Subclasses implement provider-specific scoring in :meth:`_compute_rank`
        and must not normalize themselves.
        """
        if not query or not texts:
            logging.debug("Rerank %s short-circuits on empty input: query=%s texts=%d, returning 0 tokens", self.__class__.__name__, bool(query), len(texts) if texts else 0)
            return np.zeros(len(texts) if texts else 0, dtype=float), 0
        if RERANK_TOKEN_LIMIT_MODE != "passthrough":
            query_tokens = num_tokens_from_string(query)
            if RERANK_TOKEN_LIMIT_MODE == "truncate":
                texts = [truncate(text, max(self.max_token - query_tokens, 0)) for text in texts]
            else:
                for index, text in enumerate(texts):
                    input_tokens = query_tokens + num_tokens_from_string(text)
                    if input_tokens > self.max_token:
                        raise ValueError(f"Rerank input at document index {index} has {input_tokens} tokens, exceeding the configured maximum of {self.max_token}")
        rank, token_count = self._compute_rank(query, texts)
        rank = np.asarray(rank, dtype=float)
        if rank.size:
            logging.debug(
                "Rerank %s scores before normalization: count=%d min=%.4f max=%.4f",
                self.__class__.__name__,
                rank.size,
                float(np.min(rank)),
                float(np.max(rank)),
            )
        return self._normalize_rank(rank), token_count

    def _compute_rank(self, query: str, texts: List) -> Tuple[np.ndarray, int]:
        """Provider-specific scoring. ``query`` and ``texts`` are non-empty."""
        raise NotImplementedError("Please implement _compute_rank method!")

    @staticmethod
    def _normalize_rank(rank: np.ndarray) -> np.ndarray:
        """Guarantee scores land in ``[0, 1]`` for the hybrid blend.

        Providers that already emit calibrated relevance scores in ``[0, 1]``
        (Cohere, Jina, Voyage, ...) are returned unchanged, so their absolute
        magnitudes, ``similarity_threshold`` semantics and reported
        ``vector_similarity`` are preserved. Only out-of-range output (e.g.
        NVIDIA's unbounded, often negative logits) is rescaled: a batch with a
        usable spread is min-max mapped onto ``[0, 1]`` (which stops a negative
        logit from dragging a relevant chunk below pure keyword matches once
        weighted by ``vtweight``), while a spreadless batch (including a single
        candidate) has no relative signal and is clamped instead, so a lone
        high score is not silently zeroed.
        """
        if rank.size == 0:
            return rank
        min_rank = float(np.min(rank))
        max_rank = float(np.max(rank))

        if min_rank >= 0.0 and max_rank <= 1.0:
            return rank
        span = max_rank - min_rank
        if span < 1e-3:
            return np.clip(rank, 0.0, 1.0)
        return (rank - min_rank) / span


class JinaRerank(Base):
    _FACTORY_NAME = "Jina"

    def __init__(self, key, model_name="jina-reranker-v2-base-multilingual", base_url="https://api.jina.ai/v1/rerank", max_token=MAX_RERANK_TOKEN):
        super().__init__(key, model_name, max_token)
        self.base_url = base_url or "https://api.jina.ai/v1/rerank"
        self.headers = {"Content-Type": "application/json", "Authorization": f"Bearer {key}"}
        self.model_name = model_name

    def _compute_rank(self, query: str, texts: List) -> Tuple[np.ndarray, int]:
        data = {"model": self.model_name, "query": query, "documents": texts, "top_n": len(texts)}
        response = requests.post(self.base_url, headers=self.headers, json=data, timeout=30)
        response.raise_for_status()
        res = response.json()
        rank = np.zeros(len(texts), dtype=float)
        try:
            for d in res.get("results", []):
                rank[d["index"]] = d["relevance_score"]
        except Exception as _e:
            log_exception(_e, res)
        usage = usage_from_response(res)
        token_count = usage["total_tokens"]
        if token_count:
            logging.debug(
                "Rerank %s token usage from response: total_tokens=%d prompt_tokens=%d completion_tokens=%d",
                self.__class__.__name__,
                token_count,
                usage["prompt_tokens"],
                usage["completion_tokens"],
            )
        else:
            # Moark/GiteeAI report usage in camelCase and can still send zeros, so
            # count the query and documents locally instead of reporting a
            # zero-token rerank.
            token_count = num_tokens_from_string(query) + sum(num_tokens_from_string(text) for text in texts)
            logging.debug(
                "Rerank %s response carried no usable token usage; fell back to local count=%d (model=%s endpoint=%s); response shape: %s",
                self.__class__.__name__,
                token_count,
                self.model_name,
                self.base_url,
                _describe_rerank_usage(res),
            )
        return rank, token_count


class GreenPTRerank(JinaRerank):
    """GreenPT native reranking adapter."""

    _FACTORY_NAME = "GreenPT"

    def __init__(self, key, model_name="green-rerank", base_url="https://api.greenpt.ai/v1/rerank", max_token=MAX_RERANK_TOKEN):
        endpoint = (base_url or "https://api.greenpt.ai/v1/rerank").rstrip("/")
        if not endpoint.endswith("/rerank"):
            endpoint += "/rerank"
        super().__init__(key, model_name=model_name, base_url=endpoint, max_token=max_token)


class XInferenceRerank(Base):
    _FACTORY_NAME = "Xinference"

    def __init__(self, key="x", model_name="", base_url="", max_token=MAX_RERANK_TOKEN):
        super().__init__(key, model_name, max_token)
        if base_url.find("/v1") == -1:
            base_url = urljoin(base_url, "/v1/rerank")
        if base_url.find("/rerank") == -1:
            base_url = urljoin(base_url, "/v1/rerank")
        self.model_name = model_name
        self.base_url = base_url
        self.headers = {"Content-Type": "application/json", "accept": "application/json"}
        if key and key != "x":
            self.headers["Authorization"] = f"Bearer {key}"

    def _compute_rank(self, query: str, texts: List) -> Tuple[np.ndarray, int]:
        token_count = sum(num_tokens_from_string(t) for t in texts)
        data = {"model": self.model_name, "query": query, "return_documents": "true", "return_len": "true", "documents": texts}
        response = requests.post(self.base_url, headers=self.headers, json=data, timeout=30)
        response.raise_for_status()
        res = response.json()
        rank = np.zeros(len(texts), dtype=float)
        try:
            for d in res.get("results", []):
                rank[d["index"]] = d["relevance_score"]
        except Exception as _e:
            log_exception(_e, res)
        return rank, token_count


class LocalAIRerank(Base):
    _FACTORY_NAME = "LocalAI"

    def __init__(self, key, model_name, base_url, max_token=MAX_RERANK_TOKEN):
        super().__init__(key, model_name, max_token)
        if base_url.find("/rerank") == -1:
            self.base_url = urljoin(base_url, "/rerank")
        else:
            self.base_url = base_url
        self.headers = {"Content-Type": "application/json", "Authorization": f"Bearer {key}"}
        self.model_name = model_name.split("___")[0]

    def _compute_rank(self, query: str, texts: List) -> Tuple[np.ndarray, int]:
        data = {
            "model": self.model_name,
            "query": query,
            "documents": texts,
            "top_n": len(texts),
        }
        token_count = 0
        for t in texts:
            token_count += num_tokens_from_string(t)
        response = requests.post(self.base_url, headers=self.headers, json=data, timeout=30)
        response.raise_for_status()
        res = response.json()
        rank = np.zeros(len(texts), dtype=float)
        try:
            for d in res.get("results", []):
                rank[d["index"]] = d["relevance_score"]
        except Exception as _e:
            log_exception(_e, res)
        return rank, token_count


class NvidiaRerank(Base):
    _FACTORY_NAME = "NVIDIA"

    def __init__(self, key, model_name, base_url="https://ai.api.nvidia.com/v1/retrieval/nvidia/", max_token=MAX_RERANK_TOKEN):
        super().__init__(key, model_name, max_token)
        if not base_url:
            base_url = "https://ai.api.nvidia.com/v1/retrieval/nvidia/"
        base_url = base_url.rstrip("/") + "/"
        self.model_name = model_name

        # Default to NVIDIA's generic reranking endpoint. base_url used to be
        # assigned only inside the two model-specific branches below, so any
        # other model left the attribute unset and raised AttributeError on the
        # first _compute_rank() call.
        self.base_url = urljoin(base_url, "reranking")

        if self.model_name == "nvidia/nv-rerankqa-mistral-4b-v3":
            self.base_url = urljoin(base_url, "nv-rerankqa-mistral-4b-v3/reranking")
        elif self.model_name == "nvidia/rerank-qa-mistral-4b":
            self.base_url = urljoin(base_url, "reranking")
            self.model_name = "nv-rerank-qa-mistral-4b:1"
        else:
            logging.info(
                "nvidia_rerank_fallback_endpoint_assigned",
                extra={
                    "provider": self._FACTORY_NAME,
                    "model": self.model_name,
                    "endpoint": self.base_url,
                },
            )

        self.headers = {
            "accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}",
        }

    def _compute_rank(self, query: str, texts: List) -> Tuple[np.ndarray, int]:
        token_count = num_tokens_from_string(query) + sum([num_tokens_from_string(t) for t in texts])
        data = {
            "model": self.model_name,
            "query": {"text": query},
            "passages": [{"text": text} for text in texts],
            "top_n": len(texts),
        }
        response = requests.post(self.base_url, headers=self.headers, json=data, timeout=30)
        response.raise_for_status()
        res = response.json()
        rank = np.zeros(len(texts), dtype=float)
        try:
            for d in res.get("rankings", []):
                rank[d["index"]] = d["logit"]
        except Exception as _e:
            log_exception(_e, res)
        return rank, token_count


class LmStudioRerank(Base):
    _FACTORY_NAME = "LM-Studio"

    def __init__(self, key, model_name, base_url, max_token=MAX_RERANK_TOKEN, **kwargs):
        super().__init__(key, model_name, max_token)

    def _compute_rank(self, query: str, texts: List) -> Tuple[np.ndarray, int]:
        raise NotImplementedError("The LmStudioRerank has not been implemented")


class OpenAI_APIRerank(Base):
    _FACTORY_NAME = "OpenAI-API-Compatible"

    def __init__(self, key, model_name, base_url, max_token=MAX_RERANK_TOKEN):
        super().__init__(key, model_name, max_token)
        normalized_base_url = (base_url or "").strip()
        if "/rerank" in normalized_base_url:
            self.base_url = normalized_base_url.rstrip("/")
        else:
            self.base_url = urljoin(f"{normalized_base_url.rstrip('/')}/", "rerank").rstrip("/")
        self.headers = {"Content-Type": "application/json", "Authorization": f"Bearer {key}"}
        self.model_name = model_name.split("___")[0]

    def _compute_rank(self, query: str, texts: List) -> Tuple[np.ndarray, int]:
        data = {
            "model": self.model_name,
            "query": query,
            "documents": texts,
            "top_n": len(texts),
        }
        token_count = 0
        for t in texts:
            token_count += num_tokens_from_string(t)
        response = requests.post(self.base_url, headers=self.headers, json=data, timeout=30)
        response.raise_for_status()
        res = response.json()
        rank = np.zeros(len(texts), dtype=float)
        try:
            for d in res.get("results", []):
                rank[d["index"]] = d["relevance_score"]
        except Exception as _e:
            log_exception(_e, res)
        return rank, token_count


class MWSRerank(OpenAI_APIRerank):
    """MWS reranker using its Cohere-compatible v2 endpoint."""

    _FACTORY_NAME = "MWS"

    def __init__(self, key, model_name, base_url, max_token=MAX_RERANK_TOKEN):
        """Initialize reranking access for an MWS project and deployment."""
        token = require_mws_token(key)
        super().__init__(token, model_name, mws_api_url(base_url, "cohere/v2/rerank"), max_token)

    def _compute_rank(self, query: str, texts: List) -> Tuple[np.ndarray, int]:
        """Score candidate texts and restore scores to document input order."""
        documents = texts
        log_context = {
            "provider": self._FACTORY_NAME,
            "operation": "rerank",
            "endpoint": self.base_url,
            "model": self.model_name,
            "document_count": len(documents),
        }
        logging.info("mws_rerank_request", extra=log_context)

        try:
            response = requests.post(
                self.base_url,
                headers=self.headers,
                json={
                    "model": self.model_name,
                    "query": query,
                    "documents": documents,
                    "top_n": len(documents),
                },
                timeout=30,
            )
            response.raise_for_status()
        except Exception as error:
            logging.warning(
                "mws_rerank_failed",
                extra={
                    **log_context,
                    "failure_stage": "http_request",
                    "error_type": type(error).__name__,
                },
            )
            raise

        try:
            payload = response.json()
        except Exception as error:
            logging.warning(
                "mws_rerank_failed",
                extra={
                    **log_context,
                    "failure_stage": "json_parsing",
                    "error_type": type(error).__name__,
                },
            )
            raise

        try:
            results = payload.get("results") if isinstance(payload, dict) else None
            if not isinstance(results, list) or len(results) != len(documents):
                count = len(results) if isinstance(results, list) else 0
                raise ValueError(f"MWS returned {count} rerank results for {len(documents)} documents")

            rank = np.zeros(len(documents), dtype=float)
            seen = set()
            for item in results:
                index = item.get("index") if isinstance(item, dict) else None
                if not isinstance(index, int) or isinstance(index, bool) or index < 0 or index >= len(documents) or index in seen:
                    raise ValueError(f"unexpected MWS rerank index: {index}")
                relevance_score = item.get("relevance_score")
                if isinstance(relevance_score, bool) or not isinstance(relevance_score, (int, float)) or not math.isfinite(relevance_score):
                    raise ValueError(f"unexpected MWS rerank relevance_score at index {index}: {relevance_score!r}")
                seen.add(index)
                rank[index] = relevance_score
        except Exception as error:
            logging.warning(
                "mws_rerank_failed",
                extra={
                    **log_context,
                    "failure_stage": "response_validation",
                    "error_type": type(error).__name__,
                },
            )
            raise

        token_count = num_tokens_from_string(query) + sum(num_tokens_from_string(document) for document in documents)
        return rank, token_count


class CoHereRerank(Base):
    _FACTORY_NAME = ["Cohere", "VLLM"]

    def __init__(self, key, model_name, base_url=None, max_token=MAX_RERANK_TOKEN):
        super().__init__(key, model_name, max_token)
        from cohere import Client

        client_kwargs = {"api_key": key, "timeout": 30.0}
        if base_url and base_url.strip():
            client_kwargs["base_url"] = base_url
        self.client = Client(**client_kwargs)
        self.model_name = model_name.split("___")[0]

    def _compute_rank(self, query: str, texts: List) -> Tuple[np.ndarray, int]:
        token_count = num_tokens_from_string(query) + sum([num_tokens_from_string(t) for t in texts])
        res = self.client.rerank(
            model=self.model_name,
            query=query,
            documents=texts,
            top_n=len(texts),
            return_documents=False,
        )
        rank = np.zeros(len(texts), dtype=float)
        try:
            for d in res.results:
                rank[d.index] = d.relevance_score
        except Exception as _e:
            log_exception(_e, res)
        return rank, token_count


# Reranker connector for AWS Bedrock, calling the bedrock-agent-runtime Rerank
# API (e.g. amazon.rerank-v1:0, cohere.rerank-v3-5:0). The JSON key protocol
# (auth_mode / bedrock_region / bedrock_ak / bedrock_sk) mirrors BedrockEmbed in
# embedding_model.py.
class BedrockRerank(Base):
    _FACTORY_NAME = "Bedrock"

    # A single request accepts at most 1,000 sources / numberOfResults.
    _MAX_SOURCES = 1000

    def __init__(self, key, model_name, max_token=MAX_RERANK_TOKEN, **kwargs):
        super().__init__(key, model_name, max_token)
        import boto3
        from botocore.utils import validate_region_name

        key = json.loads(key)
        mode = key.get("auth_mode")
        if not mode:
            logging.error("Bedrock auth_mode is not provided in the key")
            raise ValueError("Bedrock auth_mode must be provided in the key")

        self.bedrock_region = key.get("bedrock_region")
        if not self.bedrock_region:
            raise ValueError("Bedrock region must be provided in the key")
        validate_region_name(self.bedrock_region)
        self.model_name = model_name
        # On-demand foundation-model ARN; works for amazon.rerank-v1:0 / cohere.rerank-*.
        self.model_arn = f"arn:aws:bedrock:{self.bedrock_region}::foundation-model/{self.model_name}"
        # Rerank lives on the bedrock-agent-runtime service, not bedrock-runtime.
        if mode == "access_key_secret":
            self.client = boto3.client(
                service_name="bedrock-agent-runtime",
                region_name=self.bedrock_region,
                aws_access_key_id=key.get("bedrock_ak"),
                aws_secret_access_key=key.get("bedrock_sk"),
            )
        elif mode == "iam_role":
            sts_client = boto3.client("sts", region_name=self.bedrock_region)
            resp = sts_client.assume_role(RoleArn=key.get("aws_role_arn"), RoleSessionName="BedrockSession")
            creds = resp["Credentials"]
            self.client = boto3.client(
                service_name="bedrock-agent-runtime",
                region_name=self.bedrock_region,
                aws_access_key_id=creds["AccessKeyId"],
                aws_secret_access_key=creds["SecretAccessKey"],
                aws_session_token=creds["SessionToken"],
            )
        elif mode == "assume_role":
            self.client = boto3.client("bedrock-agent-runtime", region_name=self.bedrock_region)
        elif mode == "bedrock_api_key":
            raise ValueError("Bedrock API key authentication is not supported for rerank")
        else:
            raise ValueError(f"Unsupported Bedrock auth_mode: {mode}")

    def _compute_rank(self, query: str, texts: List) -> Tuple[np.ndarray, int]:
        # Bedrock does not report token usage; count locally like CoHereRerank.
        token_count = num_tokens_from_string(query) + sum(num_tokens_from_string(t) for t in texts)

        rank = np.zeros(len(texts), dtype=float)
        result_count = 0
        started = time.perf_counter()
        # Both `sources` and `numberOfResults` are capped at 1,000 per request;
        # rerank in batches and map each score back to its global position.
        for offset in range(0, len(texts), self._MAX_SOURCES):
            batch = texts[offset : offset + self._MAX_SOURCES]
            sources = [{"type": "INLINE", "inlineDocumentSource": {"type": "TEXT", "textDocument": {"text": t}}} for t in batch]
            reranking_config = {
                "type": "BEDROCK_RERANKING_MODEL",
                "bedrockRerankingConfiguration": {
                    "numberOfResults": len(batch),
                    "modelConfiguration": {"modelArn": self.model_arn},
                },
            }
            # Drain paginated results: the API may split a batch across nextToken pages.
            next_token = None
            while True:
                request = {"queries": [{"type": "TEXT", "textQuery": {"text": query}}], "sources": sources, "rerankingConfiguration": reranking_config}
                if next_token:
                    request["nextToken"] = next_token
                res = self.client.rerank(**request)
                try:
                    for d in res.get("results", []):
                        rank[offset + d["index"]] = d["relevanceScore"]
                        result_count += 1
                except (KeyError, IndexError, TypeError) as _e:
                    log_exception(_e, res)
                next_token = res.get("nextToken")
                if not next_token:
                    break
        # Safe diagnostics only: no query, document text or credentials.
        logging.debug(
            "BedrockRerank model=%s region=%s sources=%d tokens=%d results=%d elapsed=%.3fs",
            self.model_name,
            self.bedrock_region,
            len(texts),
            token_count,
            result_count,
            time.perf_counter() - started,
        )
        return rank, token_count


class TogetherAIRerank(Base):
    _FACTORY_NAME = "TogetherAI"

    def __init__(self, key, model_name, base_url, max_token=MAX_RERANK_TOKEN, **kwargs):
        super().__init__(key, model_name, max_token)

    def _compute_rank(self, query: str, texts: List) -> Tuple[np.ndarray, int]:
        raise NotImplementedError("The api has not been implemented")


class SILICONFLOWRerank(Base):
    _FACTORY_NAME = "SILICONFLOW"

    def __init__(self, key, model_name, base_url="https://api.siliconflow.cn/v1/rerank", max_token=MAX_RERANK_TOKEN):
        super().__init__(key, model_name, max_token)
        normalized_base_url = (base_url or "").strip()
        if not normalized_base_url:
            normalized_base_url = "https://api.siliconflow.cn/v1/rerank"
        if "/rerank" not in normalized_base_url:
            normalized_base_url = urljoin(f"{normalized_base_url.rstrip('/')}/", "rerank").rstrip("/")
        self.model_name = model_name
        self.base_url = normalized_base_url
        self.headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "authorization": f"Bearer {key}",
        }

    def _compute_rank(self, query: str, texts: List) -> Tuple[np.ndarray, int]:
        payload = {
            "model": self.model_name,
            "query": query,
            "documents": texts,
            "top_n": len(texts),
            "return_documents": False,
            "max_chunks_per_doc": 1024,
            "overlap_tokens": 80,
        }
        response = requests.post(self.base_url, json=payload, headers=self.headers, timeout=30)
        response.raise_for_status()
        res = response.json()
        rank = np.zeros(len(texts), dtype=float)
        try:
            for d in res.get("results", []):
                rank[d["index"]] = d["relevance_score"]
        except Exception as _e:
            log_exception(_e, response)
        return rank, total_token_count_from_response(res)


class BaiduYiyanRerank(Base):
    _FACTORY_NAME = "BaiduYiyan"

    def __init__(self, key, model_name, base_url=None, max_token=MAX_RERANK_TOKEN):
        super().__init__(key, model_name, max_token)
        from qianfan.resources import Reranker

        try:
            key_obj = json.loads(key)
        except (json.JSONDecodeError, TypeError):
            key_obj = key
        if isinstance(key_obj, dict):
            ak = key_obj.get("yiyan_ak", "")
            sk = key_obj.get("yiyan_sk", "")
            self.client = Reranker(ak=ak, sk=sk, request_timeout=30)
        else:
            # adapt to one-line api_key
            self.client = Reranker(access_token=key_obj, request_timeout=30)
        self.model_name = model_name

    def _compute_rank(self, query: str, texts: List) -> Tuple[np.ndarray, int]:
        res = self.client.do(
            model=self.model_name,
            query=query,
            documents=texts,
            top_n=len(texts),
        ).body
        rank = np.zeros(len(texts), dtype=float)
        try:
            for d in res.get("results", []):
                rank[d["index"]] = d["relevance_score"]
        except Exception as _e:
            log_exception(_e, res)
        return rank, total_token_count_from_response(res)


class VoyageRerank(Base):
    _FACTORY_NAME = "Voyage AI"

    def __init__(self, key, model_name, base_url=None, max_token=MAX_RERANK_TOKEN):
        super().__init__(key, model_name, max_token)
        import voyageai

        self.client = voyageai.Client(api_key=key, timeout=30.0)
        self.model_name = model_name

    def _compute_rank(self, query: str, texts: List) -> Tuple[np.ndarray, int]:
        rank = np.zeros(len(texts), dtype=float)

        res = self.client.rerank(query=query, documents=texts, model=self.model_name, top_k=len(texts))
        try:
            for r in res.results:
                rank[r.index] = r.relevance_score
        except Exception as _e:
            log_exception(_e, res)
        return rank, res.total_tokens


class QWenRerank(Base):
    _FACTORY_NAME = "Tongyi-Qianwen"

    def __init__(self, key, model_name="gte-rerank-v2", max_token=MAX_RERANK_TOKEN, **kwargs):
        super().__init__(key, model_name, max_token)
        self.api_key = key
        self.model_name = model_name if model_name else "gte-rerank-v2"
        # Remove invalid global timeout, use official SDK per-request timeout parameter
        self.request_timeout = 30.0

    def _compute_rank(self, query: str, texts: List) -> Tuple[np.ndarray, int]:
        import dashscope

        # Pass official request_timeout parameter to both API call branches
        if self.model_name.startswith("qwen3-rerank"):
            resp = dashscope.TextReRank.call(api_key=self.api_key, model=self.model_name, query=query, documents=texts, top_n=len(texts), request_timeout=self.request_timeout)
        else:
            resp = dashscope.TextReRank.call(api_key=self.api_key, model=self.model_name, query=query, documents=texts, top_n=len(texts), return_documents=False, request_timeout=self.request_timeout)

        rank = np.zeros(len(texts), dtype=float)
        if resp.status_code == HTTPStatus.OK:
            try:
                for r in resp.output.results:
                    rank[r.index] = r.relevance_score
            except Exception as _e:
                log_exception(_e, resp)
            return rank, total_token_count_from_response(resp)
        else:
            try:
                error_body = resp["text"] if isinstance(resp, dict) and "text" in resp else None
            except Exception:
                error_body = None
            if not error_body:
                try:
                    error_body = json.dumps(dict(resp), ensure_ascii=False)
                except Exception:
                    error_body = str(resp)
            raise ValueError(f"Error calling QWenRerank model {self.model_name}: {resp.status_code} - {error_body}")


class HuggingfaceRerank(Base):
    _FACTORY_NAME = "HuggingFace"

    @staticmethod
    def post(query: str, texts: list, url: str = "http://127.0.0.1"):
        exc = None
        scores = [0 for _ in range(len(texts))]
        batch_size = 8
        # FIX: Robust URL construction to avoid duplicate "/rerank" path suffix
        base_url = url.rstrip("/")
        if not base_url.startswith(("http://", "https://")):
            base_url = f"http://{base_url}"
        # Only append "/rerank" when endpoint does not already end with it
        endpoint = base_url if base_url.endswith("/rerank") else f"{base_url}/rerank"

        for i in range(0, len(texts), batch_size):
            try:
                # Fix: Add request timeout
                res = requests.post(
                    endpoint, headers={"Content-Type": "application/json"}, json={"query": query, "texts": texts[i : i + batch_size], "raw_scores": False, "truncate": False}, timeout=30
                )
                res.raise_for_status()
                for o in res.json():
                    scores[o["index"] + i] = o["score"]
            except Exception as e:
                exc = e

        if exc:
            raise exc
        return np.array(scores)

    def __init__(self, key, model_name="BAAI/bge-reranker-v2-m3", base_url="http://127.0.0.1", max_token=MAX_RERANK_TOKEN):
        super().__init__(key, model_name, max_token)
        self.model_name = model_name.split("___")[0]
        self.base_url = base_url

    def _compute_rank(self, query: str, texts: List) -> tuple[np.ndarray, int]:
        token_count = 0
        for t in texts:
            token_count += num_tokens_from_string(t)
        return HuggingfaceRerank.post(query, texts, self.base_url), token_count


class GPUStackRerank(Base):
    """GPUStack reranking adapter."""

    _FACTORY_NAME = "GPUStack"

    def __init__(self, key, model_name, base_url, max_token=MAX_RERANK_TOKEN):
        super().__init__(key, model_name, max_token)
        if not base_url:
            raise ValueError("url cannot be None")

        self.model_name = model_name
        self.base_url = append_api_path(ensure_v1(base_url), "rerank")
        self.headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "authorization": f"Bearer {key}",
        }

    def _post_batch(self, query: str, documents: List[str]):
        payload = {
            "model": self.model_name,
            "query": query,
            "documents": documents,
            "top_n": len(documents),
        }
        response = requests.post(self.base_url, json=payload, headers=self.headers, timeout=30)
        response.raise_for_status()
        return response.json()

    def _score_batch(self, query: str, indices: List[int], batch: List[str], rank: np.ndarray, depth: int = 0) -> int:
        """Score one batch, splitting it when the backend rejects the size.

        The local tokenizer estimate can undershoot the server tokenizer, so a
        context-length 400 is handled here instead of being escalated: the
        batch is halved, and a single document is shortened, until it fits.
        """
        try:
            response_json = self._post_batch(query, batch)
        except requests.exceptions.HTTPError as error:
            status = getattr(getattr(error, "response", None), "status_code", None)
            if status == 400 and depth < RERANK_MAX_SPLIT_DEPTH:
                if len(batch) > 1:
                    middle = len(batch) // 2
                    return (
                        self._score_batch(query, indices[:middle], batch[:middle], rank, depth + 1)
                        + self._score_batch(query, indices[middle:], batch[middle:], rank, depth + 1)
                    )
                shortened = batch[0][: max(int(len(batch[0]) * 0.75), 1)]
                if shortened and shortened != batch[0]:
                    return self._score_batch(query, indices, [shortened], rank, depth + 1)
            raise ValueError(f"Error calling GPUStackRerank model {self.model_name}: {str(error)}") from error
        except requests.exceptions.RequestException as error:
            raise ValueError(f"Error calling GPUStackRerank model {self.model_name}: {str(error)}") from error

        try:
            for result in response_json.get("results", []):
                local_index = result.get("index")
                if not isinstance(local_index, int) or not 0 <= local_index < len(indices):
                    continue
                rank[indices[local_index]] = result["relevance_score"]
        except Exception as _e:
            log_exception(_e, response_json)
        return sum(num_tokens_from_string(text) for text in batch)

    def _compute_rank(self, query: str, texts: List) -> Tuple[np.ndarray, int]:
        rank = np.zeros(len(texts), dtype=float)
        prepared = [
            truncate(text, RERANK_DOC_TOKEN_CAP)
            if RERANK_DOC_TOKEN_CAP > 0 and num_tokens_from_string(text) > RERANK_DOC_TOKEN_CAP
            else text
            for text in texts
        ]
        batches = build_rerank_batches(query, prepared, self.max_token)
        if not batches:
            return (rank, 0)

        workers = min(RERANK_MAX_WORKERS, len(batches))
        if workers == 1:
            token_count = sum(self._score_batch(query, indices, batch, rank) for indices, batch in batches)
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                token_count = sum(pool.map(lambda item: self._score_batch(query, item[0], item[1], rank), batches))
        return (rank, token_count)


class NovitaRerank(JinaRerank):
    _FACTORY_NAME = "NovitaAI"

    def __init__(self, key, model_name, base_url="https://api.novita.ai/v3/openai/rerank", max_token=MAX_RERANK_TOKEN):
        if not base_url:
            base_url = "https://api.novita.ai/v3/openai/rerank"
        super().__init__(key, model_name, base_url, max_token)


class GiteeRerank(JinaRerank):
    _FACTORY_NAME = "GiteeAI"

    def __init__(self, key, model_name, base_url="https://api.moark.com/v1/rerank", max_token=MAX_RERANK_TOKEN):
        endpoint = (base_url or "https://api.moark.com/v1/rerank").rstrip("/")
        if endpoint.endswith("/v1"):
            endpoint += "/rerank"
        super().__init__(key, model_name, base_url=endpoint, max_token=max_token)


class Ai302Rerank(Base):
    _FACTORY_NAME = "302.AI"

    def __init__(self, key, model_name, base_url="https://api.302.ai/v1/rerank", max_token=MAX_RERANK_TOKEN):
        super().__init__(key, model_name, max_token)
        self.base_url = base_url or "https://api.302.ai/v1/rerank"
        self.headers = {"Content-Type": "application/json", "Authorization": f"Bearer {key}"}
        self.model_name = model_name

    def _compute_rank(self, query: str, texts: List) -> Tuple[np.ndarray, int]:
        data = {"model": self.model_name, "query": query, "documents": texts, "top_n": len(texts)}
        response = requests.post(self.base_url, headers=self.headers, json=data, timeout=30)
        response.raise_for_status()
        res = response.json()
        rank = np.zeros(len(texts), dtype=float)
        try:
            for d in res.get("results", []):
                rank[d["index"]] = d["relevance_score"]
        except Exception as _e:
            log_exception(_e, res)
        return rank, total_token_count_from_response(res)


class JiekouAIRerank(JinaRerank):
    _FACTORY_NAME = "Jiekou.AI"

    def __init__(self, key, model_name, base_url="https://api.jiekou.ai/openai/v1/rerank", max_token=MAX_RERANK_TOKEN):
        if not base_url:
            base_url = "https://api.jiekou.ai/openai/v1/rerank"
        super().__init__(key, model_name, base_url, max_token)


class FuturMixRerank(OpenAI_APIRerank):
    _FACTORY_NAME = "FuturMix"

    def __init__(self, key, model_name, base_url="https://futurmix.ai/v1/rerank", max_token=MAX_RERANK_TOKEN):
        if not base_url:
            base_url = "https://futurmix.ai/v1/rerank"
        super().__init__(key, model_name, base_url, max_token)
        logging.info("[FuturMix] Rerank initialized with model %s", model_name)


class RAGconRerank(Base):
    _FACTORY_NAME = "RAGcon"

    def __init__(self, key, model_name, base_url=None, max_token=MAX_RERANK_TOKEN, **kwargs):
        super().__init__(key, model_name, max_token)
        if not base_url:
            base_url = "https://connect.ragcon.com/v1"

        self._api_key = key
        self._base_url = base_url

        self.headers = {"Content-Type": "application/json", "Authorization": f"Bearer {key}"}
        self.model_name = model_name

    def _compute_rank(self, query: str, texts: List) -> Tuple[np.ndarray, int]:
        data = {
            "model": self.model_name,
            "query": query,
            "documents": texts,
            "top_n": len(texts),
        }
        token_count = sum(num_tokens_from_string(t) for t in texts)
        response = requests.post(self._base_url + "/rerank", headers=self.headers, json=data, timeout=30)
        response.raise_for_status()
        res = response.json()
        rank = np.zeros(len(texts), dtype=float)
        try:
            for d in res.get("results", []):
                rank[d["index"]] = d["relevance_score"]
        except Exception as _e:
            log_exception(_e, res)
        return rank, token_count


class NewAPIRerank(Base):
    _FACTORY_NAME = "New API"

    def __init__(self, key, model_name, base_url, max_token=MAX_RERANK_TOKEN):
        super().__init__(key, model_name, max_token)
        normalized_base_url = (base_url or "").strip()
        if "/rerank" in normalized_base_url:
            self.base_url = normalized_base_url.rstrip("/")
        else:
            self.base_url = urljoin(f"{normalized_base_url.rstrip('/')}/", "rerank").rstrip("/")
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}",
        }
        self.model_name = model_name.split("___")[0]

    def _compute_rank(self, query: str, texts: list):
        data = {
            "model": self.model_name,
            "query": query,
            "documents": texts,
            "top_n": len(texts),
        }
        token_count = sum(num_tokens_from_string(t) for t in texts)
        res = requests.post(self.base_url, headers=self.headers, json=data).json()
        rank = np.zeros(len(texts), dtype=float)
        try:
            for d in res["results"]:
                rank[d["index"]] = d["relevance_score"]
        except Exception as _e:
            log_exception(_e, res)
        return rank, token_count
