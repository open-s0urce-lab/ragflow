#
#  Copyright 2026 The InfiniFlow Authors. All Rights Reserved.
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

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import requests

import rag.llm.rerank_model as rerank_model
from rag.llm.rerank_model import GPUStackRerank

pytestmark = pytest.mark.p2


@pytest.fixture(autouse=True)
def _sequential_batches(monkeypatch):
    """Детерминированный порядок запросов в тестах батчинга."""
    monkeypatch.setattr(rerank_model, "RERANK_MAX_WORKERS", 1)


@pytest.mark.parametrize(
    ("gpustack_base_url", "expected_url"),
    [
        ("http://gpustack:80", "http://gpustack:80/v1/rerank"),
        ("http://gpustack:80/v1", "http://gpustack:80/v1/rerank"),
        ("http://gpustack:80/v1/rerank", "http://gpustack:80/v1/rerank"),
    ],
)
def test_gpustack_rerank_uses_single_versioned_endpoint(gpustack_base_url, expected_url):
    response = MagicMock()
    response.raise_for_status.return_value = None
    response.json.return_value = {"results": [{"index": 0, "relevance_score": 0.8}, {"index": 1, "relevance_score": 0.2}]}

    with patch("rag.llm.rerank_model.requests.post", return_value=response) as post, patch("rag.llm.rerank_model.num_tokens_from_string", return_value=1):
        rank, token_count = GPUStackRerank("sk-test", "qwen3-reranker-4b", gpustack_base_url).similarity("query", ["first", "second"])

    post.assert_called_once()
    assert post.call_args.args[0] == expected_url
    assert post.call_args.kwargs["json"]["model"] == "qwen3-reranker-4b"
    assert token_count == 2
    assert np.allclose(rank, [0.8, 0.2])


def _response_for(batch_size, offset=0.1):
    response = MagicMock()
    response.raise_for_status.return_value = None
    response.json.return_value = {
        "results": [
            {"index": index, "relevance_score": offset + index * 0.1}
            for index in range(batch_size)
        ]
    }
    return response


def test_gpustack_rerank_splits_documents_into_context_sized_batches():
    """64 кандидатов по ~512 токенов не влезают в 8192-токенный контекст одним запросом."""
    texts = ["first", "second", "third", "fourth"]
    token_sizes = {"query": 10}
    calls = []

    def fake_post(url, json=None, headers=None, timeout=None):
        calls.append(json)
        return _response_for(len(json["documents"]))

    with patch("rag.llm.rerank_model.requests.post", side_effect=fake_post) as post, patch(
        "rag.llm.rerank_model.num_tokens_from_string",
        side_effect=lambda text: token_sizes.get(text, 30),
    ):
        model = GPUStackRerank("sk-test", "bge-reranker-v2-m3-en-ru", "http://gpustack:80", max_token=100)
        rank, token_count = model._compute_rank("query", texts)

    assert post.call_count == 2
    assert [len(call["documents"]) for call in calls] == [2, 2]
    assert [call["top_n"] for call in calls] == [2, 2]
    assert [call["documents"] for call in calls] == [["first", "second"], ["third", "fourth"]]
    assert token_count == 120
    assert np.allclose(rank, [0.1, 0.2, 0.1, 0.2])


def test_gpustack_rerank_keeps_scores_on_original_positions_across_batches():
    texts = ["first", "second", "third", "fourth"]

    def fake_post(url, json=None, headers=None, timeout=None):
        base = 0.1 if json["documents"][0] == "first" else 0.9
        return _response_for(len(json["documents"]), offset=base)

    with patch("rag.llm.rerank_model.requests.post", side_effect=fake_post), patch(
        "rag.llm.rerank_model.num_tokens_from_string",
        side_effect=lambda text: 10 if text == "query" else 30,
    ):
        model = GPUStackRerank("sk-test", "bge-reranker-v2-m3-en-ru", "http://gpustack:80", max_token=100)
        rank, _ = model._compute_rank("query", texts)

    assert np.allclose(rank, [0.1, 0.2, 0.9, 1.0])


def test_gpustack_rerank_truncates_oversized_document_to_batch_budget():
    def fake_post(url, json=None, headers=None, timeout=None):
        return _response_for(len(json["documents"]))

    with patch("rag.llm.rerank_model.requests.post", side_effect=fake_post) as post, patch(
        "rag.llm.rerank_model.num_tokens_from_string",
        side_effect=lambda text: {"query": 10, "huge": 200}.get(text, 5),
    ), patch("rag.llm.rerank_model.truncate", return_value="short") as truncate_mock:
        model = GPUStackRerank("sk-test", "bge-reranker-v2-m3-en-ru", "http://gpustack:80", max_token=100)
        model._compute_rank("query", ["huge"])

    truncate_mock.assert_called_once_with("huge", 70)
    assert post.call_args.kwargs["json"]["documents"] == ["short"]


def _context_400():
    response = MagicMock()
    error = requests.exceptions.HTTPError("400 Client Error: context length")
    error.response = MagicMock(status_code=400)
    response.raise_for_status.side_effect = error
    return response


def test_gpustack_rerank_halves_batch_rejected_with_400():
    """Если сервер отверг пачку по длине контекста, режем её пополам, а не роняем чат."""
    texts = ["first", "second"]
    calls = []

    def fake_post(url, json=None, headers=None, timeout=None):
        assert json is not None
        calls.append(list(json["documents"]))
        if len(json["documents"]) > 1:
            return _context_400()
        return _response_for(1)

    with patch("rag.llm.rerank_model.requests.post", side_effect=fake_post), patch(
        "rag.llm.rerank_model.num_tokens_from_string",
        side_effect=lambda text: 10 if text == "query" else 30,
    ):
        model = GPUStackRerank("sk-test", "bge-reranker-v2-m3-en-ru", "http://gpustack:80", max_token=200)
        rank, token_count = model._compute_rank("query", texts)

    assert calls == [["first", "second"], ["first"], ["second"]]
    assert np.allclose(rank, [0.1, 0.1])
    assert token_count == 60


def test_gpustack_rerank_shortens_single_document_rejected_with_400():
    calls = []

    def fake_post(url, json=None, headers=None, timeout=None):
        assert json is not None
        document = json["documents"][0]
        calls.append(document)
        if len(document) > 10:
            return _context_400()
        return _response_for(1)

    with patch("rag.llm.rerank_model.requests.post", side_effect=fake_post), patch(
        "rag.llm.rerank_model.num_tokens_from_string",
        side_effect=lambda text: 10 if text == "query" else 1,
    ):
        model = GPUStackRerank("sk-test", "bge-reranker-v2-m3-en-ru", "http://gpustack:80", max_token=100)
        rank, _ = model._compute_rank("query", ["x" * 20])

    assert len(calls) > 1
    assert all(len(call) < len(calls[0]) for call in calls[1:])
    assert len(calls[-1]) <= 10
    assert np.allclose(rank, [0.1])


def test_gpustack_rerank_caps_document_before_batching(monkeypatch):
    monkeypatch.setattr(rerank_model, "RERANK_DOC_TOKEN_CAP", 10)

    def fake_post(url, json=None, headers=None, timeout=None):
        assert json is not None
        return _response_for(len(json["documents"]))

    with patch("rag.llm.rerank_model.requests.post", side_effect=fake_post), patch(
        "rag.llm.rerank_model.num_tokens_from_string",
        side_effect=lambda text: 50 if text == "long" else 5,
    ), patch("rag.llm.rerank_model.truncate", return_value="capped") as truncate_mock:
        model = GPUStackRerank("sk-test", "bge-reranker-v2-m3-en-ru", "http://gpustack:80", max_token=1000)
        model._compute_rank("query", ["long"])

    truncate_mock.assert_called_once_with("long", 10)


def test_gpustack_rerank_scores_every_batch_when_parallel(monkeypatch):
    monkeypatch.setattr(rerank_model, "RERANK_MAX_WORKERS", 4)
    texts = [f"doc{index}" for index in range(8)]

    def fake_post(url, json=None, headers=None, timeout=None):
        assert json is not None
        return _response_for(len(json["documents"]))

    with patch("rag.llm.rerank_model.requests.post", side_effect=fake_post) as post, patch(
        "rag.llm.rerank_model.num_tokens_from_string",
        side_effect=lambda text: 10 if text == "query" else 30,
    ):
        model = GPUStackRerank("sk-test", "bge-reranker-v2-m3-en-ru", "http://gpustack:80", max_token=100)
        rank, _ = model._compute_rank("query", texts)

    assert post.call_count == 4
    assert np.all(rank > 0), rank
