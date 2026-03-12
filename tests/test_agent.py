#!/usr/bin/env python
"""
Tests for TimeMixerAgent.

Tests cover:
1. Agent initialization (model-only, no-model, full config)
2. Single-sample prediction (y1 only, fallback modes)
3. Batch prediction
4. AgentResult structure and serialization
5. Agent status reporting
6. Input validation
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
import numpy as np
import torch
import tempfile
import os

from timemixerpp import TimeMixerPPConfig, TimeMixerPPForBinaryCls, TimeMixerAgent, AgentResult


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def simple_config():
    return TimeMixerPPConfig(seq_len=48, d_model=32, n_layers=1, top_k=2)


@pytest.fixture(scope="module")
def simple_model(simple_config):
    model = TimeMixerPPForBinaryCls(simple_config)
    # Trigger lazy parameter init
    with torch.no_grad():
        _ = model(torch.randn(1, 48))
    model.eval()
    return model


@pytest.fixture(scope="module")
def checkpoint_path(simple_config, simple_model, tmp_path_factory):
    """Save a minimal checkpoint and return its path."""
    tmp_dir = tmp_path_factory.mktemp("checkpoints")
    ckpt_path = str(tmp_dir / "test_model.pt")
    torch.save(
        {
            "model_state_dict": simple_model.state_dict(),
            "config": {
                "seq_len": simple_config.seq_len,
                "c_in": simple_config.c_in,
                "d_model": simple_config.d_model,
                "n_layers": simple_config.n_layers,
                "n_heads": simple_config.n_heads,
                "top_k": simple_config.top_k,
                "dropout": simple_config.dropout,
            },
        },
        ckpt_path,
    )
    return ckpt_path


@pytest.fixture
def sample_x():
    """A single 48-dimensional input vector."""
    return np.random.randn(48).astype(np.float32)


@pytest.fixture
def sample_X():
    """A batch of 10 samples."""
    return np.random.randn(10, 48).astype(np.float32)


# ─── Agent Initialization ─────────────────────────────────────────────────────

class TestAgentInitialization:

    def test_model_only_agent(self, simple_model):
        """Agent can be created with just a model (no RAG, no LLM)."""
        agent = TimeMixerAgent(model=simple_model)
        assert agent.has_model
        assert not agent.has_rag
        assert not agent.has_llm

    def test_no_component_agent(self):
        """Agent can be created with no components (all disabled)."""
        agent = TimeMixerAgent()
        assert not agent.has_model
        assert not agent.has_rag
        assert not agent.has_llm

    def test_from_checkpoint(self, checkpoint_path):
        """Agent can be loaded from a checkpoint file."""
        agent = TimeMixerAgent.from_checkpoint(checkpoint_path, device="cpu")
        assert agent.has_model
        assert not agent.has_rag
        assert not agent.has_llm

    def test_rag_disabled_without_prefix(self):
        """RAG is disabled when collection_prefix is missing."""
        agent = TimeMixerAgent(
            qdrant_url="http://localhost:6333",
            collection_prefix=None,
        )
        assert not agent.has_rag

    def test_rag_disabled_without_url(self):
        """RAG is disabled when qdrant_url is missing."""
        agent = TimeMixerAgent(
            qdrant_url=None,
            collection_prefix="temperature_kb",
        )
        assert not agent.has_rag

    def test_llm_disabled_when_mode_none(self, simple_model):
        """LLM is always disabled when llm_mode='none'."""
        agent = TimeMixerAgent(
            model=simple_model,
            ollama_url="http://localhost:11434",
            llm_mode="none",
        )
        assert not agent.has_llm

    def test_device_cpu(self, simple_model):
        """Agent respects explicit device setting."""
        agent = TimeMixerAgent(model=simple_model, device="cpu")
        assert agent.device == torch.device("cpu")

    def test_status_dict(self, simple_model):
        """Agent.status() returns a dict with three boolean keys."""
        agent = TimeMixerAgent(model=simple_model)
        status = agent.status()
        assert isinstance(status, dict)
        assert len(status) == 3
        assert all(isinstance(v, bool) for v in status.values())


# ─── Single-Sample Prediction ─────────────────────────────────────────────────

class TestSingleSamplePrediction:

    def test_predict_returns_agent_result(self, simple_model, sample_x):
        """predict() returns an AgentResult."""
        agent = TimeMixerAgent(model=simple_model)
        result = agent.predict(sample_x)
        assert isinstance(result, AgentResult)

    def test_predict_probability_in_range(self, simple_model, sample_x):
        """Final probability is in [0, 1]."""
        agent = TimeMixerAgent(model=simple_model)
        result = agent.predict(sample_x)
        assert 0.0 <= result.probability <= 1.0

    def test_predict_prediction_binary(self, simple_model, sample_x):
        """Prediction is 0 or 1."""
        agent = TimeMixerAgent(model=simple_model)
        result = agent.predict(sample_x)
        assert result.prediction in (0, 1)

    def test_predict_confidence_in_range(self, simple_model, sample_x):
        """Confidence is in [0, 1]."""
        agent = TimeMixerAgent(model=simple_model)
        result = agent.predict(sample_x)
        assert 0.0 <= result.confidence <= 1.0

    def test_predict_y1_populated(self, simple_model, sample_x):
        """y1 is populated when a model is present."""
        agent = TimeMixerAgent(model=simple_model)
        result = agent.predict(sample_x)
        assert result.y1 is not None
        assert 0.0 <= result.y1 <= 1.0

    def test_predict_no_y2_without_rag(self, simple_model, sample_x):
        """y2 is None when RAG is not configured."""
        agent = TimeMixerAgent(model=simple_model)
        result = agent.predict(sample_x)
        assert result.y2 is None

    def test_predict_no_y3_without_llm(self, simple_model, sample_x):
        """y3 is None when LLM is not configured."""
        agent = TimeMixerAgent(model=simple_model)
        result = agent.predict(sample_x)
        assert result.y3 is None

    def test_predict_list_input(self, simple_model):
        """predict() accepts a Python list."""
        agent = TimeMixerAgent(model=simple_model)
        x_list = [float(np.random.randn()) for _ in range(48)]
        result = agent.predict(x_list)
        assert isinstance(result, AgentResult)

    def test_predict_invalid_dimension_raises(self, simple_model):
        """predict() raises ValueError for wrong input dimension."""
        agent = TimeMixerAgent(model=simple_model)
        with pytest.raises(ValueError, match="48"):
            agent.predict(np.random.randn(32))

    def test_predict_threshold_effect(self, simple_model, sample_x):
        """Changing threshold changes prediction for borderline cases."""
        agent_low = TimeMixerAgent(model=simple_model, threshold=0.01)
        agent_high = TimeMixerAgent(model=simple_model, threshold=0.99)
        result_low = agent_low.predict(sample_x)
        result_high = agent_high.predict(sample_x)
        # With threshold=0.01 almost all → prediction=1
        assert result_low.prediction == 1
        # With threshold=0.99 almost all → prediction=0
        assert result_high.prediction == 0

    def test_predict_with_normalizer(self, simple_model, sample_x):
        """Agent applies normalizer when mean/std are provided."""
        mean = np.zeros(48, dtype=np.float32)
        std = np.ones(48, dtype=np.float32)
        agent = TimeMixerAgent(model=simple_model, normalizer_mean=mean, normalizer_std=std)
        result = agent.predict(sample_x)
        assert isinstance(result, AgentResult)
        assert 0.0 <= result.probability <= 1.0


# ─── Batch Prediction ─────────────────────────────────────────────────────────

class TestBatchPrediction:

    def test_predict_batch_length(self, simple_model, sample_X):
        """predict_batch() returns one result per input sample."""
        agent = TimeMixerAgent(model=simple_model)
        results = agent.predict_batch(sample_X)
        assert len(results) == len(sample_X)

    def test_predict_batch_all_valid(self, simple_model, sample_X):
        """All batch results are valid AgentResult instances."""
        agent = TimeMixerAgent(model=simple_model)
        results = agent.predict_batch(sample_X)
        for r in results:
            assert isinstance(r, AgentResult)
            assert 0.0 <= r.probability <= 1.0
            assert r.prediction in (0, 1)

    def test_predict_batch_with_sample_ids(self, simple_model, sample_X):
        """predict_batch() accepts sample_ids without error."""
        agent = TimeMixerAgent(model=simple_model)
        ids = list(range(len(sample_X)))
        results = agent.predict_batch(sample_X, sample_ids=ids)
        assert len(results) == len(sample_X)

    def test_predict_batch_list_input(self, simple_model):
        """predict_batch() accepts a list of lists."""
        agent = TimeMixerAgent(model=simple_model)
        X_list = [np.random.randn(48).tolist() for _ in range(3)]
        results = agent.predict_batch(X_list)
        assert len(results) == 3


# ─── AgentResult ─────────────────────────────────────────────────────────────

class TestAgentResult:

    def test_to_dict_keys(self, simple_model, sample_x):
        """to_dict() contains expected keys."""
        agent = TimeMixerAgent(model=simple_model)
        result = agent.predict(sample_x)
        d = result.to_dict()
        for key in ["probability", "prediction", "confidence", "fusion_mode"]:
            assert key in d

    def test_to_dict_probability_value(self, simple_model, sample_x):
        """to_dict() probability matches result.probability (rounded)."""
        agent = TimeMixerAgent(model=simple_model)
        result = agent.predict(sample_x)
        d = result.to_dict()
        assert d["probability"] == round(result.probability, 4)

    def test_repr_contains_probability(self, simple_model, sample_x):
        """repr() includes the probability value."""
        agent = TimeMixerAgent(model=simple_model)
        result = agent.predict(sample_x)
        repr_str = repr(result)
        assert str(round(result.probability, 4)) in repr_str

    def test_repr_no_error(self, simple_model, sample_x):
        """repr() runs without error."""
        agent = TimeMixerAgent(model=simple_model)
        result = agent.predict(sample_x)
        _ = repr(result)  # should not raise

    def test_default_fields(self):
        """AgentResult can be constructed with only required fields."""
        result = AgentResult(probability=0.7, prediction=1, confidence=0.4)
        assert result.y1 is None
        assert result.y2 is None
        assert result.y3 is None
        assert result.llm_explanation is None
        assert isinstance(result.details, dict)


# ─── From-checkpoint Integration ─────────────────────────────────────────────

class TestFromCheckpoint:

    def test_from_checkpoint_predict(self, checkpoint_path, sample_x):
        """Full round-trip: load from checkpoint, run predict."""
        agent = TimeMixerAgent.from_checkpoint(checkpoint_path, device="cpu")
        result = agent.predict(sample_x)
        assert isinstance(result, AgentResult)
        assert result.y1 is not None

    def test_from_checkpoint_batch(self, checkpoint_path, sample_X):
        """Batch predict works with checkpoint-loaded agent."""
        agent = TimeMixerAgent.from_checkpoint(checkpoint_path, device="cpu")
        results = agent.predict_batch(sample_X)
        assert len(results) == len(sample_X)

    def test_from_checkpoint_with_normalizer(self, checkpoint_path, tmp_path, sample_x):
        """Checkpoint with normalizer stats is applied correctly."""
        mean = np.zeros(48, dtype=np.float32)
        std = np.ones(48, dtype=np.float32)

        # Save checkpoint with normalizer stats
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        ckpt["normalizer_mean"] = mean
        ckpt["normalizer_std"] = std
        ckpt_path = str(tmp_path / "ckpt_with_norm.pt")
        torch.save(ckpt, ckpt_path)

        agent = TimeMixerAgent.from_checkpoint(ckpt_path, device="cpu")
        assert agent.normalizer_mean is not None
        assert agent.normalizer_std is not None

        result = agent.predict(sample_x)
        assert isinstance(result, AgentResult)


# ─── Package-level imports ────────────────────────────────────────────────────

class TestPackageExports:

    def test_agent_exported_from_package(self):
        """TimeMixerAgent is exported from the top-level package."""
        from timemixerpp import TimeMixerAgent
        assert TimeMixerAgent is not None

    def test_agent_result_exported_from_package(self):
        """AgentResult is exported from the top-level package."""
        from timemixerpp import AgentResult
        assert AgentResult is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
