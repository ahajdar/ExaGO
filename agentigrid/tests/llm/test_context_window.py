"""Ollama context window: num_ctx is sent when configured, and a call whose
prompt may not fit the window is flagged, counted and fails the run."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from agentigrid.backends import LLMBackend, LLMResponse, UsageMeter
from agentigrid.config import LLMConfig, load_config

ollama = pytest.importorskip("ollama")


def _cfg(**over) -> LLMConfig:
    base = dict(backend="ollama", model="qwen2.5:7b", api_key_env="X", openai_base_url=None,
                ollama_host="http://localhost:11434", ollama_cloud_host=None,
                temperature=0.3, max_tokens=256)
    base.update(over)
    return LLMConfig(**base)


class _FakeClient:
    def __init__(self, prompt_eval=100, eval_count=10):
        self.kwargs = None
        self.pe, self.ec = prompt_eval, eval_count

    def chat(self, **kw):
        self.kwargs = kw
        return {"message": {"content": '{"action": "complete"}'},
                "prompt_eval_count": self.pe, "eval_count": self.ec}


def _backend(cfg, client):
    from agentigrid.backends.ollama_backend import OllamaBackend
    b = OllamaBackend(cfg)
    b._client = client
    return b


def test_context_tokens_env_override(monkeypatch):
    monkeypatch.setenv("AGENTIGRID_CONTEXT_TOKENS", "32768")
    assert load_config(None).llm.context_tokens == 32768
    monkeypatch.setenv("AGENTIGRID_CONTEXT_TOKENS", "big")
    assert load_config(None).llm.context_tokens is None
    monkeypatch.delenv("AGENTIGRID_CONTEXT_TOKENS")
    assert load_config(None).llm.context_tokens is None


def test_num_ctx_sent_only_when_configured():
    c = _FakeClient()
    _backend(_cfg(context_tokens=32768), c).complete("sys", "user")
    assert c.kwargs["options"]["num_ctx"] == 32768
    c = _FakeClient()
    _backend(_cfg(), c).complete("sys", "user")
    assert "num_ctx" not in c.kwargs["options"]


def test_long_prompt_without_context_is_flagged():
    long_sys = "x" * 31557                        # the PFLOW system prompt size
    r = _backend(_cfg(), _FakeClient()).complete(long_sys, "u" * 5000)
    assert r.context_overflow                       # > Ollama's 4,096 default
    r = _backend(_cfg(context_tokens=32768), _FakeClient()).complete(long_sys, "u" * 5000)
    assert not r.context_overflow


def test_window_filled_during_call_is_flagged():
    r = _backend(_cfg(context_tokens=8192), _FakeClient(prompt_eval=8000, eval_count=300)).complete("s", "u")
    assert r.context_overflow


def test_meter_counts_overflows_and_runner_fails_the_run(tmp_path):
    class _Stub(LLMBackend):
        def __init__(self, flags):
            self.flags = list(flags)

        def complete(self, system_prompt, user_prompt, temperature=None):
            return LLMResponse(raw_text="{}", json_data={}, json_error=None, model="m", backend="ollama",
                               prompt_tokens=5, completion_tokens=1, context_overflow=self.flags.pop(0))

        def name(self):
            return "ollama"

        def supports_json_mode(self):
            return False

    meter = UsageMeter(_Stub([False, True, True]))
    for _ in range(3):
        meter.complete("abc", "de")
    t = meter.totals()
    assert t["context_overflows"] == 2 and t["max_prompt_chars"] == 5

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "rag" / "tools"))
    from experiment_runner import llm_failure
    j = tmp_path / "journal.json"
    j.write_text(json.dumps({"llm_usage": {**t, "context_tokens": None}, "entries": []}))
    assert "context window" in llm_failure(j)
    j.write_text(json.dumps({"llm_usage": {**t, "context_overflows": 0}, "entries": []}))
    assert llm_failure(j) is None
