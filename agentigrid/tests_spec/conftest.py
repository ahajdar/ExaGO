"""Shared configuration for the independent, spec-derived test suite.

Ownership: QA (see specs/QA_PROCESS.md). The implementer must not edit files
in tests_spec/ or specs/.

Guardrails enforced here:
  * A skipped test FAILS. Missing data or dependencies must be fixed, not
    silently skipped (the legacy suite skips ~17% of its tests this way).
  * xfail is not allowed. A test that is "expected to fail" is a weakened test.
  * Every test must carry a spec ID (SPEC-<AREA>-<NN>) in its docstring or via
    @pytest.mark.spec("SPEC-<AREA>-<NN>"), so results trace back to the spec.
  * Hypothesis runs are deterministic (derandomized, no deadline, no database).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from hypothesis import HealthCheck, settings

# Deterministic Hypothesis runs (QA rule: fixed seeds, no timing assertions):
# derandomized examples, no deadline, and no example database carried between runs.
settings.register_profile(
    "qa",
    derandomize=True,
    deadline=None,
    database=None,
    max_examples=60,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)
settings.load_profile("qa")

SPEC_ID = re.compile(r"\bSPEC-[A-Z][A-Z0-9_]*-\d{2,}\b")

TESTS_SPEC_DIR = Path(__file__).resolve().parent
DATA_DIR = TESTS_SPEC_DIR / "data"  # all test data lives inside tests_spec/


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "spec(id): spec item(s) this test verifies, e.g. SPEC-PARSER-03")


def _spec_ids(item: pytest.Item) -> list[str]:
    ids: list[str] = []
    for mark in item.iter_markers(name="spec"):
        ids.extend(str(a) for a in mark.args)
    doc = getattr(getattr(item, "function", None), "__doc__", None) or ""
    ids.extend(SPEC_ID.findall(doc))
    return ids


def _in_spec_suite(item: pytest.Item) -> bool:
    return TESTS_SPEC_DIR in Path(str(item.path)).resolve().parents


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    # This hook sees the whole session (e.g. legacy tests/ too), so scope it.
    items = [item for item in items if _in_spec_suite(item)]
    missing = [item.nodeid for item in items if not _spec_ids(item)]
    if missing:
        listing = "\n  ".join(missing)
        raise pytest.UsageError(
            f"{len(missing)} test(s) in tests_spec/ have no spec ID "
            f"(add SPEC-<AREA>-<NN> to the docstring or @pytest.mark.spec):\n  {listing}"
        )
    for item in items:
        if item.get_closest_marker("xfail"):
            raise pytest.UsageError(f"xfail is not allowed in tests_spec/: {item.nodeid}")


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo):
    outcome = yield
    report = outcome.get_result()
    if report.skipped:
        reason = report.longrepr[2] if isinstance(report.longrepr, tuple) else str(report.longrepr)
        report.outcome = "failed"
        report.longrepr = f"Skipped tests are not allowed in tests_spec/ (fix the cause): {reason}"
    report.user_properties.append(("spec_ids", ",".join(_spec_ids(item))))


@pytest.fixture
def data_dir() -> Path:
    """Directory for small, committed test fixtures (synthetic MATPOWER cases etc.)."""
    return DATA_DIR
