# ADR-001 — UI & production stack: decision and triggers

**Status:** Accepted (research phase) · revisit on the triggers below
**Date:** 2026-09-08
**Context owner:** Amir Hajdar (AgentiGrid / ExaGO)

## Context
AgentiGrid drives ExaGO OPF via an LLM loop, with a deterministic verifier as the
central thesis contribution and RAG as a subordinate generation-stage component.
The current UI is a **Streamlit** app (`launcher/app.py`). The question: is Streamlit
the production UI, or must it become a "modern"/industry-standard app — and if so,
when is that built?

Key facts:
- Streamlit is excellent for prototyping, demos, and committee/advisor walkthroughs,
  but is not suited to grid-operations production: whole-script-rerun model, weak
  multi-user concurrency, no real auth/RBAC, no clean API surface, UI coupled to logic.
- The engine is already **largely UI-agnostic**: `session_manager` bridges UI↔engine
  through a controller with `on_iteration`/`on_phase` callbacks + a queue, and both the
  CLI and the experiment harness drive the engine with no Streamlit involved.
- Correctness lives in the **deterministic verifier**, not the UI — so UI technology is
  orthogonal to thesis validity.

## Decision
1. **Keep Streamlit for the research phase.** It is the right tool for iteration and
   demonstration through the PhD.
2. **Do not build a production frontend now.** Defer it; it earns no thesis credit,
   requirements are still unstable (RAG variants, verifier), and it risks rework (YAGNI).
3. **Treat the UI as a thin client over a decoupled engine.** The product is the engine
   (agent_loop → ExaGO, verifier, RAG); any UI is one client of it. "Modernize the app"
   therefore means *add a new frontend against the same engine*, never rewrite the core.
4. **Protect the seam now (the one active obligation).** Keep business logic out of
   `app.py`; ensure everything the UI does is callable programmatically on the engine.
   This is near-zero cost and is what keeps a future frontend a frontend project.

## Target production architecture (for when a trigger fires)
- **Core engine** — UI-agnostic library/service; verifier and RAG live here, never in the UI.
- **Backend API — FastAPI** (Python std for ML/LLM services), with **WebSocket/SSE** for
  streaming iteration progress (maps directly onto the existing `on_iteration`/`on_phase`
  callbacks).
- **Job execution** — OPF searches are long-running, so they run in a task queue/worker
  (Celery/RQ/Dramatiq); on HPC the API submits to the cluster scheduler (SLURM) rather
  than running inline in a request.
- **Frontend — React/TypeScript (Next.js)** (or Vue) SPA over REST + WebSocket.
- **Production table stakes** — auth/authz, structured logging/observability,
  containerization, config/secrets management.

## Tiers of "production" (match effort to the tier actually needed)
- **T0 — research/demo (now):** Streamlit run locally. No change.
- **T1 — shared internal demo:** deploy *Streamlit itself*, containerized behind auth (or
  Streamlit Community Cloud). Hours of work, not a rewrite. Use this if a multi-user demo
  is needed before the PhD ends.
- **T2 — industry operations tool:** the FastAPI + task-queue/SLURM + React stack above.
  A team software effort with proper review — likely not solo-PhD work.

## Triggers to revisit (build T2 only when one is true)
- The ExaGO team commits to deploying AgentiGrid in an operations context.
- Real multi-user / concurrent usage appears (T1 no longer sufficient).
- An operations/regulatory requirement (audit, RBAC, SLAs) that Streamlit cannot meet.
- Post-PhD productization is funded/staffed.

## Consequences
- **Positive:** no wasted UI engineering during the thesis; the science is defensible on
  Streamlit; the eventual frontend is additive, not a rewrite; production UI choice never
  affects correctness (verifier is independent).
- **Negative / watch:** the decoupling must be *maintained* — every feature added to
  `app.py` is a chance to leak logic into the UI. Review new UI work against rule (4).
- **Note:** the `AGENTIGRID_RAG_MODE` switch already follows this pattern — one engine-level
  control, set by both the harness and the UI — which is the model for all UI↔engine
  interaction going forward.

## Options considered
- **A. Harden Streamlit into production.** Rejected — fights the framework's design
  (concurrency, auth, API); more effort than it looks, still not industry-grade.
- **B. Rewrite now as FastAPI + React.** Rejected — premature; no thesis value; builds on
  unstable requirements.
- **C. (Chosen) Streamlit now behind a clean engine seam; FastAPI + React later, trigger-based.**
