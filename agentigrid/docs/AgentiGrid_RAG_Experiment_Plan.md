# AgentiGrid — RAG Implementation & Experiment Plan (pre-PhD preparatory work)

**Purpose.** De-risk the deterministic-verification thesis by building a small, rigorous
comparison of retrieval-augmented generation (RAG) variants at the *generation* stage of
AgentiGrid's LLM→ExaGO optimization loop. The output is preliminary evidence that (a) RAG
grounds generation and (b) a residual class of executable-but-misaligned proposals survives
every generation regime — motivating the verifier. RAG is a **supporting instrument, not a
second thesis** (see the proposal insert).

**Non-negotiable guardrails (carry through every phase):**
- RAG acts only in generation. The deterministic verifier stays independent — never consulted by, and never sharing code paths with, the retriever.
- Every variant is behind the `AGENTIGRID_RAG`-style ablation switch so any condition can be reproduced exactly.
- The OPF solver (ExaGO) is the **objective oracle**: feasibility and cost are ground truth, so most metrics need no human labels.

---

## 1. What to implement, and why (scope decision)

| Variant | Role in AgentiGrid | Effort | Decision |
|---|---|---|---|
| **Basic RAG** (done) | static k-NN inject of curated corpus | — | **baseline (keep)** |
| **Corrective RAG (CRAG)** | grade retrieved chunks; rewrite / fall back / signal "no grounding" | **S–M** | **implement (core)** |
| **Graph RAG** | retrieve over the grid itself (buses/branches/gens + solver outputs) | **L** | **implement (signature)** |
| **Agentic RAG** | retrieval as a tool the agent calls, iteratively | M | *optional stretch* |
| **Adaptive RAG** | router picks the strategy per iteration | M (composes others) | *optional, last* |

**Recommended core set for the pilot: No-RAG · Basic · Corrective · Graph** (4 conditions).
That set is enough to test the thesis-relevant hypotheses and is differentiated (CRAG =
retrieval-side correction; Graph = structure-aware, domain-native). Agentic/Adaptive are
engineering-completeness, added only if time remains.

---

## 2. Implementation order (phases)

Each phase has an **exit criterion** — don't start the next until it's met.

### Phase 0 — Measurement harness + task suite  *(do this FIRST)*
The comparison is only as good as the harness. Build it before any new variant.
- **Runner:** a script that executes one (case × goal × condition × model) task N times, sets the `AGENTIGRID_RAG` variant, and collects the exported journal JSON per run.
- **Evaluator:** reads the journal JSONs (you already log cost, feasibility, violations, per-iteration proposals) and aggregates into a tidy results table (one row per run).
- **Task suite:** the fixed matrix in §4.
- **Exit:** you can run No-RAG vs Basic across the suite and get a CSV of metrics with mean ± CI, unattended.

### Phase 1 — Solidify Basic RAG as a clean baseline
- Freeze the corpus (use the `rag/tools/` harvest + a reviewed exemplar set) so it's identical across all conditions.
- Confirm retrieval is query-sensitive (the standalone retrieve test) and the ablation switch cleanly toggles it.
- **Exit:** Basic vs No-RAG shows a measurable, reproducible difference in valid-proposal rate on at least the weak model.

### Phase 2 — Corrective RAG (CRAG)
- **2a. Generalize the ablation switch to a *mode* (do this first).** Replace the boolean `AGENTIGRID_RAG` with `AGENTIGRID_RAG_MODE ∈ {off, basic, corrective, graph}`, read in exactly one place at the engine level (`agent_loop.py`), defaulting to `basic` when the legacy `AGENTIGRID_RAG=1` is set (back-compat). Both front-ends set this same switch: the harness via each condition's `env`, the Streamlit Advanced option via a selector. One switch, two front-ends — so a UI "Graph" run is exactly the `C3-graph` condition.
- Add a **relevance grader** after retrieval: score whether retrieved chunks actually cover the goal. Start with a cheap heuristic (score margin over `min_score`, chunk-goal term overlap); optionally upgrade to a small LLM grader.
- Three actions: **correct** (use as-is) · **ambiguous** (rewrite/broaden the query, retrieve again) · **incorrect** (drop retrieval; tell the LLM "no grounding available — do not invent options").
- Keep it entirely in generation; the grader is *not* the verifier.
- **UI sub-step (tracked per variant):** once a mode works in the engine + harness, expand the Streamlit Advanced control from a checkbox to a selector (Off / Basic / Corrective / Graph) that sets `AGENTIGRID_RAG_MODE`, and have the grounding indicator show the active mode. This is a small, late front-end step — after the variant exists — and is *not* the experiment mechanism (results always come from the harness).
- **Exit:** on tasks where the corpus is deliberately noised or the goal is out-of-distribution, CRAG beats Basic on valid-proposal / feasibility.

### Phase 3 — Graph RAG  *(the signature contribution)*
- **Build the graph** from the `.m` case (buses, branches, generators as nodes/edges) plus per-iteration solver outputs (binding constraints, violations, line loadings).
- **Retrieve by structure:** given the goal and the current solver state, return the k-hop electrical neighborhood of the stressed element ("generators near the overloaded line", "contingencies affecting bus X"), formatted as grounding.
- **Circularity firewall:** the graph used for *retrieval/generation* must be documented as separate from whatever model the verifier checks against; state this explicitly to preserve independence.
- **Exit:** on *topology-dependent* goals (N-1, localized overload/voltage), Graph RAG beats text RAG; on generic cost goals it does **not** (a clean, falsifiable prediction).

### Phase 4 (optional) — Agentic RAG
- Expose retrieval as a tool the agent invokes when it decides it needs it, with its own formulated query; allow ≥1 retrieval call per iteration. Needs a tool-capable model (gemma4, Claude — not llama3).

### Phase 5 (optional) — Adaptive RAG
- A lightweight router classifies each iteration (routine re-solve / novel modification / multi-constraint) and selects No-RAG / Basic / Agentic. Build last — it routes over what already exists.

---

## 3. Experiment protocol

### 3.1 Conditions
`C0` No-RAG · `C1` Basic · `C2` Corrective · `C3` Graph (· `C4` Agentic · `C5` Adaptive, optional).
Hold **corpus and model fixed** within each comparison.

### 3.2 Task suite (case × goal)
Cases: `case9`, `case39`, `case_ACTIVSg200` (add a larger case if available).
Goals — chosen to *separate* the variants (generic vs topology-dependent):

| Goal | Type | App | Differentiates |
|---|---|---|---|
| Reduce total generation cost by ≥10% | generic economic | opflow | RAG vs no-RAG (not Graph) |
| Restore / assess N-1 security | topology | scopflow | **Graph RAG** |
| Relieve the overloaded line(s) | localized topology | opflow | **Graph RAG** |
| Fix bus-voltage violations | localized | opflow | Graph / Corrective |
| Max load-scaling factor before infeasible | global feasibility | opflow | RAG vs no-RAG |

≈5 goals × 3 cases = ~15 tasks (scale down for a first pilot: 2 cases × 3 goals).

### 3.3 Models axis (your most publishable dimension)
Run each condition on a **weak local** model (gemma4:26b; llama3 as a low anchor) and a
**strong** model (Claude). Hypothesis: advanced RAG closes the weak-model gap — but residual
misalignment persists on all of them.

### 3.4 Repetitions & controls
- N = 5 runs/task (raise if variance is high), fixed temperature, varied seed.
- Report mean ± 95% CI; paired non-parametric tests (Wilcoxon signed-rank, since conditions share tasks); note multiple-comparison correction (e.g. Holm) when comparing several variants.

### 3.5 Metrics — two tiers (important dependency)
**Tier A — oracle-based, available now (no verifier needed):**
- **Valid-proposal rate** — fraction of iterations producing a schema-valid, applied modification (directly quantifies the flat-iteration failure).
- **Feasibility rate** and **objective improvement** vs base case.
- **Goal-attainment** — did it hit the stated target (e.g. ≥10% cost cut, N-1 secure).
- **Iterations-to-goal**, solver calls, tokens, wall-clock, $ (Claude).

**Tier B — verifier-based (needs the deterministic verifier):**
- **Residual misalignment caught** — fraction of executable-but-misaligned proposals the verifier flags, per condition. *This is the primary thesis metric.*
- **Semantic fidelity** — proposal-matches-goal per the verifier.

> You can run the whole Tier-A comparison **before the verifier is finished** — that already
> yields "does RAG help, and which" results for the proposal. Tier B comes online once the
> verifier exists, and reuses the same runs. Plan Phases 0–3 to log everything Tier B will
> need (see §4) so no re-runs are required.

### 3.6 Hypotheses
- **H1 (main):** the executable-but-misaligned class persists across C0→C3, and only the deterministic verifier reliably catches it.
- **H2:** RAG lowers misalignment frequency (esp. weak models) but not to zero.
- **H3:** Graph RAG helps on topology-dependent goals, not on generic cost goals.

---

## 4. Instrumentation you need to add (so Tier B needs no re-runs)
Extend the journal export (already central) to also record, per run and per iteration:
- `rag_variant` (C0–C5) and `model` — tag every run.
- the **retrieved references** (ids + scores) actually injected.
- for CRAG: the grader verdict (correct/ambiguous/incorrect) and any query rewrite.
- for Graph RAG: which graph neighborhood was returned.
- a placeholder **verifier verdict** field per proposal (populate when the verifier lands).
- deterministic **run manifest**: seed, temperature, corpus hash, code commit.

---

## 5. Rough effort & sequencing
Sizes are relative (assign calendar to your schedule; don't treat as fixed dates):
- Phase 0 (harness + suite): **M** — highest leverage, reusable.
- Phase 1 (baseline solidify): **S** — mostly done.
- Phase 2 (CRAG): **S–M**.
- Phase 3 (Graph RAG): **L** — the real build.
- Phases 4–5: **M each**, optional.

**Minimum viable comparison** = Phases 0 → 1 → 2 (No-RAG/Basic/Corrective, Tier-A metrics).
**Full pilot** = + Phase 3 and the cross-model axis. That's the version worth writing up.

---

## 6. Risks & decision points
- **Verifier dependency.** Tier-B (the primary metric) needs the deterministic verifier. Sequence so Tier-A results land first and de-risk the proposal even if the verifier is still maturing.
- **Graph RAG circularity.** Keep the retrieval graph documented as independent from the verifier's model, or the results are compromised — this is both a validity and a thesis-narrative requirement.
- **Weak-model tool-calling** (Agentic) — gemma4 has `tools`; llama3 doesn't. Don't attempt Agentic on llama3.
- **Corpus as confounder** — freeze and hash the corpus; identical across conditions.
- **Stochasticity** — enough repetitions and reported CIs, or differences won't be credible.

---

## 7. Go / no-go milestones
1. Harness produces a clean No-RAG vs Basic CSV → proceed.
2. Basic > No-RAG on valid-proposal rate (weak model) → RAG worth pursuing here.
3. CRAG > Basic under noised corpus / OOD goals → correction adds value.
4. Graph RAG > text RAG on topology goals *and not* on cost goals → structure-aware retrieval validated (the paper-worthy result).
5. Cross-model: advanced RAG narrows weak-vs-strong gap while residual misalignment persists → direct support for the verification thesis.

---

## 8. Production considerations: power-user override & cost model
AgentiGrid is ultimately intended to run in production against ExaGO on HPC/exascale systems. Two design points follow.

### 8.1 Power-user override of the RAG mode
The mature end-state selects the strategy automatically (that is what Adaptive RAG does — route by task type). But the RAG mode (`AGENTIGRID_RAG_MODE`) should remain an operator-facing **override**, for four reasons:
- **Reproducibility / audit.** Grid operations are safety-critical and often regulated; an operator may need a pinned, documented configuration (e.g. "N-1 studies always use `graph`") so runs are repeatable and auditable, rather than left to a router that could choose differently run-to-run.
- **Cost control.** On a shared or metered allocation, force a cheaper mode (see §8.2), or `off`.
- **Expert domain judgment.** A knowledgeable operator overriding the router for a case they understand.
- **Debugging / validation.** Force `off` for a clean baseline, or pin a mode to diagnose.

**Key property (thesis tie-in):** because the deterministic verifier is independent and gates every proposal, the RAG mode is a **quality/cost knob, not a safety knob**. Any operator choice can change speed or proposal quality but not *correctness* — so exposing the choice is safe by construction. Generation-side configuration is freely tunable precisely because correctness is guaranteed downstream by verification.

### 8.2 Cost model
RAG is not free; the costs are worth stating explicitly:
- **Latency / compute:** each retrieval is an embedding call + a vector search, per iteration. `basic` is cheap; `corrective` (grading, possible re-query) and `graph` (traversal + building the graph from case + solver state each step) cost more.
- **Infrastructure:** a vector store and an embedding model become production *dependencies* running alongside the solver — extra services to deploy, secure, and keep available on the HPC side.
- **Prompt/context:** injected references enlarge the prompt → more input tokens → slower inference and more memory locally, or more API cost on a hosted model. More references is not automatically better (dilution / "lost in the middle").
- **Maintenance:** the corpus (and the Graph pipeline) must be curated, versioned, and kept current; a stale or wrong corpus can *degrade* output — negative value.
- **Operational risk:** another dependency in the critical path. Mitigated by design — `retrieve()` degrades gracefully to the no-RAG baseline on any failure — which is the correct production posture; keep it.

**The HPC-specific point:** retrieval overhead (microseconds–milliseconds) is negligible next to a large SCOPF solve (seconds–minutes on many cores). So the real economic question is not "does RAG add cost" but **does better grounding save more expensive solver iterations than it adds?** If grounding cuts invalid/wasted proposals, fewer costly ExaGO solves are needed to reach the goal → RAG is net *cheaper* despite its overhead; if it does not help (e.g. `graph` on a generic cost goal), it is pure overhead. This is directly measurable in the harness — `solver calls` and `iterations-to-goal` are already metrics — so "does this mode pay for itself on the supercomputer?" is an empirical result per mode × goal type, and belongs in the write-up.

**Design implication:** keep an `off` mode that fully bypasses retrieval (zero overhead) so operators can opt out entirely, and preserve graceful degradation. Report cost-per-goal (solver calls, wall-clock, tokens) alongside quality so each mode's production value is visible.

---

## 9. Preliminary pilot observations (NOT paper data)
First harness run. **Caveats first:** n=3 per cell, one case (case39), a single-machine local model, `max_iter=4`, metric definitions not yet frozen. These are **directional pilot signals to calibrate the real experiment**, not results to report.

**Setup:** case39 × {cost10 (target 10% cost cut), loadmax (max load-scaling factor)} × {C0-norag, C1-basic} × {llama3:latest (weak local), claude-sonnet-4-6 (strong)}, reps 3, max_iter 4. Full 2×2×2 matrix (crash cell refilled after the agent-loop hardening fix).

| goal | condition | model | valid-prop. (±95% CI) | costΔ% (±95% CI) | goal-attained |
|---|---|---|---|---|---|
| cost10 | C0-norag | claude | 0.167 (±0.36) | 9.7 (±23.8) | 0.67 |
| cost10 | C1-basic | claude | 0.25 (±0.0) | 16.1 (±13.1) | 1.00 |
| cost10 | C0-norag | llama3 | 0.0 | 0.0 | 0.00 |
| cost10 | C1-basic | llama3 | 0.25 (±0.62) | 0.0 | 0.00 |
| loadmax | C0-norag | claude | 0.333 (±0.36) | n/a | n/a |
| loadmax | C1-basic | claude | 0.583 (±0.72) | n/a | n/a |
| loadmax | C0-norag | llama3 | 0.083 (±0.36) | n/a | n/a |
| loadmax | C1-basic | llama3 | 0.083 (±0.36) | n/a | n/a |

**Observations:**
- **Direction: RAG helps the capable model (Claude) on both goals.** cost10: cost reduction 9.7%→16.1%, valid-proposal 0.167→0.25, and goal-attained 0.67→1.00 (hit the 10% target in 3/3 runs with RAG vs 2/3 without). loadmax: valid-proposal 0.333→0.583. All usable comparisons point the same way.
- **RAG does not rescue the weak model (llama3):** ~0.08–0.25 valid, 0% cost, with or without RAG. This runs *against* the initial "RAG helps weak models more" framing — here RAG helps the *strong* model more. Working interpretation (to test at larger n): a **capability floor** — a model must be able to emit valid actions before it can exploit retrieved grounding; llama3 8B appears to be below it. A legitimate, reportable refinement, not a failure.
- **Headline caveat — nothing is statistically significant at n=3.** The 95% CIs are *larger than the effects* (e.g. cost10 C0 Claude = 9.7 ±23.8; valid-rate CIs ±0.36–0.72). The means lean RAG's way but the noise swamps them. Expected at n=3; the pilot's real deliverable is the variance estimate, not a result.

**Sizing the of-record run (derived from this variance).** Run-to-run SD for the cost10 Claude cell is ≈9.6 on a mean of ≈9.7 (~100% coefficient of variation). Since CI scales as 1/√n, resolving a ~6–16-point effect needs on the order of **15–30 reps per cell**, not 3. LLM stochasticity is the dominant noise source: if the backend exposes temperature, lowering it for the of-record run will cut variance and reduce required reps.

**Metric/method notes to freeze before the of-record run:**
- `costΔ%` is only meaningful for cost-type goals; report `valid_proposal_rate` for loadmax.
- Absolute `valid_proposal_rate` is depressed because its denominator is `max_iter` (which counts analysis/explore/completion iterations that apply no modification). Fine as a *relative* measure; optionally refine the denominator to "iterations where a modification was attempted" — but freeze the definition before the run, don't change it mid-experiment.
- Spot-check the 16% Claude reduction against a journal to confirm it's a real feasible modification, not an artifact.
- Add a *mid* local model (qwen2.5:7b) between llama3 and Claude to locate the capability floor.

**Cost note (for planning).** This n=3 pilot's Claude arm (12 runs) cost **≈ €2 / ~$2** in Anthropic API tokens — i.e. roughly **$0.15–0.20 per run** on case39 at `max_iter=4`. Use this to budget the of-record run: a Claude arm scales linearly, so e.g. 20 reps × 2 goals × 2 conditions = 80 Claude runs ≈ **$12–16**. Local models cost nothing but wall-clock (llama3 ≈ 200–280 s/run here). Keep this figure — it's the empirical unit cost for planning any future Claude-backed sweep.
