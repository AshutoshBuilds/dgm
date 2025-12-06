# DGM Enhancements PRD: Selection, Archive, Retrieval, Motivation, Auto-Debug, Preflight, Local HF Models

## Overview
This PRD defines incremental enhancements to the Darwin Gödel Machine (DGM) to improve search quality, stability, and reuse of prior knowledge while preserving the current architecture and developer ergonomics. All changes are opt-in via flags or environment variables with sensible defaults to avoid regressions.

## Goals
- Improve parent selection with UCT/score-informed strategies.
- Maintain a high-quality, bounded archive (top-k) to reduce noise and drift.
- Leverage prior attempts via lightweight local retrieval (FAISS) to guide diagnosis and improvements.
- Persist a short “motivation” string per attempt for retrieval and auditing.
- Increase robustness with one-pass auto-debug retries and preflight checks.
- Support running fully offline/locally using Hugging Face models.

## Non-goals
- No architectural rewrite (no external DB/RAG services).
- No changes to existing successful defaults unless a flag is provided.
- No change to SWE-bench/Polyglot harness behavior besides optional preflight and auto-debug.

## User Stories & Requirements
- As a researcher, I can choose parent commits using UCT to balance exploration/exploitation.
- As an operator, I can cap the archive to top-k runs by accuracy to keep only best variants.
- As an agent author, I can reuse prior problems/motivations/patches via retrieval to improve prompts.
- As a maintainer, I can see a short “motivation” text persisted in `metadata.json` for each run.
- As a user, if an evaluation fails to compile, the system attempts a single quick auto-debug then retries.
- As a user, the system can run an inexpensive preflight check to catch obvious failures early.
- As a privacy-sensitive user, I can set `CODE_MODEL=hf/<repo_id>` to run the agent locally.

## Functional Spec
- Parent selection
  - New method: `--choose_selfimproves_method uct`.
  - Implementation computes UCB1 per candidate using historical success stats from `metadata.json` of archive nodes.
  - Keep existing methods; default unchanged.
- Archive policy
  - New update strategy: `--update_archive keep_top_k` with `--top_k <int>`.
  - Uses `overall_performance.accuracy_score` to retain best K, deterministic tie-break by timestamp.
- Retrieval (RAG-lite)
  - Flags: `--use_retrieval`, `--retrieval_k <int=5>`, `--retrieval_fields problem,motivation,patch`.
  - Build/maintain a local FAISS index under `output_dgm/retrieval_index/` with `sentence-transformers` (e5-base small or similar).
  - On `diagnose_problem` and `diagnose_improvement`, fetch top-k similar contexts and prepend to prompts.
  - Opt-in only; disabled by default.
- Motivation
  - After diagnosis, request a short `motivation` (≤300 chars) and persist in `metadata.json`.
  - Include in retrieval corpus if enabled.
- Auto-debug retry
  - Flags: `--enable_auto_debug`, `--auto_debug_max_rounds 1`, `--auto_debug_timeout_sec 120`.
  - If compile/eval fails, capture logs, run a brief repair agent step, apply the small patch, retry once.
- Preflight checks
  - Flag: `--preflight_checks`.
  - Inside container, run a quick `python -m py_compile` for repo, import smoke tests, or `pytest -q -k smoke` if available.
- Local HF models
  - `CODE_MODEL=hf/<repo_id>` and `DIAGNOSE_MODEL=hf/<repo_id>` supported.
  - Uses `transformers` pipeline with CPU by default; respects `device_map="auto"`.

## CLI & Env Additions
- `DGM_outer.py`
  - `--choose_selfimproves_method uct`
  - `--update_archive keep_top_k` and `--top_k <int>`
  - `--use_retrieval` `--retrieval_k <int>` `--retrieval_fields <csv>`
  - `--enable_auto_debug` `--auto_debug_max_rounds <int>` `--auto_debug_timeout_sec <int>`
  - `--preflight_checks`
- Environment
  - `CODE_MODEL` (default: current CLAUDE)
  - `DIAGNOSE_MODEL` (default: current o1)

## Technical Design
- Selection: Extend `choose_selfimproves` to compute UCB1 from per-parent stats: successes (resolved), attempts (children_count), with small epsilon for unseen.
- Archive: Implement `keep_top_k` branch in `update_archive`; read scores from each child `metadata.json`.
- Retrieval:
  - New `utils/retrieval.py` to handle embedding/indexing/search (faiss-cpu + sentence-transformers).
  - Ingestion at end of each attempt; query at diagnosis points.
  - Store index files under `output_dgm/retrieval_index/` and rebuild incrementally.
- Motivation: Prompt appended to diagnosis; save `metadata['motivation']`.
- Auto-debug: Wrap `run_harness_*` with error capture → short tool-enabled repair → re-run harness once.
- Preflight: Before harness, run quick checks and short-circuit failure early.

## Data Model Changes (metadata.json)
- Add fields: `motivation: str`, `retrieval_context_ids: [str]`, `auto_debug_attempted: bool`, `preflight_passed: bool`.

## Dependencies
- New (optional): `faiss-cpu`, `sentence-transformers` (small e5-base), kept behind `--use_retrieval`.
- Existing HF local support via `transformers`, `torch`, `accelerate`, `safetensors`.

## Performance & Safety
- All new features off by default.
- Retrieval index bounded by top-k archive if `keep_top_k` is enabled.
- Auto-debug limited to one short round; logs persisted.

## Rollout Plan
1) Smoke-test local HF model end-to-end on a single self-improve attempt (small subset) to validate offline path.
2) Implement UCT selection + keep_top_k archive.
3) Add retrieval (index + query) with flags, default off.
4) Add auto-debug + preflight (flags), default off.

## Success Criteria
- No regression in default mode on the same seeds/settings.
- With `uct+keep_top_k`, faster convergence on small/medium subsets.
- With retrieval, higher fix rate on previously failed IDs.
- With auto-debug, fewer compile/eval aborts.

## Risks & Mitigations
- Heavy HF models on CPU → recommend small instruct models, allow `device_map` override.
- Retrieval prompting increases token usage → keep contexts short, limit to K, use summaries.
- Auto-debug could produce noisy patches → single attempt, bounded time, patch size limit.

## Open Questions
- Preferred default small HF model for CPU? (e.g., `TinyLlama/TinyLlama-1.1B-Chat-v1.0`).
- Any constraints on adding `faiss-cpu` and `sentence-transformers` to deps versus optional extras?
