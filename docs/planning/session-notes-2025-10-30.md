# Session Notes - 2025-10-30

## Highlights
- Confirmed alignment direction: favour emergent behaviour via subsystem incentives while keeping hard safety guardrails; appended v0.17.3 addendum to `alignment.md`.
- Reorganised planning materials into a single cascade (`docs/planning/roadmap.md`) and archived historical roadmaps/updates.
- Discussed adopting `mistral-inference` with the base `Mistral-7B-v0.3.Q4_K_M.gguf` model to curb helper drift and rely on internal feedback loops.
- Logged need for enhanced diagnostics, behaviour probes, and safety instrumentation before changing the default model.

## Decisions & Assumptions
- Planning structure now lives under `docs/planning/`; archived files are reference-only.
- Emergent selfhood remains the guiding principle; conscious phrasing is treated as simulated, not literal cognition.
- Base model experimentation is exploratory; instruct pipeline remains available until metrics confirm improvement.
- Safety filters stay mandatory even while soft guardrails shift toward hormone and penalty-driven incentives.

## Follow-Ups
- Integrate the `mistral-inference` client into the engine stack and design probe comparisons (authenticity, assistant_drift, safety rates).
- Implement diagnostics harness covering hormones<->memory, router<->sampling, reinforcement feedback, and API endpoints.
- Build the voice guard module with penalty/regeneration feedback and diary-driven cravings loop.
- Standardise multi-cycle probe suite and establish rollback thresholds tied to behavioural metrics.
