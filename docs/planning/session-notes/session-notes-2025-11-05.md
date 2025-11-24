# Session Notes - 2025-11-05

## What Happened
- Delivered a live telemetry surface: `/telemetry/snapshot` + `/telemetry/stream`, a console-friendly viewer (`scripts/live_telemetry.py`), and automatic console launch on server boot so hormone metrics, controller deltas, and behavioural thresholds stay visible during unattended runs.
- Added a server-side session reset hook and wired probes/UI to call it, ensuring every automated cycle starts fresh without restarting FastAPI or the llama engine.
- Hardened `scripts/continuous_probes.py` with duration guards (`--max-hours`, `--until`), concurrency locking, idle cool-downs, per-batch log rotation, and structured error handling/summaries after each run.
- Wrapped the whole overnight flow in `scripts/probes/run_overnight.cmd`, which cleans stale locks, spins up the server (telemetry in tow), waits for warm-up, then launches the guarded probe loop with configurable cadence/cutoffs.

## Keep in Mind for Future Runs
- Adjust `PROBE_UNTIL`, `PROBE_MAX_HOURS`, profiles, and cadence in `scripts/probes/run_overnight.cmd` before launching; defaults stop at the earlier of 8 hours or 05:30 local time.
- The probe lock lives at `logs/probe_runs/.probe.lock`; if the process is interrupted unexpectedly, delete the file before starting a new injector.
- Telemetry now opens in a separate console whenever the server boots—close that window only after shutting down uvicorn, or it will restart on the next launch.
- Per-batch logs drop in `logs/probe_runs/probe_log_YYYYMMDD_HHMMSS_####.jsonl`; each file contains either the metrics or a captured exception record for that iteration.

## Next Steps
1. Kick off the overnight probe run via `scripts\probes\run_overnight.cmd` and review the per-batch JSONL metrics in the morning.
2. Promote the probe summaries into a lightweight dashboard (or ship to the existing logging stack) so alerting can happen without manual inspection.
3. Fold the lock/telemetry/run harness into CI tooling once the overnight workflow has run cleanly for a few nights.
