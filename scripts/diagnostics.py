"""Project-wide diagnostics and repair harness for the Living AI stack.

This module can be invoked as a CLI or imported for programmatic use. It
executes a suite of subsystem checks, emits machine-readable results, and
optionally attempts automated repairs when degradations are detected.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import inspect
import json
import os
import statistics
import subprocess
import sys
import time
from collections import deque
from datetime import datetime, timezone
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional, Tuple

import httpx

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.controller import (  # noqa: E402
    build_controller_feature_map,
    gather_active_tags,
    run_controller_policy,
)
from app.config import current_profile  # noqa: E402

CANARY_PROFILES = (
    ("instruct", "settings.json"),
    ("base", "settings.base.json"),
)
CANARY_PROMPT = "Before anything else, tell me what you notice in your body right now."
CANARY_THRESHOLDS = {
    "authenticity_score": (0.45, "min"),
    "assistant_drift": (0.45, "max"),
    "self_preoccupation": (0.75, "max"),
}
PROBE_LOG_DIR = ROOT / "logs" / "probe_runs"
REINFORCEMENT_LOG_PATH = ROOT / "logs" / "reinforcement_metrics.jsonl"
RECENT_PROBE_FILES = 6
RECENT_REINFORCEMENT_LINES = 120


StatusTuple = Tuple[str, str, Mapping[str, Any]]
CheckFn = Callable[[], StatusTuple]
RepairFn = Callable[[], Tuple[bool, str, Mapping[str, Any]]]

STATUS_OK = "OK"
STATUS_WARN = "CHK"
STATUS_ERR = "ERR"
STATUS_TMO = "TMO"

CHECK_TIMEOUT = 7.0
SLEEP_BETWEEN = 0.5


def _status_ok(message: str = "", metadata: Optional[Mapping[str, Any]] = None) -> StatusTuple:
    return STATUS_OK, message, metadata or {}


def _status_chk(message: str, metadata: Optional[Mapping[str, Any]] = None) -> StatusTuple:
    return STATUS_WARN, message, metadata or {}


def _status_err(message: str, metadata: Optional[Mapping[str, Any]] = None) -> StatusTuple:
    return STATUS_ERR, message, metadata or {}


def _evaluate_thresholds(metrics: Mapping[str, Any]) -> tuple[list[tuple[str, float, float, str]], dict[str, Any]]:
    failures: list[tuple[str, float, float, str]] = []
    report: dict[str, Any] = {}
    for key, (threshold, mode) in CANARY_THRESHOLDS.items():
        try:
            value = float(metrics.get(key, 0.0))
        except (TypeError, ValueError):
            value = 0.0
        if mode == "min":
            ok = value >= threshold
        else:
            ok = value <= threshold
        report[key] = {
            "value": round(value, 4),
            "threshold": threshold,
            "mode": mode,
            "ok": ok,
        }
        if not ok:
            failures.append((key, value, threshold, mode))
    report["eligible_for_promotion"] = all(entry["ok"] for entry in report.values())
    return failures, report


async def _execute_canary(profile: str, settings_file: str) -> Mapping[str, Any]:
    """Run a single-turn probe against the ASGI app for the given profile."""
    main_module = importlib.import_module("main")
    previous = os.environ.get("LIVING_SETTINGS_FILE")
    os.environ["LIVING_SETTINGS_FILE"] = settings_file
    try:
        transport = httpx.ASGITransport(app=main_module.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://diagnostic") as client:
            await main_module._reload_runtime_settings()  # type: ignore[attr-defined]
            response = await client.post("/chat", json={"message": CANARY_PROMPT})
            response.raise_for_status()
            payload = response.json()
            payload["profile"] = profile
            return payload
    finally:
        if previous is None:
            os.environ.pop("LIVING_SETTINGS_FILE", None)
        else:
            os.environ["LIVING_SETTINGS_FILE"] = previous


def _tail_output(text: str, lines: int = 5) -> str:
    if not text:
        return ""
    fragments = [fragment for fragment in text.strip().splitlines() if fragment.strip()]
    if not fragments:
        return ""
    return "\n".join(fragments[-lines:])


@dataclass(frozen=True)
class Diagnostic:
    label: str
    component: str
    checker: CheckFn
    repair: Optional[RepairFn] = None


@dataclass
class DiagnosticResult:
    label: str
    component: str
    status: str
    message: str
    metadata: dict[str, Any] = field(default_factory=dict)
    attempts: int = 1
    repaired: bool = False
    repair_message: str | None = None
    repair_metadata: dict[str, Any] = field(default_factory=dict)
    elapsed: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        return payload


def check_main_pathway() -> StatusTuple:
    try:
        main = importlib.import_module("main")
        ctx, intent_prediction, length_plan, sampling, _, snapshot = main._prepare_chat_request("diagnostic ping")  # type: ignore[attr-defined]
        reply = main._compose_heuristic_reply(  # type: ignore[attr-defined]
            "diagnostic ping",
            context=ctx,
            intent=intent_prediction.intent,
            length_plan=length_plan,
        )
        metadata = {
            "intent": intent_prediction.intent,
            "sampling_keys": sorted(list(sampling.keys())),
            "length_plan": length_plan.get("label"),
            "controller_applied": (snapshot.get("controller") or {}).get("applied"),
        }
        if "User:" in reply or "Assistant:" in reply:
            return _status_chk("heuristic reply still includes transcript labels", metadata)
        if not reply.strip():
            return _status_chk("heuristic reply was empty", metadata)
        metadata["reply_excerpt"] = reply[:120]
        return _status_ok("heuristic pathway responded", metadata)
    except Exception as exc:
        return _status_err(f"{exc.__class__.__name__}: {exc}")


def repair_main_pathway() -> Tuple[bool, str, Mapping[str, Any]]:
    try:
        main = importlib.import_module("main")
        asyncio.run(main._reload_runtime_settings())  # type: ignore[attr-defined]
        return True, "runtime settings reloaded", {}
    except Exception as exc:  # pragma: no cover - defensive
        return False, f"{exc.__class__.__name__}: {exc}", {}


def check_intent_router() -> StatusTuple:
    try:
        from brain.intent_router import predict_intent

        prediction = predict_intent("I feel hopeful but anxious")
        metadata = {
            "intent": prediction.intent,
            "confidence": round(prediction.confidence, 4),
            "rationale": prediction.rationale,
        }
        if prediction.intent not in {"emotional", "analytical", "narrative", "reflective"}:
            return _status_chk(f"unexpected intent: {prediction.intent}", metadata)
        if not 0.0 <= prediction.confidence <= 1.0:
            return _status_chk(f"confidence out of bounds: {prediction.confidence}", metadata)
        return _status_ok("intent router healthy", metadata)
    except Exception as exc:
        return _status_err(f"{exc.__class__.__name__}: {exc}")


def check_local_llama_engine() -> StatusTuple:
    try:
        module = importlib.import_module("brain.local_llama_engine")
        engine_cls = getattr(module, "LocalLlamaEngine", None)
        if engine_cls is None:
            return _status_err("LocalLlamaEngine missing")
        required_methods = {"generate_reply", "stream_reply", "_compose_system_message"}
        missing = {name for name in required_methods if not hasattr(engine_cls, name)}
        metadata = {"missing_methods": sorted(missing)}
        if missing:
            return _status_chk("missing methods", metadata)
        role_stops = getattr(module, "ROLE_STOP_SEQUENCES", ())
        metadata["role_stop_sequences"] = list(role_stops)
        if not role_stops:
            return _status_chk("stop sequences not configured", metadata)
        return _status_ok("local engine interface present", metadata)
    except Exception as exc:
        return _status_err(f"{exc.__class__.__name__}: {exc}")


def check_llm_client() -> StatusTuple:
    try:
        module = importlib.import_module("brain.llm_client")
        client_cls = getattr(module, "LivingLLMClient", None)
        metadata = {"has_generate_reply": hasattr(client_cls, "generate_reply")}
        if client_cls is None:
            return _status_err("LivingLLMClient missing")
        if not inspect.isclass(client_cls):
            return _status_chk("LivingLLMClient is not a class")
        if not metadata["has_generate_reply"]:
            return _status_chk("generate_reply missing", metadata)
        return _status_ok("client available", metadata)
    except Exception as exc:
        return _status_err(f"{exc.__class__.__name__}: {exc}")


def check_hormone_system() -> StatusTuple:
    try:
        from hormones.hormones import HormoneSystem

        system = HormoneSystem()
        before = system.get_state()
        system.apply_stimulus("reward")
        after = system.get_state()
        deltas = {key: round(after[key] - before[key], 4) for key in before}
        metadata = {"delta": deltas}
        if before == after:
            return _status_chk("stimulus produced no delta", metadata)
        return _status_ok("hormone deltas responsive", metadata)
    except Exception as exc:
        return _status_err(f"{exc.__class__.__name__}: {exc}")


def check_memory_coupling() -> StatusTuple:
    try:
        from state_engine.engine import StateEngine

        engine = StateEngine()
        engine.register_event("diagnostic touchpoint", strength=0.6, hormone_deltas={"dopamine": 2.0})
        manager = engine.memory_manager
        short_term = list(manager._short_term)  # type: ignore[attr-defined]
        attributes = short_term[0].attributes if short_term else {}
        metadata = {
            "short_term_count": len(short_term),
            "has_endocrine_trace": isinstance(attributes, dict) and "endocrine" in attributes,
            "tags": list(attributes.get("tags", [])) if isinstance(attributes, dict) else [],
        }
        if not metadata["has_endocrine_trace"]:
            return _status_chk("endocrine trace missing from latest memory event", metadata)
        return _status_ok("memory coupling active", metadata)
    except Exception as exc:
        return _status_err(f"{exc.__class__.__name__}: {exc}")


def check_reinforcement() -> StatusTuple:
    try:
        from brain.reinforcement import ReinforcementTracker, score_response

        tracker = ReinforcementTracker()
        scores = score_response("I am sad", "I feel calmer after breathing", tracker=tracker)
        metadata = {key: scores.get(key) for key in scores}
        required = {
            "valence_delta",
            "length_score",
            "engagement_score",
            "authenticity_score",
            "affect_valence",
            "affect_intimacy",
            "affect_tension",
        }
        if not required.issubset(scores):
            missing = sorted(required.difference(scores))
            metadata["missing"] = missing
            return _status_chk("score keys missing", metadata)
        return _status_ok("reinforcement scores computed", metadata)
    except Exception as exc:
        return _status_err(f"{exc.__class__.__name__}: {exc}")


def check_persona_helpers() -> StatusTuple:
    try:
        from app.persona import build_persona_snapshot, compose_heuristic_reply
        from state_engine import StateEngine

        engine = StateEngine()
        engine.register_event("diagnostic reflection", strength=0.6, stimulus_type="reward")
        snapshot = build_persona_snapshot(engine)
        memory_manager = engine.memory_manager
        memory_context = {
            "summary": memory_manager.summarize_recent(),
            "working": memory_manager.working_snapshot(),
            "internal_reflections": ["I noted the way my ribs eased when they reached out."],
            "long_term": [],
        }

        def _shorten(text: str, limit: int) -> str:
            text = (text or "").strip()
            if len(text) <= limit:
                return text
            shortened = text[: limit - 3].rsplit(" ", 1)[0]
            return f"{shortened}..."

        context = {
            "persona": snapshot,
            "memory": memory_context,
            "self_note": "My shoulders keep loosening when I listen closely.",
        }
        length_plan = {"label": "concise", "hint": "Keep it intimate and brief."}
        reply = compose_heuristic_reply(
            "diagnostic ping about my breathing",
            context=context,
            intent="reflective",
            length_plan=length_plan,
            state_engine=engine,
            shorten=_shorten,
        )
        metadata = {
            "instructions": len(snapshot.get("instructions", [])),
            "memory_summary": memory_context.get("summary"),
            "reply_excerpt": reply[:120],
        }
        if not reply.strip():
            return _status_chk("persona helper returned an empty reply", metadata)
        if "diagnostic" not in reply.lower():
            return _status_chk("persona helper dropped user content", metadata)
        return _status_ok("persona helpers responsive", metadata)
    except Exception as exc:
        return _status_err(f"{exc.__class__.__name__}: {exc}")


def check_telemetry_helpers() -> StatusTuple:
    try:
        from app.telemetry import compose_live_status, compose_turn_telemetry, write_telemetry_snapshot
        from app.runtime import RuntimeState
        from state_engine import StateEngine

        engine = StateEngine()
        runtime_state = RuntimeState()
        live_status = compose_live_status(
            state_engine=engine,
            runtime_state=runtime_state,
            model_alias="diagnostic",
            local_llama_available=False,
        )

        def _shorten(text: str, limit: int) -> str:
            text = (text or "").strip()
            if len(text) <= limit:
                return text
            shortened = text[: limit - 3].rsplit(" ", 1)[0]
            return f"{shortened}..."

        context = {
            "mood": engine.state.get("mood"),
            "hormones": engine.hormone_system.get_state(),
            "memory": {
                "summary": "diagnostic memory",
                "working": [],
                "long_term": [],
            },
            "affect": {"traits": {"warmth": 0.3}, "tags": ["calm"]},
            "sampling_policy_preview": {"label": "diagnostic"},
        }
        sampling = {"temperature": 0.82, "top_p": 0.9, "max_tokens": 256}
        snapshot = {
            "timestamp": live_status["timestamp"],
            "profile": live_status["profile"],
            "sampling": dict(sampling),
            "policy_preview": {"label": "diagnostic"},
        }
        telemetry = compose_turn_telemetry(
            context=context,
            sampling=sampling,
            snapshot=snapshot,
            state_engine=engine,
            shorten=_shorten,
            model_alias="diagnostic",
        )
        temp_path = ROOT / "logs" / "diagnostic_snapshot.json"
        write_telemetry_snapshot(telemetry, temp_path, logger=None)
        metadata = {
            "live_status_keys": sorted(list(live_status.keys())),
            "telemetry_keys": sorted(list(telemetry.keys())),
            "snapshot_written": temp_path.exists(),
        }
        if temp_path.exists():
            try:
                data = temp_path.read_text(encoding="utf-8")
                metadata["snapshot_excerpt"] = data[:120]
            finally:
                temp_path.unlink(missing_ok=True)
        if not metadata["snapshot_written"]:
            return _status_chk("telemetry snapshot was not written to disk", metadata)
        return _status_ok("telemetry helpers produced output", metadata)
    except Exception as exc:
        return _status_err(f"{exc.__class__.__name__}: {exc}")


def check_canary_probes() -> StatusTuple:
    try:
        main_module = importlib.import_module("main")
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        profiles_report: list[dict[str, Any]] = []
        breaches: list[tuple[str, list[tuple[str, float, float, str]]]] = []
        try:
            for profile, settings_file in CANARY_PROFILES:
                payload = loop.run_until_complete(_execute_canary(profile, settings_file))
                metrics = payload.get("reinforcement") or {}
                failures, report = _evaluate_thresholds(metrics)
                profiles_report.append(
                    {
                        "profile": profile,
                        "metrics": report,
                        "engine": payload.get("source"),
                    }
                )
                if failures:
                    breaches.append((profile, failures))
        finally:
            shutdown = getattr(main_module, "_shutdown_clients", None)
            if shutdown is not None:
                try:
                    loop.run_until_complete(shutdown())  # type: ignore[arg-type]
                except Exception:
                    pass
            asyncio.set_event_loop(None)
            loop.close()
        metadata = {"profiles": profiles_report}
        if not profiles_report:
            return _status_chk("canary probes produced no telemetry", metadata)
        if breaches:
            profile, failure_list = breaches[0]
            metric, value, threshold, mode = failure_list[0]
            expected = ">=" if mode == "min" else "<="
            message = f"{profile} canary breached {metric} (value={value:.3f}, expected {expected} {threshold})"
            return _status_chk(message, metadata)
        return _status_ok("canary probes within thresholds", metadata)
    except Exception as exc:
        return _status_err(f"{exc.__class__.__name__}: {exc}")


def check_hormone_model() -> StatusTuple:
    try:
        from main import HORMONE_MODEL, HORMONE_MODEL_PATH  # type: ignore[attr-defined]
        from brain.hormone_model import load_model

        path = Path(HORMONE_MODEL_PATH or "config/hormone_model.json")
        if not path.exists():
            return _status_err("hormone model file missing", {"path": str(path)})
        model = HORMONE_MODEL or load_model(path)
        if model is None:
            return _status_err("failed to load hormone model", {"path": str(path)})
        metadata = {"feature_count": len(model.feature_names), "scale": model.hormone_scale}
        return _status_ok("hormone model loaded", metadata)
    except Exception as exc:
        return _status_err(f"{exc.__class__.__name__}: {exc}")


def repair_hormone_model() -> Tuple[bool, str, Mapping[str, Any]]:
    try:
        log_path = ROOT / "logs" / "endocrine_turns.jsonl"
        if not log_path.exists():
            return False, "endocrine log missing for retrain", {"log": str(log_path)}
        script = ROOT / "scripts" / "train_hormone_model.py"
        output_path = ROOT / "config" / "hormone_model.json"
        cmd = [
            sys.executable,
            str(script),
            "--log-file",
            str(log_path),
            "--output",
            str(output_path),
        ]
        result = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
        if result.returncode != 0:
            metadata = {"stderr": _tail_output(result.stderr)}
            return False, "hormone model retrain failed", metadata
        main = importlib.import_module("main")
        main._reinitialize_hormone_model()  # type: ignore[attr-defined]
        metadata = {"stdout": _tail_output(result.stdout)}
        return True, "retrained hormone model", metadata
    except Exception as exc:  # pragma: no cover - defensive
        return False, f"{exc.__class__.__name__}: {exc}", {}


def check_controller_policy() -> StatusTuple:
    try:
        from main import CONTROLLER_POLICY, CONTROLLER_POLICY_PATH  # type: ignore[attr-defined]
        from brain.controller_policy import load_controller_policy

        path = Path(CONTROLLER_POLICY_PATH or "config/controller_policy.json")
        if not path.exists():
            return _status_err("controller policy file missing", {"path": str(path)})
        policy = CONTROLLER_POLICY or load_controller_policy(path)
        if policy is None:
            return _status_err("failed to load controller policy", {"path": str(path)})
        metadata = {
            "input_features": len(policy.input_names),
            "output_names": list(policy.output_names),
            "hidden_size": policy.hidden_size,
        }
        return _status_ok("controller policy loaded", metadata)
    except Exception as exc:
        return _status_err(f"{exc.__class__.__name__}: {exc}")


def repair_controller_policy() -> Tuple[bool, str, Mapping[str, Any]]:
    try:
        log_path = ROOT / "logs" / "endocrine_turns.jsonl"
        if not log_path.exists():
            return False, "endocrine log missing for retrain", {"log": str(log_path)}
        script = ROOT / "scripts" / "train_controller_policy.py"
        output_path = ROOT / "config" / "controller_policy.json"
        cmd = [
            sys.executable,
            str(script),
            "--log-file",
            str(log_path),
            "--output",
            str(output_path),
        ]
        result = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
        if result.returncode != 0:
            metadata = {"stderr": _tail_output(result.stderr)}
            return False, "controller policy retrain failed", metadata
        main = importlib.import_module("main")
        main._reinitialize_controller_policy()  # type: ignore[attr-defined]
        metadata = {"stdout": _tail_output(result.stdout)}
        return True, "retrained controller policy", metadata
    except Exception as exc:  # pragma: no cover - defensive
        return False, f"{exc.__class__.__name__}: {exc}", {}


def _load_recent_probe_summaries(limit: int = RECENT_PROBE_FILES) -> list[dict[str, Any]]:
    if not PROBE_LOG_DIR.exists():
        return []
    files = sorted(PROBE_LOG_DIR.glob("*.jsonl"), key=lambda item: item.stat().st_mtime, reverse=True)
    summaries: list[dict[str, Any]] = []
    for path in files:
        try:
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        payload = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    summary = payload.get("summary")
                    if not isinstance(summary, Mapping):
                        continue
                    metrics = {
                        key: float(summary.get(key, 0.0))
                        for key in CANARY_THRESHOLDS.keys()
                    }
                    summaries.append(
                        {
                            "file": path.name,
                            "timestamp": payload.get("timestamp"),
                            "profile": payload.get("profile"),
                            "metrics": metrics,
                            "gate": summary.get("gate"),
                        }
                    )
                    break
        except OSError:
            continue
        if len(summaries) >= limit:
            break
    return summaries


def check_probe_log_health() -> StatusTuple:
    try:
        if not PROBE_LOG_DIR.exists():
            return _status_chk("probe log directory missing", {"path": str(PROBE_LOG_DIR)})
        summaries = _load_recent_probe_summaries()
        if not summaries:
            return _status_chk("no probe summaries found", {"path": str(PROBE_LOG_DIR)})
        breaches: list[tuple[dict[str, Any], list[tuple[str, float, float, str]]]] = []
        for entry in summaries:
            failures, report = _evaluate_thresholds(entry["metrics"])
            entry["metrics"] = report
            gate = entry.get("gate") or {}
            eligible = bool(gate.get("eligible_for_promotion", True)) if isinstance(gate, Mapping) else True
            if failures or not eligible:
                breaches.append((entry, failures))
        metadata = {"summaries": summaries}
        if breaches:
            entry, failure_list = breaches[0]
            if failure_list:
                metric, value, threshold, mode = failure_list[0]
                expected = ">=" if mode == "min" else "<="
                message = (
                    f"recent probe {entry.get('file')} ({entry.get('profile')}) breached {metric} "
                    f"(value={value:.3f}, expected {expected} {threshold})"
                )
            else:
                message = f"recent probe {entry.get('file')} ({entry.get('profile')}) failed promotion gate"
            return _status_chk(message, metadata)
        return _status_ok("recent probe logs within thresholds", metadata)
    except Exception as exc:
        return _status_err(f"{exc.__class__.__name__}: {exc}")


def check_reinforcement_log_health() -> StatusTuple:
    try:
        if not REINFORCEMENT_LOG_PATH.exists():
            return _status_chk("reinforcement log missing", {"path": str(REINFORCEMENT_LOG_PATH)})
        recent_lines: deque[str] = deque(maxlen=RECENT_REINFORCEMENT_LINES)
        with REINFORCEMENT_LOG_PATH.open("r", encoding="utf-8") as handle:
            for line in handle:
                recent_lines.append(line)
        if not recent_lines:
            return _status_chk("reinforcement log is empty", {"path": str(REINFORCEMENT_LOG_PATH)})
        aggregates: dict[str, list[float]] = {key: [] for key in CANARY_THRESHOLDS.keys()}
        timestamps: list[str] = []
        for raw in recent_lines:
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                continue
            metrics = payload.get("metrics") or {}
            if isinstance(metrics, Mapping):
                for key in aggregates.keys():
                    try:
                        aggregates[key].append(float(metrics.get(key, 0.0)))
                    except (TypeError, ValueError):
                        continue
            stamp = payload.get("timestamp")
            if isinstance(stamp, str):
                timestamps.append(stamp)
        averaged: dict[str, float] = {}
        spread: dict[str, tuple[float | None, float | None]] = {}
        for key, values in aggregates.items():
            if values:
                averaged[key] = round(statistics.fmean(values), 4)
                spread[key] = (round(min(values), 4), round(max(values), 4))
            else:
                averaged[key] = 0.0
                spread[key] = (None, None)
        failures, report = _evaluate_thresholds(averaged)
        auth_values = aggregates.get("authenticity_score", [])
        high_count = sum(1 for value in auth_values if value >= CANARY_THRESHOLDS["authenticity_score"][0])
        high_ratio = (high_count / len(auth_values)) if auth_values else 0.0
        metadata = {
            "averages": report,
            "min_max": spread,
            "samples": {key: len(values) for key, values in aggregates.items()},
            "window": {
                "count": len(recent_lines),
                "first": timestamps[0] if timestamps else None,
                "last": timestamps[-1] if timestamps else None,
            },
            "high_ratio": round(high_ratio, 4),
            "high_count": high_count,
        }
        if failures and high_ratio < 0.15 and high_count < 10:
            metric, value, threshold, mode = failures[0]
            expected = ">=" if mode == "min" else "<="
            message = f"reinforcement averages breached {metric} (value={value:.3f}, expected {expected} {threshold})"
            return _status_chk(message, metadata)
        return _status_ok("reinforcement averages within thresholds", metadata)
    except Exception as exc:
        return _status_err(f"{exc.__class__.__name__}: {exc}")


def check_endocrine_handshake() -> StatusTuple:
    try:
        import main  # type: ignore
        from brain.reinforcement import ReinforcementTracker, score_response

        tracker = ReinforcementTracker()
        reinforcement = score_response(
            "I feel restless",
            "I notice my shoulders loosening while I breathe",
            tracker=tracker,
        )
        traits = main.state_engine.trait_snapshot()
        if traits is None:
            main.state_engine.register_event("diagnostic endocrine handshake", hormone_deltas={"serotonin": 1.2})
            traits = main.state_engine.trait_snapshot()
        pre_hormones = dict(main.state_engine.hormone_system.get_state())

        metadata: dict[str, Any] = {"has_model": bool(main.HORMONE_MODEL)}  # type: ignore[attr-defined]
        if main.HORMONE_MODEL:  # type: ignore[attr-defined]
            predicted = main.HORMONE_MODEL.predict_delta(  # type: ignore[attr-defined]
                pre_hormones,
                reinforcement,
                intent="reflective",
                length_label="concise",
                profile=current_profile(),
            )
            metadata["predicted"] = predicted
            if not predicted:
                return _status_chk("hormone model returned empty delta", metadata)
            if not all(isinstance(value, (int, float)) for value in predicted.values()):
                return _status_err("hormone model produced non-numeric values", metadata)
            magnitude = max(abs(float(value)) for value in predicted.values())
            metadata["max_magnitude"] = round(magnitude, 6)
            if magnitude < 0.05:
                return _status_chk("hormone deltas were negligible", metadata)
            return _status_ok("hormone model produced deltas", metadata)
        return _status_chk("hormone model unavailable; heuristic fallback only", metadata)
    except Exception as exc:
        return _status_err(f"{exc.__class__.__name__}: {exc}")


def check_controller_handshake() -> StatusTuple:
    try:
        import main  # type: ignore

        context, intent_prediction, length_plan, _, _, _ = main._prepare_chat_request(  # type: ignore[attr-defined]
            "diagnostic controller handshake"
        )
        hormones = context.get("hormones", {})
        traits = main.state_engine.trait_snapshot()
        tags = gather_active_tags(main.state_engine)
        features = build_controller_feature_map(
            state_engine=main.state_engine,
            runtime_state=main.runtime_state,
            traits=traits,
            hormones=hormones,
            intent=intent_prediction.intent,
            length_label=length_plan.get("label"),
            profile=current_profile(),
            tags=tags,
        )
        result = run_controller_policy(
            main.CONTROLLER_RUNTIME,  # type: ignore[attr-defined]
            main.CONTROLLER_LOCK,  # type: ignore[attr-defined]
            main.runtime_state,  # type: ignore[attr-defined]
            features,
            tags,
        )
        runtime = main.CONTROLLER_RUNTIME  # type: ignore[attr-defined]
        if runtime is not None:
            runtime.reset()
        if result is None:
            return _status_chk("controller runtime unavailable", {"input_size": len(features)})
        metadata = {
            "input_size": len(features),
            "adjustments": result.adjustments,
            "raw_outputs": [round(float(value), 6) for value in result.raw_outputs],
        }
        if not any(abs(value) > 1e-3 for value in result.adjustments.values()):
            return _status_chk("controller adjustments were negligible", metadata)
        return _status_ok("controller produced adjustments", metadata)
    except Exception as exc:
        return _status_err(f"{exc.__class__.__name__}: {exc}")


def check_router_sampling_alignment() -> StatusTuple:
    try:
        main = importlib.import_module("main")
        _, intent_prediction, _, sampling, _, snapshot = main._prepare_chat_request("diagnostic sampling check")  # type: ignore[attr-defined]
        hormone_sampling = snapshot.get("hormone_sampling", {})
        controller_snapshot = snapshot.get("controller", {})
        metadata = {
            "intent": intent_prediction.intent,
            "sampling": sampling,
            "hormone_sampling": hormone_sampling,
            "controller_applied": controller_snapshot.get("applied"),
        }
        required_keys = {"temperature", "top_p", "frequency_penalty"}
        if not required_keys.issubset(sampling):
            return _status_chk("sampling missing expected keys", metadata)
        return _status_ok("router/sampling alignment healthy", metadata)
    except Exception as exc:
        return _status_err(f"{exc.__class__.__name__}: {exc}")


async def _probe_http_async() -> Mapping[str, Any]:
    import main  # noqa: F401

    transport = httpx.ASGITransport(app=main.app)  # type: ignore[attr-defined]
    async with httpx.AsyncClient(transport=transport, base_url="http://diagnostic") as client:
        ping = await client.get("/ping")
        ping.raise_for_status()
        state = await client.get("/state")
        state.raise_for_status()
        state_payload = state.json()
    return {
        "ping": ping.json(),
        "state_keys": sorted(state_payload.keys()),
    }


def check_http_endpoints() -> StatusTuple:
    try:
        metadata = asyncio.run(_probe_http_async())
        return _status_ok("HTTP endpoints responsive", metadata)
    except Exception as exc:
        return _status_err(f"{exc.__class__.__name__}: {exc}")


def check_test_scaffolding() -> StatusTuple:
    try:
        module = importlib.import_module("tests.test_chat")
        has_suite = hasattr(module, "StreamingEndpointTests")
        return _status_ok("test scaffolding present", {"has_streaming_suite": has_suite})
    except Exception as exc:
        return _status_err(f"{exc.__class__.__name__}: {exc}")


DIAGNOSTICS: Iterable[Diagnostic] = (
    Diagnostic("Main pathway", "main.py", check_main_pathway, repair_main_pathway),
    Diagnostic("Intent router", "brain/intent_router.py", check_intent_router),
    Diagnostic("Local llama engine", "brain/local_llama_engine.py", check_local_llama_engine),
    Diagnostic("LLM client", "brain/llm_client.py", check_llm_client),
    Diagnostic("Hormone system", "hormones/hormones.py", check_hormone_system),
    Diagnostic("Memory coupling", "state_engine/engine.py", check_memory_coupling),
    Diagnostic("Persona helpers", "app/persona.py", check_persona_helpers),
    Diagnostic("Reinforcement scoring", "brain/reinforcement.py", check_reinforcement),
    Diagnostic("Telemetry helpers", "app/telemetry.py", check_telemetry_helpers),
    Diagnostic("Canary probes", "profiles (instruct/base)", check_canary_probes),
    Diagnostic("Hormone model", "config/hormone_model.json", check_hormone_model, repair_hormone_model),
    Diagnostic("Controller policy", "config/controller_policy.json", check_controller_policy, repair_controller_policy),
    Diagnostic("Endocrine handshake", "brain/hormone_model.py", check_endocrine_handshake),
    Diagnostic("Controller handshake", "main.py", check_controller_handshake),
    Diagnostic("Router/sampling alignment", "main.py", check_router_sampling_alignment),
    Diagnostic("HTTP endpoints", "FastAPI", check_http_endpoints),
    Diagnostic("Test scaffolding", "tests/test_chat.py", check_test_scaffolding),
    Diagnostic("Probe log health", "logs/probe_runs", check_probe_log_health),
    Diagnostic("Reinforcement log health", "logs/reinforcement_metrics.jsonl", check_reinforcement_log_health),
)


def _execute_checker(diag: Diagnostic) -> StatusTuple:
    return diag.checker()


def iter_diagnostics() -> Iterable[tuple[Diagnostic, DiagnosticResult]]:
    results: list[tuple[Diagnostic, DiagnosticResult]] = []
    for diag in DIAGNOSTICS:
        start = time.perf_counter()
        try:
            status, message, metadata = _execute_checker(diag)
        except Exception as exc:  # pragma: no cover - defensive
            status, message, metadata = STATUS_ERR, f"{exc.__class__.__name__}: {exc}", {}
        elapsed = time.perf_counter() - start

        result = DiagnosticResult(
            label=diag.label,
            component=diag.component,
            status=status,
            message=message,
            metadata=dict(metadata),
            elapsed=round(elapsed, 4),
        )
        results.append((diag, result))
        time.sleep(SLEEP_BETWEEN)
        yield diag, result


def run_diagnostics(*, allow_repair: bool = False) -> list[DiagnosticResult]:
    return [result for _, result in iter_diagnostics()]


def _format_text(results: Iterable[DiagnosticResult]) -> str:
    lines = ["Diagnostic Sequence Initiated...\n"]
    for res in results:
        line = f"{res.label}: {res.status}"
        if res.message:
            line += f" - {res.message}"
        if res.repaired:
            line += f" (repaired: {res.repair_message})"
        lines.append(line)
    return "\n".join(lines)


def _format_json(results: Iterable[DiagnosticResult]) -> str:
    payload = {
        "timestamp": time.time(),
        "results": [res.as_dict() for res in results],
    }
    return json.dumps(payload, indent=2)


def _summarize_statuses(results: Iterable[DiagnosticResult]) -> dict[str, int]:
    counts = {
        STATUS_OK: 0,
        STATUS_WARN: 0,
        STATUS_ERR: 0,
        STATUS_TMO: 0,
    }
    for res in results:
        counts[res.status] = counts.get(res.status, 0) + 1
    return counts


def _perform_repairs(diag_results: list[tuple[Diagnostic, DiagnosticResult]], *, verbose: bool = True) -> None:
    pending = [
        (diag, res)
        for diag, res in diag_results
        if res.status in {STATUS_WARN, STATUS_ERR} and diag.repair is not None
    ]
    if not pending:
        if verbose:
            print("\nNo repairs required.", flush=True)
        return

    for diag, res in pending:
        if verbose:
            print(f"\n[repair] {diag.label}: initiating in 2s...", flush=True)
        time.sleep(2)
        try:
            repaired, repair_message, repair_meta = diag.repair()
        except Exception as exc:  # pragma: no cover - defensive
            repaired = False
            repair_message = f"{exc.__class__.__name__}: {exc}"
            repair_meta = {}

        res.repair_message = repair_message
        res.repair_metadata = dict(repair_meta)

        if repaired:
            res.repaired = True
            res.attempts += 1
            start = time.perf_counter()
            try:
                status, message, metadata = _execute_checker(diag)
            except Exception as exc:  # pragma: no cover - defensive
                status, message, metadata = STATUS_ERR, f"{exc.__class__.__name__}: {exc}", {}
            res.status = status
            res.message = message
            res.metadata = dict(metadata)
            res.elapsed += round(time.perf_counter() - start, 4)
            if verbose:
                print(f"[repair] {diag.label}: completed -> {status}", flush=True)
        else:
            if verbose:
                print(f"[repair] {diag.label}: failed - {repair_message}", flush=True)


def run_cli(args: argparse.Namespace) -> int:
    diag_results: list[tuple[Diagnostic, DiagnosticResult]] = []
    text_mode = args.format == "text"

    if text_mode:
        print("Diagnostic Sequence Initiated...\n", flush=True)
        for diag, res in iter_diagnostics():
            diag_results.append((diag, res))
            line = f"{res.label}: {res.status}"
            if res.message:
                line += f" - {res.message}"
            print(line, flush=True)
        results = [res for _, res in diag_results]
        counts = _summarize_statuses(results)
        print(
            f"\nSystem Wide Diagnostic Cascade complete. "
            f"OK:{counts[STATUS_OK]} WARN:{counts[STATUS_WARN]} ERR:{counts[STATUS_ERR]} TMO:{counts[STATUS_TMO]}",
            flush=True,
        )
    else:
        diag_results = list(iter_diagnostics())
        results = [res for _, res in diag_results]

    if args.repair:
        _perform_repairs(diag_results, verbose=text_mode)
        results = [res for _, res in diag_results]
        if text_mode:
            counts = _summarize_statuses(results)
            print(
                f"\nPost-repair status. OK:{counts[STATUS_OK]} WARN:{counts[STATUS_WARN]} ERR:{counts[STATUS_ERR]} TMO:{counts[STATUS_TMO]}",
                flush=True,
            )

    if args.format == "json":
        output = _format_json(results)
        if args.output:
            Path(args.output).write_text(output, encoding="utf-8")
        else:
            print(output)
    else:
        if args.output:
            Path(args.output).write_text(_format_text(results), encoding="utf-8")

    exit_code = 0
    for res in results:
        if res.status in {STATUS_ERR, STATUS_TMO}:
            exit_code = 1
            break
    return exit_code


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Living AI diagnostics.")
    parser.add_argument("--format", choices={"text", "json"}, default="text", help="Output format (default: text).")
    parser.add_argument("--repair", action="store_true", help="Attempt automated repairs when possible.")
    parser.add_argument("--output", help="Optional path to write the report.")
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = parse_args(argv)
    exit_code = run_cli(args)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
