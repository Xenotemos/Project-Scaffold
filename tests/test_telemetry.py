from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.runtime import RuntimeState
from app.telemetry import compose_live_status, compose_turn_telemetry, write_telemetry_snapshot
from state_engine import StateEngine


def _shorten(text: str, limit: int) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    shortened = text[: limit - 3].rsplit(" ", 1)[0]
    return f"{shortened}..."


class TelemetryHelperTests:
    def setup_method(self) -> None:
        self.engine = StateEngine()
        self.runtime_state = RuntimeState()

    def _base_context(self) -> dict[str, Any]:
        return {
            "mood": self.engine.state.get("mood"),
            "hormones": self.engine.hormone_system.get_state(),
            "memory": {"summary": "diagnostic memory", "working": [], "long_term": []},
            "affect": {"traits": {"warmth": 0.3}, "tags": ["calm"]},
            "sampling_policy_preview": {"label": "diagnostic"},
        }

    def test_compose_live_status_contains_metric_summary(self) -> None:
        status = compose_live_status(
            state_engine=self.engine,
            runtime_state=self.runtime_state,
            model_alias="diag",
            local_llama_available=False,
        )
        assert "metrics" in status
        assert "controller" in status

    def test_turn_telemetry_snapshot_written(self, tmp_path: Path) -> None:
        status = compose_live_status(
            state_engine=self.engine,
            runtime_state=self.runtime_state,
            model_alias="diag",
            local_llama_available=False,
        )
        context = self._base_context()
        sampling = {"temperature": 0.82, "top_p": 0.9, "max_tokens": 256}
        snapshot = {
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
            "profile": status["profile"],
            "sampling": dict(sampling),
            "policy_preview": {"label": "diagnostic"},
            "controller": {"adjustments": {}, "applied": {}, "raw_outputs": [], "hidden_state": []},
        }
        telemetry = compose_turn_telemetry(
            context=context,
            sampling=sampling,
            snapshot=snapshot,
            state_engine=self.engine,
            shorten=_shorten,
            model_alias="diag",
        )
        output_path = tmp_path / "telemetry.json"
        write_telemetry_snapshot(telemetry, output_path, logger=None)
        assert output_path.exists()
        data = output_path.read_text(encoding="utf-8")
        assert '"profile"' in data
        assert '"sampling"' in data

