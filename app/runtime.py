"""Runtime state container for mutable cross-cutting variables."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Deque, Dict, Tuple

from brain.reinforcement import ReinforcementTracker


@dataclass
class RuntimeState:
    """Mutable state shared across chat/controller cycles."""

    history_window: int = 40
    session_counter: int = 1
    metric_sample_counter: int = 0
    last_sampling_snapshot: Dict[str, Any] = field(default_factory=dict)
    last_hormone_delta: Dict[str, float] | None = None
    last_metric_averages: Dict[str, float] = field(default_factory=dict)
    last_reinforcement_metrics: Dict[str, Any] = field(default_factory=dict)
    last_controller_result: Any | None = None
    last_controller_applied: Dict[str, Any] | None = None
    last_controller_features: Dict[str, float] | None = None
    last_controller_tags: Tuple[str, ...] = ()
    helper_drift_level: float = 0.0
    self_focus_streak: int = 0
    clamp_recovery_turns: int = 0
    last_clamp_reset: datetime | None = None
    clamp_priming_turns: int = 0
    recovery_good_streak: int = 0
    reset_priming_bias: float = 0.0
    recovery_lowself_streak: int = 0
    last_user_prompt: str = ""
    low_self_success_streak: int = 0
    self_narration_note: str = ""
    memory_spotlight_keys: list[str] = field(default_factory=list)
    auth_history: Deque[float] = field(init=False)
    drift_history: Deque[float] = field(init=False)
    self_history: Deque[float] = field(init=False)
    affect_valence_history: Deque[float] = field(init=False)
    affect_intimacy_history: Deque[float] = field(init=False)
    affect_tension_history: Deque[float] = field(init=False)
    reinforcement_tracker: ReinforcementTracker = field(default_factory=ReinforcementTracker)

    def __post_init__(self) -> None:
        self._init_histories()

    def _init_histories(self) -> None:
        self.auth_history = deque(maxlen=self.history_window)
        self.drift_history = deque(maxlen=self.history_window)
        self.self_history = deque(maxlen=self.history_window)
        self.affect_valence_history = deque(maxlen=self.history_window)
        self.affect_intimacy_history = deque(maxlen=self.history_window)
        self.affect_tension_history = deque(maxlen=self.history_window)

    def reset_controller(self) -> None:
        self.last_controller_result = None
        self.last_controller_applied = None
        self.last_controller_features = None
        self.last_controller_tags = ()

    def clear_metric_state(self) -> None:
        self.metric_sample_counter = 0
        self.last_metric_averages.clear()
        self.last_reinforcement_metrics.clear()
        self.auth_history.clear()
        self.drift_history.clear()
        self.self_history.clear()
        self.affect_valence_history.clear()
        self.affect_intimacy_history.clear()
        self.affect_tension_history.clear()


__all__ = ["RuntimeState"]
