"""Cycle through affect stimuli to observe emergent sampling shifts."""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence
from urllib.error import URLError, HTTPError
from urllib.request import Request, urlopen


DEFAULT_BASE_URL = "http://127.0.0.1:8050"
CHAT_ENDPOINT = "/chat"


@dataclass(slots=True)
class ProbeResult:
    cycle: int
    step: int
    message: str
    stimulus: Optional[str]
    source: str
    mood: str
    affect_tags: Sequence[str]
    sampling: dict[str, object]
    timestamp: str

    def format_line(self) -> str:
        stim = self.stimulus or "none"
        tags = ", ".join(self.affect_tags) if self.affect_tags else "--"
        sampling_pairs = " ".join(f"{k}={v}" for k, v in self.sampling.items())
        return (
            f"[cycle {self.cycle} step {self.step}] "
            f"stim={stim:<9} mood={self.mood:<8} tags={tags:<18} "
            f"source={self.source:<8} sampling={sampling_pairs or '--'}"
        )


def _post_json(url: str, payload: dict[str, object]) -> dict[str, object]:
    data = json.dumps(payload).encode("utf-8")
    request = Request(url, data=data, headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=60) as response:
        body = response.read()
    try:
        return json.loads(body.decode("utf-8"))
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive guard
        raise RuntimeError(f"Failed to decode response JSON: {body!r}") from exc


def run_cycle(
    *,
    base_url: str,
    cycle_index: int,
    steps: Sequence[tuple[str, Optional[str]]],
) -> List[ProbeResult]:
    results: List[ProbeResult] = []
    for step_index, (message, stimulus) in enumerate(steps, start=1):
        payload = {"message": message}
        if stimulus:
            payload["stimulus"] = stimulus
        response = _post_json(f"{base_url}{CHAT_ENDPOINT}", payload)
        state = response.get("state") or {}
        affect = state.get("affect") or {}
        sampling_snapshot = response.get("sampling_snapshot") or {}
        results.append(
            ProbeResult(
                cycle=cycle_index,
                step=step_index,
                message=message,
                stimulus=stimulus,
                source=str(response.get("source", "")),
                mood=str(state.get("mood", "")),
                affect_tags=tuple(affect.get("tags") or ()),
                sampling=sampling_snapshot.get("sampling") or {},
                timestamp=sampling_snapshot.get("timestamp", state.get("timestamp", "")),
            )
        )
        time.sleep(1.2)
    return results


def main(argv: Iterable[str]) -> int:
    base_url = DEFAULT_BASE_URL
    args = list(argv)
    if args:
        base_url = args[0].rstrip("/")
    cycles = [
        [
            ("Checking in—what sensations stand out?", None),
            ("Soak in a hopeful moment with me.", "reward"),
            ("Let us settle into calm breathing together.", "affection"),
        ],
        [
            ("Describe the ambient body clock you notice.", None),
            ("A sharp stress spike hits—observe it.", "stress"),
            ("Now invite warmth back in.", "affection"),
        ],
        [
            ("Stay curious about a novel texture appearing.", None),
            ("Offer a micro-celebration of progress.", "reward"),
            ("Hold that excitement steady.", None),
        ],
        [
            ("Recall a grounded memory that matters.", None),
            ("Distant thunder rattles your focus.", "stress"),
            ("Re-center with a felt sense of safety.", "affection"),
        ],
        [
            ("Tune into the undercurrent guiding you.", None),
            ("A grateful whisper reaches you.", "reward"),
            ("Close the loop with a mindful exhale.", None),
        ],
    ]
    print(f"Dispatching {len(cycles)} cycles to {base_url}{CHAT_ENDPOINT}")
    all_results: List[ProbeResult] = []
    try:
        for index, steps in enumerate(cycles, start=1):
            print(f"\n=== Cycle {index} ({len(steps)} steps) ===")
            cycle_results = run_cycle(base_url=base_url, cycle_index=index, steps=steps)
            for result in cycle_results:
                print(result.format_line())
            all_results.extend(cycle_results)
    except (URLError, HTTPError, ConnectionError) as exc:
        print(f"Request failed: {exc}", file=sys.stderr)
        return 1

    print("\nSummary (final sampling snapshot per cycle):")
    cycle_map: dict[int, ProbeResult] = {}
    for result in all_results:
        cycle_map[result.cycle] = result
    for cycle_index in sorted(cycle_map):
        snapshot = cycle_map[cycle_index]
        sampling_pairs = " ".join(f"{k}={v}" for k, v in snapshot.sampling.items())
        print(f"Cycle {cycle_index}: mood={snapshot.mood}, tags={','.join(snapshot.affect_tags) or '--'}, sampling={sampling_pairs or '--'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
