"""Auto-label affect samples using qwen3-4b-thinking via llama.cpp CLI."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import requests

LLAMA_SERVER = Path("third_party/llama.cpp/build/bin/Release/llama-server.exe")
MODEL_PATH = Path(r"D:\AI\LLMs\Mistral-7B-v0.3.Q4_K_M.gguf")
DEFAULT_INPUT = Path("fine_tune/harvest/run003_external_subset.label.jsonl")
DEFAULT_OUTPUT = Path("fine_tune/harvest/run003_external_auto.jsonl")

PROMPT_TEMPLATE = """You are an affect classifier that outputs JSON only.
Read the user's latest utterance and estimate affect scores on [-1,1] plus 0-1 confidence.
Definitions:
- valence: pleasant (+1) vs painful (-1)
- intimacy: closeness / relational pull (+1) vs distance (-1)
- tension: bodily arousal / urgency (+1) vs relaxed (-1)
Also provide a short list of tags capturing tone (e.g., hostile, caring, numb).
Consider the archetype hint when relevant, but prioritise the text itself.
Answer ONLY with JSON: {{"valence": float, "intimacy": float, "tension": float, "confidence": float, "tags": [..], "reason": "brief"}}.
Archetype hint: {archetype}
User text: ```{text}```"""

JSON_PATTERN = re.compile(r"\{.*\}", re.DOTALL)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Auto-label affect subset via llama.cpp")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Subset label JSONL file.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Destination JSONL for auto labels.")
    parser.add_argument("--resume", type=int, default=0, help="Number of already processed rows (append mode).")
    parser.add_argument("--temp", type=float, default=0.2, help="Sampling temperature.")
    parser.add_argument("--max-tokens", type=int, default=256, help="Max tokens per response.")
    parser.add_argument("--port", type=int, default=8088, help="Port for llama-server.")
    return parser.parse_args()


def load_entries(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            rows.append(payload)
    return rows


def start_server(port: int) -> subprocess.Popen[str]:
    if not LLAMA_SERVER.exists():
        raise FileNotFoundError(f"llama-server not found at {LLAMA_SERVER}")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
    cmd = [
        str(LLAMA_SERVER),
        "--model",
        str(MODEL_PATH),
        "--port",
        str(port),
        "--alias",
        "autolabel",
        "--ctx-size",
        "4096",
        "--temp",
        "0.2",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    base_url = f"http://127.0.0.1:{port}"
    for _ in range(60):
        try:
            resp = requests.get(f"{base_url}/v1/models", timeout=2)
            if resp.status_code == 200:
                time.sleep(15)
                return proc
        except requests.RequestException:
            time.sleep(1)
    proc.terminate()
    raise RuntimeError("Failed to start llama-server")


def stop_server(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


def run_llama(prompt: str, temp: float, max_tokens: int, port: int) -> str:
    base_url = f"http://127.0.0.1:{port}"
    payload = {
        "model": "autolabel",
        "messages": [
            {"role": "system", "content": "You convert user text into affect scores."},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "temperature": temp,
        "max_tokens": max_tokens,
    }
    for attempt in range(5):
        try:
            response = requests.post(f"{base_url}/v1/chat/completions", json=payload, timeout=180)
            response.raise_for_status()
            break
        except requests.HTTPError as exc:
            if response.status_code >= 500 and attempt < 4:
                time.sleep(2.0)
                continue
            raise exc
    else:
        raise RuntimeError("Failed to get completion after retries")
    data = response.json()
    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError("No choices returned")
    content = choices[0]["message"]["content"]
    return content


def extract_json(response: str) -> dict[str, Any] | None:
    match = JSON_PATTERN.search(response)
    if not match:
        print("Response missing JSON block:", response[:200], file=sys.stderr)
        return None
    snippet = match.group(0)
    try:
        payload = json.loads(snippet)
        return payload
    except json.JSONDecodeError:
        return None


def main() -> None:
    args = parse_args()
    entries = load_entries(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    processed = 0
    if args.resume and args.output.exists():
        processed = args.resume
    server = start_server(args.port)
    try:
        with args.output.open("a", encoding="utf-8") as out_handle:
            for idx, row in enumerate(entries):
                if idx < processed:
                    continue
                text = row.get("text", "")
                archetype = row.get("archetype", "")
                prompt = PROMPT_TEMPLATE.format(archetype=archetype, text=text)
                try:
                    response = run_llama(prompt, args.temp, args.max_tokens, args.port)
                except Exception as exc:
                    print(f"[{idx}] llama-server error: {exc}", file=sys.stderr)
                    time.sleep(1.0)
                    continue
                payload = extract_json(response)
                if not payload:
                    print(f"[{idx}] failed to parse JSON, skipping", file=sys.stderr)
                    continue
                record = {
                    "text": text,
                    "archetype": archetype,
                    "source": row.get("source"),
                    "auto_label": {
                        "valence": payload.get("valence"),
                        "intimacy": payload.get("intimacy"),
                        "tension": payload.get("tension"),
                        "confidence": payload.get("confidence"),
                        "tags": payload.get("tags"),
                        "reason": payload.get("reason"),
                    },
                }
                out_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                print(f"[{idx}] labeled archetype={archetype}")
    finally:
        stop_server(server)


if __name__ == "__main__":
    main()
