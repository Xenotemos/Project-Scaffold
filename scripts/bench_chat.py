from __future__ import annotations

import asyncio
import re
from typing import Any

import httpx

import main


async def run_bench() -> list[dict[str, Any]]:
    transport = httpx.ASGITransport(app=main.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://bench") as client:
        await main._reload_runtime_settings()
        turns = [
            "hey there",
            "can you briefly explain how streaming works here?",
            "ok, now give me a two-sentence example reply to a user question about setup",
            "nice, and keep it casual",
            "thanks!",
        ]
        findings: list[dict[str, Any]] = []
        for i, msg in enumerate(turns, 1):
            res = await client.post("/chat", json={"message": msg})
            res.raise_for_status()
            data = res.json()
            reply: str = data.get("reply", "")
            source = data.get("source")
            lp = (data.get("length_plan") or {}).get("label")
            artifacts = {
                "has_User_label": bool(re.search(r"(^|\n)\s*User:\s", reply)),
                "has_Assistant_label": bool(re.search(r"(^|\n)\s*Assistant:\s", reply)),
                "has_ai_local": "ai[local]" in reply,
                "has_json": reply.strip().startswith("{") or reply.strip().startswith("["),
            }
            findings.append(
                {
                    "i": i,
                    "msg": msg,
                    "source": source,
                    "length": len(reply),
                    "length_plan": lp,
                    "artifacts": artifacts,
                    "snippet": reply[:240],
                }
            )
        return findings


def main_cli() -> None:
    out = asyncio.run(run_bench())
    for row in out:
        print(row)


if __name__ == "__main__":
    main_cli()
