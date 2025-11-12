"""Attached-mode launcher for boot_system.

Wraps scripts.boot_system.main while forcing --stay-attached unless the caller
explicitly provides an override. This allows us to package a dedicated .exe
that always stays attached for live log viewing.
"""

from __future__ import annotations

import asyncio
import sys
from typing import Iterable

from boot_system import main as boot_main


def _merge_args(user_args: Iterable[str]) -> list[str]:
    args = list(user_args)
    if not any(arg.startswith("--stay-attached") for arg in args):
        args.insert(0, "--stay-attached")
    return args


if __name__ == "__main__":
    combined = _merge_args(sys.argv[1:])
    raise SystemExit(asyncio.run(boot_main(combined)))
