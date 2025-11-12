"""Clone and build llama.cpp with GPU acceleration on Windows/WSL."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPO_DIR = PROJECT_ROOT / "third_party" / "llama.cpp"


class BuildError(RuntimeError):
    """Raised when a subprocess exits with a non-zero return code."""


def run(command: list[str], *, cwd: Path | None = None) -> None:
    """Execute a subprocess, raising BuildError on failure."""
    display = " ".join(command)
    print(f"[llama-build] {display}")
    result = subprocess.run(command, cwd=cwd, check=False)
    if result.returncode != 0:
        raise BuildError(f"command failed ({result.returncode}): {display}")


def ensure_repo(repo_dir: Path, *, branch: str) -> None:
    """Clone the llama.cpp repository if needed, otherwise fetch updates."""
    if repo_dir.exists():
        if not (repo_dir / ".git").exists():
            raise BuildError(f"expected git repository at {repo_dir}")
        run(["git", "fetch", "--prune"], cwd=repo_dir)
        run(["git", "checkout", branch], cwd=repo_dir)
        run(["git", "pull", "--ff-only"], cwd=repo_dir)
        return

    repo_dir.parent.mkdir(parents=True, exist_ok=True)
    run(
        [
            "git",
            "clone",
            "--depth",
            "1",
            "--branch",
            branch,
            "https://github.com/ggerganov/llama.cpp",
            str(repo_dir),
        ]
    )


def configure_cmake(
    repo_dir: Path,
    *,
    build_dir: Path,
    backend: str,
    generator: str | None,
    enable_curl: bool,
) -> None:
    """Configure CMake with the requested GPU backend."""
    cmake_args = [
        "cmake",
        "-S",
        str(repo_dir),
        "-B",
        str(build_dir),
        "-DLLAMA_BUILD_SERVER=ON",
    ]

    backend = backend.lower()
    backend_flag = backend
    if backend == "hipblas":
        cmake_args.extend(["-DGGML_HIP=ON", "-DLLAMA_HIPBLAS=ON"])
    elif backend == "vulkan":
        cmake_args.extend(["-DGGML_VULKAN=ON", "-DLLAMA_VULKAN=ON"])
    elif backend == "opencl":
        cmake_args.append("-DGGML_OPENCL=ON")
    else:
        raise ValueError(f"unsupported backend '{backend}'")

    if generator:
        cmake_args.extend(["-G", generator])

    if not enable_curl:
        cmake_args.append("-DLLAMA_CURL=OFF")

    # Use release builds by default.
    cmake_args.append("-DCMAKE_BUILD_TYPE=Release")
    run(cmake_args)


def build_targets(build_dir: Path, *, config: str) -> None:
    """Build the llama server binary."""
    run(["cmake", "--build", str(build_dir), "--config", config, "--target", "llama-server"])


def detect_generator() -> str | None:
    """Detect a suitable CMake generator on Windows."""
    if os.name != "nt":
        return None
    # Prefer Ninja if available since it simplifies HIP builds.
    ninja = shutil.which("ninja")
    if ninja:
        return "Ninja"
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build llama.cpp with GPU acceleration.")
    parser.add_argument("--repo-dir", type=Path, default=DEFAULT_REPO_DIR, help="Repository location.")
    parser.add_argument("--branch", default="master", help="Git branch or tag to checkout.")
    parser.add_argument(
        "--backend",
        choices=("hipblas", "vulkan", "opencl"),
        default="hipblas",
        help="GPU backend to enable.",
    )
    parser.add_argument(
        "--config",
        default="Release",
        help="CMake build configuration (Release, RelWithDebInfo, etc.).",
    )
    parser.add_argument(
        "--generator",
        default=None,
        help="Override the CMake generator (e.g. Ninja, Visual Studio 17 2022).",
    )
    parser.add_argument(
        "--enable-curl",
        action="store_true",
        help="Enable libcurl support (requires curl libraries to be installed).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_dir: Path = args.repo_dir
    build_dir = repo_dir / "build"
    generator = args.generator or detect_generator()

    ensure_repo(repo_dir, branch=args.branch)
    configure_cmake(
        repo_dir,
        build_dir=build_dir,
        backend=args.backend,
        generator=generator,
        enable_curl=args.enable_curl,
    )
    build_targets(build_dir, config=args.config)

    server_path = build_dir / args.config / "bin" / "llama-server.exe" if os.name == "nt" else build_dir / "bin" / "llama-server"
    print(f"[llama-build] llama-server binary is located at: {server_path}")


if __name__ == "__main__":
    try:
        main()
    except BuildError as exc:
        print(f"[llama-build] ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
