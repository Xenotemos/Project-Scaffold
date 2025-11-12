"""Bootstrap virtual environment and install dependencies for the Living AI project."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENV_DIR = PROJECT_ROOT / ".venv"
REQUIREMENTS_FILE = PROJECT_ROOT / "requirements.txt"


def run_command(command: list[str], *, cwd: Path | None = None) -> None:
    """Run a subprocess, raising if it fails."""
    print(f"[setup] running: {' '.join(command)}")
    subprocess.run(command, check=True, cwd=cwd or PROJECT_ROOT)


def create_virtualenv(env_dir: Path, python_exe: str) -> None:
    """Create the virtual environment if it does not exist."""
    if env_dir.exists():
        print(f"[setup] virtual environment already exists at {env_dir}")
        return
    print(f"[setup] creating virtual environment at {env_dir}")
    run_command([python_exe, "-m", "venv", str(env_dir)])


def install_dependencies(env_dir: Path, *, upgrade_pip: bool) -> None:
    """Install project dependencies using the venv's pip executable."""
    if not REQUIREMENTS_FILE.exists():
        raise FileNotFoundError(f"requirements file not found at {REQUIREMENTS_FILE}")

    candidate_paths = []
    if os.name == "nt":
        candidate_paths.append(env_dir / "Scripts" / "pip.exe")
        candidate_paths.append(env_dir / "Scripts" / "pip")  # fallback for WSL-created envs
    candidate_paths.append(env_dir / "bin" / "pip")

    pip_path = next((path for path in candidate_paths if path.exists()), None)
    if pip_path is None:
        raise FileNotFoundError(
            "pip executable not found in the virtual environment. "
            f"Checked: {', '.join(str(path) for path in candidate_paths)}"
        )

    if pip_path.name == "pip.exe":
        python_path = pip_path.with_name("python.exe")
    elif pip_path.name == "pip":
        python_path = pip_path.with_name("python")
    else:
        python_path = pip_path.parent / "python"

    if upgrade_pip:
        run_command([str(python_path), "-m", "pip", "install", "--upgrade", "pip"])

    run_command([str(python_path), "-m", "pip", "install", "-r", str(REQUIREMENTS_FILE)])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Set up a virtual environment for the project.")
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python interpreter to use for creating the virtual environment.",
    )
    parser.add_argument(
        "--env-dir",
        type=Path,
        default=DEFAULT_ENV_DIR,
        help=f"Virtual environment directory (default: {DEFAULT_ENV_DIR})",
    )
    parser.add_argument(
        "--no-upgrade-pip",
        action="store_true",
        help="Skip upgrading pip before installing dependencies.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env_dir: Path = args.env_dir
    python_exe: str = args.python
    upgrade_pip = not args.no_upgrade_pip

    create_virtualenv(env_dir, python_exe)
    install_dependencies(env_dir, upgrade_pip=upgrade_pip)
    print("[setup] environment ready.")


if __name__ == "__main__":
    main()
