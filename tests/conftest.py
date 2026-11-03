from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def tiny_dir(tmp_path_factory) -> Path:
    env = os.environ.get("LOCALTRIO_TINY_DIR")
    if env and (Path(env) / "tiny-qwen3").exists():
        return Path(env)
    out = tmp_path_factory.mktemp("tiny")
    subprocess.run([sys.executable, str(ROOT / "tests" / "make_tiny_models.py"), str(out)], check=True)
    return out


@pytest.fixture(scope="session")
def text_model(tiny_dir) -> str:
    return str(tiny_dir / "tiny-qwen3")


@pytest.fixture(scope="session")
def vl_model(tiny_dir) -> str:
    return str(tiny_dir / "tiny-qwen35-vl")
