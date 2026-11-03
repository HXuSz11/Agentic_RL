"""引擎子进程的生命周期与采样引擎配置的规整。

- worker 就绪后异常退出（OOM / SIGKILL）时，即使 vLLM fork 出的 EngineCore 还握着 RPC socket、
  主进程收不到 EOF，在途调用也要报错，同组子进程也要被回收；
- ``max_lora_rank`` 向上取到 vLLM 接受的值。
"""

from __future__ import annotations

import dataclasses
import os
import shutil
import signal
import time
from pathlib import Path

import pytest

from localtrio.config import EngineConfig, load_config, round_lora_rank
from localtrio.rpc import WorkerError, WorkerHandle

ROOT = Path(__file__).resolve().parent.parent

# 模拟 vLLM 0.33 默认用 fork 启动 EngineCore：子进程继承 worker 的 RPC socket，并且一直活着
FORKING_PATCH = '''

def _forking_from_engine_args(cls, engine_args, **kwargs):
    import os, time
    pid = os.fork()
    if pid == 0:
        time.sleep(600)
        os._exit(0)
    with open(os.environ["LOCALTRIO_TEST_FORK_PID_FILE"], "w") as handle:
        handle.write(str(pid))
    return cls(engine_args)


AsyncLLM.from_engine_args = classmethod(_forking_from_engine_args)
'''


def _running(pid: int) -> bool:
    try:
        state = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
    except (OSError, IndexError):
        return False
    return state != "Z"


@pytest.mark.skipif(not hasattr(os, "fork") or not Path("/proc/self/stat").exists(), reason="需要 Linux")
def test_dead_sampler_with_forked_engine_core_fails_calls(tmp_path, monkeypatch, text_model):
    fake = tmp_path / "fake_vllm"
    shutil.copytree(ROOT / "tests" / "fake_vllm", fake, ignore=shutil.ignore_patterns("__pycache__"))
    with (fake / "vllm" / "v1" / "engine" / "async_llm.py").open("a") as handle:
        handle.write(FORKING_PATCH)
    pid_file = tmp_path / "engine_core.pid"
    monkeypatch.setenv("PYTHONPATH", str(fake))
    monkeypatch.setenv("LOCALTRIO_TEST_FORK_PID_FILE", str(pid_file))
    spec = {
        "base_model": text_model,
        "model_dir": text_model,
        "backend": "vllm",
        "engine": dataclasses.asdict(EngineConfig()),
    }
    worker = WorkerHandle("sampler[fork-test]", "sampler", spec, gpus=[], log_dir=tmp_path / "logs", start_timeout=300)
    child = None
    try:
        worker.wait_ready(timeout=300)
        child = int(pid_file.read_text())
        assert _running(child)
        os.kill(worker.proc.pid, signal.SIGKILL)
        worker.proc.wait(timeout=10)
        fut = worker.call("generate", chunks=[], num_samples=1, params={})
        with pytest.raises(WorkerError, match="进程已退出"):
            fut.result(timeout=30)
        assert not worker.alive
        deadline = time.time() + 15
        while _running(child) and time.time() < deadline:
            time.sleep(0.2)
        assert not _running(child), "fork 出的 EngineCore 没有被回收"
    finally:
        if child is not None and _running(child):
            os.kill(child, signal.SIGKILL)
        worker.shutdown(timeout=5)


def test_max_lora_rank_rounds_up_to_vllm_choices(tmp_path, monkeypatch):
    assert [round_lora_rank(r) for r in (1, 4, 8, 48, 64, 96, 128, 200, 1024)] == [1, 8, 8, 64, 64, 128, 128, 256, 1024]
    cfg_file = tmp_path / "localtrio.yaml"
    cfg_file.write_text("gpus: []\nmodels:\n  default: {max_lora_rank: 96}\n  Qwen/Qwen3.6-27B: {max_lora_rank: 32}\n")
    monkeypatch.setenv("LOCALTRIO_CONFIG", str(cfg_file))
    cfg = load_config()
    assert cfg.engine_for("Qwen/Qwen3.5-4B").max_lora_rank == 128
    assert cfg.engine_for("Qwen/Qwen3.6-27B").max_lora_rank == 32
