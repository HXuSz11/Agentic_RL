"""采样副本路由：前缀亲和 + 负载回退（不启动任何进程）。"""

from __future__ import annotations

from concurrent.futures import Future

from localtrio.config import EngineConfig
from localtrio.runtime import SamplerPool
from localtrio.sampling import _prefix_affinity


class FakeReplica:
    def __init__(self, name: str) -> None:
        self.name = name
        self.inflight = 0
        self.alive = True
        self.calls = 0

    def call(self, method: str, **kwargs) -> Future:
        self.calls += 1
        self.inflight += 1
        fut: Future = Future()
        fut.set_result(self.name)
        return fut


def test_affinity_routes_same_prefix_to_same_replica():
    replicas = [FakeReplica(f"r{i}") for i in range(3)]
    pool = SamplerPool("m", replicas, EngineConfig())
    key = _prefix_affinity([("text", list(range(600)))])
    # 多轮轨迹：前缀相同、后缀不断增长
    assert key == _prefix_affinity([("text", list(range(600)) + [1, 2, 3])])
    targets = {pool.call("generate", affinity=key).result() for _ in range(5)}
    assert len(targets) == 1


def test_affinity_falls_back_when_preferred_is_busy():
    replicas = [FakeReplica(f"r{i}") for i in range(3)]
    pool = SamplerPool("m", replicas, EngineConfig())
    key = 4  # 4 % 3 == 1 → 偏好 r1
    replicas[1].inflight = SamplerPool.AFFINITY_SLACK + 5
    assert pool.call("generate", affinity=key).result() != "r1"


def test_no_affinity_balances_load():
    replicas = [FakeReplica(f"r{i}") for i in range(3)]
    pool = SamplerPool("m", replicas, EngineConfig())
    for _ in range(9):
        pool.call("generate")
    assert [r.calls for r in replicas] == [3, 3, 3]
    assert _prefix_affinity([("image", b"x", "png", 4)]) is None


def test_default_layout_skips_gpus_of_explicit_engines(tmp_path, monkeypatch):
    from localtrio.config import load_config

    cfg_file = tmp_path / "localtrio.yaml"
    cfg_file.write_text("models:\n  Qwen/Qwen3.6-27B: {gpus: [2, 3], tensor_parallel_size: 2}\n")
    monkeypatch.setenv("LOCALTRIO_CONFIG", str(cfg_file))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3")
    monkeypatch.delenv("LOCALTRIO_SAMPLER_GPUS", raising=False)
    monkeypatch.delenv("LOCALTRIO_TRAINER_GPUS", raising=False)
    cfg = load_config()
    assert cfg.trainer.gpus == [0]
    assert cfg.models["default"].gpus == [1]
    assert cfg.engine_for("Qwen/Qwen3.6-27B").gpus == [2, 3]
