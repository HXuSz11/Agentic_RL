"""localtrio：在自己的 GPU 上运行的 PyTRIO 兼容训练 / 采样 API。

用法与 PyTRIO 相同，只需替换导入::

    import localtrio as trio

    service_client = trio.ServiceClient()
    training_client = service_client.create_lora_training_client(base_model="Qwen/Qwen3.5-4B", rank=32)
    sampling_client = training_client.save_weights_and_get_sampling_client()
    ...

训练：HF transformers + PEFT（LoRA），采样：vLLM（LoRA 热加载）。GPU 布局见
``localtrio.yaml`` / ``configs/*.yaml``。
"""

from . import types as types
from .config import configure as configure
from .config import load_config as load_config
from .errors import DependencyError as DependencyError
from .errors import PyTrioError as PyTrioError
from .errors import ValidationError as ValidationError
from .futures import APIFuture as APIFuture
from .sampling import SamplingClient as SamplingClient
from .service import ServiceClient as ServiceClient
from .training import TrainingClient as TrainingClient
from .types import AdamParams as AdamParams
from .types import Datum as Datum
from .types import ImageChunk as ImageChunk
from .types import ModelInput as ModelInput
from .types import SamplingParams as SamplingParams

__version__ = "0.1.0"


def get_model_dir(base_model: str) -> str:
    """返回 localtrio 为该基模实际使用的本地目录（必要时下载）。

    用它加载与训练 / 采样引擎完全一致的 tokenizer、image processor 或 chat template。
    """
    from .runtime import Runtime

    return Runtime.get().model_dir(base_model)


def get_image_processor(base_model: str) -> object:
    """返回与引擎一致的 image processor（多模态模型计算 ``ImageChunk.expected_tokens`` 用）。"""
    from .models import load_image_processor

    return load_image_processor(get_model_dir(base_model))


def shutdown() -> None:
    """立即释放所有引擎子进程（通常不需要手动调用，进程退出时会自动清理）。"""
    from .runtime import Runtime

    Runtime.reset()


def configure_observability(*args: object, **kwargs: object) -> None:  # pragma: no cover - 兼容占位
    """PyTRIO 的遥测接口；本地版不上报任何数据。"""


def shutdown_observability(*args: object, **kwargs: object) -> None:  # pragma: no cover - 兼容占位
    """PyTRIO 的遥测接口；本地版不上报任何数据。"""
