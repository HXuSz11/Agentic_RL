"""localtrio 异常类型（名称与 PyTRIO 保持一致，方便 ``except trio.PyTrioError``）。"""

from __future__ import annotations


class PyTrioError(Exception):
    """所有 localtrio 错误的基类（沿用 PyTRIO 的名字）。"""

    def __init__(self, message: str, *, code: str | None = None, details: dict | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.details = details or {}


LocalTrioError = PyTrioError


class ValidationError(PyTrioError, ValueError):
    """输入不合法。"""


class DependencyError(PyTrioError, ImportError):
    """缺少依赖。"""
