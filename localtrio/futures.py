"""可阻塞、可 await、可 map 的结果对象，语义与 PyTRIO 的 ``APIFuture`` 一致。"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Generator
from concurrent.futures import CancelledError
from concurrent.futures import Future as ConcurrentFuture
from typing import Generic, TypeVar

T = TypeVar("T")
U = TypeVar("U")


class APIFuture(Generic[T]):
    """包装 ``concurrent.futures.Future``。

    - 同步代码：``future.result()``
    - 异步代码：``await future``
    """

    def __init__(self, future: ConcurrentFuture[T]) -> None:
        self._future = future

    @staticmethod
    def completed(value: U) -> "APIFuture[U]":
        f: ConcurrentFuture[U] = ConcurrentFuture()
        f.set_result(value)
        return APIFuture(f)

    @staticmethod
    def failed(error: BaseException) -> "APIFuture[object]":
        f: ConcurrentFuture[object] = ConcurrentFuture()
        f.set_exception(error)
        return APIFuture(f)

    def result(self, timeout: float | None = None) -> T:
        return self._future.result(timeout=timeout)

    def done(self) -> bool:
        return self._future.done()

    def exception(self, timeout: float | None = None) -> BaseException | None:
        return self._future.exception(timeout=timeout)

    def cancel(self) -> bool:
        return self._future.cancel()

    def map(self, transform: Callable[[T], U]) -> "APIFuture[U]":
        mapped: ConcurrentFuture[U] = ConcurrentFuture()

        def resolve(source: ConcurrentFuture[T]) -> None:
            if source.cancelled():
                mapped.cancel()
                return
            try:
                mapped.set_result(transform(source.result()))
            except CancelledError:
                mapped.cancel()
            except BaseException as error:  # noqa: BLE001 - 透传所有异常
                if not mapped.done():
                    mapped.set_exception(error)

        self._future.add_done_callback(resolve)
        return APIFuture(mapped)

    def __await__(self) -> Generator[object, None, T]:
        return asyncio.wrap_future(self._future).__await__()
