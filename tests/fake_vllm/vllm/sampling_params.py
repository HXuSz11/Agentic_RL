from __future__ import annotations

from enum import Enum


class RequestOutputKind(Enum):
    CUMULATIVE = 0
    DELTA = 1
    FINAL_ONLY = 2
