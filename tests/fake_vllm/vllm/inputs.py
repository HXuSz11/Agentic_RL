from __future__ import annotations

from typing import Any, NotRequired, TypedDict


class TokensPrompt(TypedDict):
    prompt_token_ids: list[int]
    multi_modal_data: NotRequired[dict[str, Any] | None]
