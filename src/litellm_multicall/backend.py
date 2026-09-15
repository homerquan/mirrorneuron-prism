from typing import Any, Protocol
from litellm import ModelResponse

class CompletionBackend(Protocol):
    async def complete(
        self,
        *,
        model_group: str,
        messages: list[dict[str, Any]],
        parameters: dict[str, Any],
        context: Any,
    ) -> ModelResponse: ...

class ProxyRouterBackend:
    def __init__(self, router):
        self.router = router

    async def complete(self, *, model_group: str, messages: list[dict[str, Any]], parameters: dict[str, Any], context: Any) -> ModelResponse:
        # Placeholder for actual router call
        raise NotImplementedError("ProxyRouterBackend requires initialized LiteLLM router")

