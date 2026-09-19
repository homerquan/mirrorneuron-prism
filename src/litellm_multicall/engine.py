from litellm_multicall.types import ExecutionResult


class ExecutionEngine:
    """Bounded per-request execution (Step 01 baseline: NOT implemented).

    Fail-closed stub: :meth:`run` raises :class:`NotImplementedError` so a
    miswired provider can never return an empty ``ExecutionResult`` as if it
    were a successful inference. Strategy execution lands here in later
    steps and must perform all model-role calls through the injected
    ``backend`` (the same LiteLLM Router instance).
    """

    def __init__(self, backend):
        self.backend = backend

    async def run(self, request_context, policy) -> ExecutionResult:
        raise NotImplementedError(
            "ExecutionEngine.run is not implemented yet (Step 01 baseline)"
        )
