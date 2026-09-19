from litellm import CustomLLM


class MulticallProvider(CustomLLM):
    model_name: str = "multicall"

    def __init__(self) -> None:
        super().__init__()

    async def acompletion(self, model, messages, **kwargs):
        # Step 01 baseline: fail closed until the engine is implemented.
        raise NotImplementedError("MulticallProvider.acompletion is not implemented yet")

    async def astreaming(self, model, messages, **kwargs):
        # Step 01 baseline: fail closed until streaming is implemented.
        raise NotImplementedError("MulticallProvider.astreaming is not implemented yet")

    def completion(self, model, messages, **kwargs):
        raise NotImplementedError("MulticallProvider.completion is not implemented yet")

    def streaming(self, model, messages, **kwargs):
        raise NotImplementedError("MulticallProvider.streaming is not implemented yet")


multicall_provider = MulticallProvider()

