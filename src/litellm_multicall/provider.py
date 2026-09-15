from litellm import CustomLLM

class MulticallProvider(CustomLLM):
    model_name: str = "multicall"
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    async def acomplete(self, model, messages, **kwargs):
        raise NotImplementedError("Provider stub")
    
    async def astreaming(self, model, messages, **kwargs):
        raise NotImplementedError("Provider stub")

multicall_provider = MulticallProvider

