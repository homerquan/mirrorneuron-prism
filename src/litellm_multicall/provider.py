from litellm import CustomLLM
from litellm.types import ModelResponse, ChatCompletionResponseMessage

class MulticallProvider(CustomLLM):
    model_name: str = "multicall"
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    async def acomplete(self, model, messages, **kwargs):
        # Minimal stub that returns a dummy response so the proxy can start
        return ModelResponse(
            choices=[{
                "message": ChatCompletionResponseMessage(role="assistant", content="stub response"),
                "finish_reason": "stop",
                "index": 0
            }],
            created=0,
            model=model,
            usage=None
        )
    
    async def astreaming(self, model, messages, **kwargs):
        # Streaming stub
        return self.acomplete(model, messages, **kwargs)

multicall_provider = MulticallProvider

