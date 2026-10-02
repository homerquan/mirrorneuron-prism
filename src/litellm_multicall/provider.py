"""Stable LiteLLM entry point; unavailable inference must fail closed."""

import os

from litellm import CustomLLM

from .config import load_policy_config


def load_runtime_config():
    path = os.environ.get("PRISM_POLICY_CONFIG") or os.environ.get("MULTICALL_CONFIG")
    if not path:
        raise RuntimeError("PRISM_POLICY_CONFIG must name an explicit policy file")
    return load_policy_config(path)


class MulticallProvider(CustomLLM):
    async def acompletion(self, model, messages, **kwargs):
        load_runtime_config()
        raise NotImplementedError("Prism generative execution is not implemented (P3)")

    async def acomplete(self, model, messages, **kwargs):
        return await self.acompletion(model, messages, **kwargs)

    async def astreaming(self, model, messages, **kwargs):
        load_runtime_config()
        raise NotImplementedError("Prism streaming execution is not implemented")

    def completion(self, *args, **kwargs):
        raise NotImplementedError("Prism synchronous execution is not implemented")


multicall_provider = MulticallProvider()
