class ExecutionEngine:
    def __init__(self, backend):
        self.backend = backend

    async def run(self, request_context, policy):
        raise NotImplementedError("Prism generative execution is not implemented (P3)")
