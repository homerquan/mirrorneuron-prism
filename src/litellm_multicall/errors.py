"""Stable CLI/service error taxonomy."""


class PrismError(Exception):
    def __init__(self, message: str, code: int = 2, kind: str = "invalid_input"):
        super().__init__(message)
        self.code = code
        self.kind = kind
