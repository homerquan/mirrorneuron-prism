class PrismError(Exception):
    """A sanitized, public error. Never include provider bodies or credentials."""

    def __init__(self, message, code="invalid_request", status=400, param=None):
        super().__init__(message)
        self.code = code
        self.status = status
        self.param = param

    def body(self):
        return {
            "error": {
                "message": str(self),
                "type": (
                    "invalid_request_error" if self.status < 500 else "server_error"
                ),
                "code": self.code,
                "param": self.param,
            }
        }
