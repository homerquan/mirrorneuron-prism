"""Prism: bounded inference behind an OpenAI-compatible endpoint."""

__version__ = "0.2.0"


def create_app(config, **kwargs):
    """Construct the ASGI app without importing ML runtimes at package import."""
    from .api import create_app as factory

    return factory(config, **kwargs)


__all__ = ["__version__", "create_app"]
