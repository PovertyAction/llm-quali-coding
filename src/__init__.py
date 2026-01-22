"""LLM Qualitative Coding Basics.

This package contains small, reusable helpers used by the runnable scripts in `examples/`.
"""

from .openai_client import get_client, load_config

__all__ = ["get_client", "load_config"]
