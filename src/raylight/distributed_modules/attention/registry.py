from typing import Dict, Type
from .interface import AttentionBackend

class AttentionRegistry:
    _backends: Dict[str, Type[AttentionBackend]] = {}

    @classmethod
    def register(cls, name: str, backend_cls: Type[AttentionBackend]):
        """Register a new attention backend class."""
        cls._backends[name] = backend_cls

    @classmethod
    def get(cls, name: str) -> AttentionBackend:
        """Get an instance of the requested backend."""
        backend_cls = cls._backends.get(name)
        if not backend_cls:
            raise ValueError(f"Attention backend '{name}' not found. Available: {list(cls._backends.keys())}")
        return backend_cls()
