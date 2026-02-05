from abc import ABC, abstractmethod
from typing import Callable

class AttentionBackend(ABC):
    """
    Abstract base class for attention backends.
    """

    @abstractmethod
    def create_attention(self, attn_type: str, sync_ulysses: bool, **kwargs) -> Callable:
        """
        Creates and returns the attention function.
        
        Args:
            attn_type: The type of attention optimization (e.g., 'FLASH_ATTN', 'SAGE_FP8').
            sync_ulysses: Whether to use synchronous Ulysses parallelism.
            **kwargs: Additional backend-specific configuration.
        """
        pass
