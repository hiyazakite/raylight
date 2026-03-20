from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from raylight.config import RaylightConfig

class AttentionBackend(ABC):
    """
    Abstract base class for attention backends.
    """

    @abstractmethod
    def create_attention(self, raylight_config: 'RaylightConfig', **kwargs) -> Callable:
        """
        Creates and returns the attention function.
        
        Args:
            raylight_config: The unified configuration object.
            **kwargs: Additional backend-specific configuration.
        """
        pass
