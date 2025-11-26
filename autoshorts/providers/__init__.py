# -*- coding: utf-8 -*-
"""
Provider abstraction layer
✅ Abstract base classes for providers
✅ Factory pattern for provider creation
✅ Easy testing with mock providers
"""
from autoshorts.providers.base import (
    BaseTTSProvider,
    BaseVideoProvider,
    BaseAIProvider,
)
from autoshorts.providers.factory import ProviderFactory

__all__ = [
    "BaseTTSProvider",
    "BaseVideoProvider",
    "BaseAIProvider",
    "ProviderFactory",
]
