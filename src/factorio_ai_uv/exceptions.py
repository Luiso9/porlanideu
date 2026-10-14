"""
Custom exceptions for Factorio AI.
"""


class FactorioAIError(Exception):
    """Base exception for Factorio AI."""
    pass


class ConnectionError(FactorioAIError):
    """Failed to connect to Factorio."""
    pass


class RCONError(FactorioAIError):
    """RCON command execution failed."""
    pass


class ModNotFoundError(FactorioAIError):
    """Required mod not found or not loaded."""
    pass


class ParseError(FactorioAIError):
    """Failed to parse game state."""
    pass


class ActionError(FactorioAIError):
    """Action execution failed."""
    pass


class ScreenCaptureError(FactorioAIError):
    """Screen capture failed."""
    pass


class ModelError(FactorioAIError):
    """Model loading/saving failed."""
    pass


class DatasetError(FactorioAIError):
    """Dataset loading/saving failed."""
    pass