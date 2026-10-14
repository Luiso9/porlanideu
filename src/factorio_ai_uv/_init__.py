"""Factorio Autonomous AI - Screen Vision based Reinforcement Learning."""

__version__ = "0.1.0"
__description__ = "AI that learns to play Factorio through screen vision and self-directed exploration"

from .main import FactorioAIAgent, main

__all__ = ["FactorioAIAgent", "main"]