"""
Factorio Autonomous AI Package.

A reinforcement learning agent that learns to play Factorio using:
- Kruise_Kontrol_Remote for reliable character actions
- RCON for game state queries
- Screen capture for visual observations (CNN input)
- PPO (Proximal Policy Optimization) for learning

Usage:
    python -m factorio_ai_uv --test-only   # Test environment
    python -m factorio_ai_uv               # Train agent
"""

from .config import (
    CaptureConfig,
    TrainingConfig,
    RewardWeights,
    PathConfig,
    CAPTURE_CONFIG,
    TRAINING_CONFIG,
    REWARD_WEIGHTS,
    PATH_CONFIG,
    ACTION_NOOP,
    ACTION_MOVE_NORTH,
    ACTION_MOVE_SOUTH,
    ACTION_MOVE_EAST,
    ACTION_MOVE_WEST,
    ACTION_BUILD,
    ACTION_MINE,
    ACTION_RESEARCH,
    ACTION_CRAFT,
    NUM_ACTIONS,
    ACTION_NAMES,
    OBS_SHAPE,
    OBS_DTYPE,
)

from .client import FactorioClient
from .env import FactorioEnv
from .screen_capture import FactorioScreenCapture
from .exceptions import (
    FactorioAIError,
    ConnectionError,
    RCONError,
    ModNotFoundError,
    ParseError,
    ActionError,
    ScreenCaptureError,
    ModelError,
    DatasetError,
)

__version__ = "0.1.0"
__all__ = [
    # Config
    "CaptureConfig",
    "TrainingConfig", 
    "RewardWeights",
    "PathConfig",
    "CAPTURE_CONFIG",
    "TRAINING_CONFIG",
    "REWARD_WEIGHTS",
    "PATH_CONFIG",
    # Action constants
    "ACTION_NOOP",
    "ACTION_MOVE_NORTH",
    "ACTION_MOVE_SOUTH",
    "ACTION_MOVE_EAST",
    "ACTION_MOVE_WEST",
    "ACTION_BUILD",
    "ACTION_MINE",
    "ACTION_RESEARCH",
    "ACTION_CRAFT",
    "NUM_ACTIONS",
    "ACTION_NAMES",
    # Observation
    "OBS_SHAPE",
    "OBS_DTYPE",
    # Components
    "FactorioClient",
    "FactorioEnv",
    "FactorioScreenCapture",
    # Exceptions
    "FactorioAIError",
    "ConnectionError",
    "RCONError",
    "ModNotFoundError",
    "ParseError",
    "ActionError",
    "ScreenCaptureError",
    "ModelError",
    "DatasetError",
]