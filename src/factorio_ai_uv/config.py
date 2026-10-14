"""
Configuration and constants for Factorio AI.
"""

from dataclasses import dataclass
from pathlib import Path
import numpy as np


@dataclass(frozen=True)
class CaptureConfig:
    """Screen capture configuration."""
    region: dict | None = None  # {"top": int, "left": int, "width": int, "height": int}
    size: tuple[int, int] = (84, 84)  # Resized observation shape (width, height)
    interval: int = 2  # Capture every N ticks


@dataclass(frozen=True)
class TrainingConfig:
    """Training loop configuration."""
    max_episode_steps: int = 5000
    train_steps_per_iter: int = 1000
    n_iterations: int = 30
    dataset_max_size: int = 20000


@dataclass(frozen=True)
class RewardWeights:
    """Reward function weights."""
    survival: float = 1.0      # Heaviest - per-tick survival bonus
    health: float = 1.0        # Health change (negative when damaged)
    tech: float = 1.0          # Technology progress
    damage_penalty: float = -0.5  # Per-tick damage penalty


@dataclass(frozen=True)
class PathConfig:
    """File system paths."""
    dataset: Path = Path("factorio_dataset.json")
    model_dir: Path = Path("factorio_ai_model")
    logs_dir: Path = Path("logs")
    pretrained_model: Path = Path("pretrained_ppo_factorio")


# Global configs (can be overridden)
CAPTURE_CONFIG = CaptureConfig()
TRAINING_CONFIG = TrainingConfig()
REWARD_WEIGHTS = RewardWeights()
PATH_CONFIG = PathConfig()

# Export commonly used constants
MAX_EPISODE_STEPS = TRAINING_CONFIG.max_episode_steps

# Action space constants
ACTION_NOOP = 0
ACTION_MOVE_NORTH = 1
ACTION_MOVE_SOUTH = 2
ACTION_MOVE_EAST = 3
ACTION_MOVE_WEST = 4
ACTION_BUILD = 5
ACTION_MINE = 6
ACTION_RESEARCH = 7
ACTION_CRAFT = 8
NUM_ACTIONS = 9

ACTION_NAMES = {
    ACTION_NOOP: "noop",
    ACTION_MOVE_NORTH: "move_N",
    ACTION_MOVE_SOUTH: "move_S",
    ACTION_MOVE_EAST: "move_E",
    ACTION_MOVE_WEST: "move_W",
    ACTION_BUILD: "build_at_cursor",
    ACTION_MINE: "mine_at_cursor",
    ACTION_RESEARCH: "research",
    ACTION_CRAFT: "craft",
}

# Observation shape: (H, W, C) = (84, 84, 3) for CnnPolicy (uint8 [0,255])
OBS_SHAPE = (CAPTURE_CONFIG.size[1], CAPTURE_CONFIG.size[0], 3)
OBS_DTYPE = np.uint8
OBS_LOW = 0
OBS_HIGH = 255