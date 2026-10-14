"""
Factorio Gym environment using Kruise_Kontrol_Remote and RCON.
"""

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .client import FactorioClient
from .config import (
    CaptureConfig,
    OBS_SHAPE,
    OBS_DTYPE,
    MAX_EPISODE_STEPS,
    REWARD_WEIGHTS,
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
    OBS_LOW,
    OBS_HIGH,
)
from .exceptions import ActionError, ScreenCaptureError
from .screen_capture import FactorioScreenCapture

logger = logging.getLogger(__name__)


@dataclass
class RewardInfo:
    """Reward breakdown for logging."""
    survival_reward: float = 0.0
    health_reward: float = 0.0
    tech_reward: float = 0.0
    damage_penalty: float = 0.0
    total: float = 0.0


class FactorioEnv(gym.Env):
    """
    Gymnasium environment for Factorio using:
    - Kruise_Kontrol_Remote for reliable character actions
    - RCON for game state queries
    - Screen capture for visual observations (CNN input)
    
    Observation: 84x84 RGB screen capture (CHW format, float32 [0,1])
    Action: Discrete(9) - noop, move N/S/E/W, build, mine, research, craft
    Reward: Survival-heavy with health/tech bonuses
    """
    
    metadata = {"render_modes": []}

    def __init__(
        self,
        capture: FactorioScreenCapture | None = None,
        capture_config: CaptureConfig | None = None,
        rcon_host: str = "127.0.0.1",
        rcon_port: int = 27015,
        rcon_password: str = "factorio_ai_password",
        max_episode_steps: int = MAX_EPISODE_STEPS,
    ) -> None:
        """
        Initialize Factorio environment.
        
        Args:
            capture: Pre-configured screen capture (optional)
            capture_config: Capture configuration (optional)
            rcon_host: RCON host
            rcon_port: RCON port
            rcon_password: RCON password
            max_episode_steps: Maximum steps per episode
        """
        super().__init__()

        # Screen capture for visual observation
        if capture is None:
            capture = FactorioScreenCapture(capture_config or CaptureConfig())
        self.capture = capture

        # Factorio client
        self.client = FactorioClient(
            host=rcon_host,
            port=rcon_port,
            password=rcon_password,
        )

        # Gym spaces
        self.action_space = spaces.Discrete(NUM_ACTIONS)
        self.observation_space = spaces.Box(
            low=OBS_LOW, high=OBS_HIGH, shape=OBS_SHAPE, dtype=OBS_DTYPE
        )

        # Episode state
        self._max_episode_steps = max_episode_steps
        self._total_steps = 0
        self._episode_start_time = 0.0
        self._last_player_health = 100
        self._last_tech_count = 0
        self._episode_reward = 0.0
        self._current_action = ACTION_NOOP

    def connect(self) -> bool:
        """Connect to Factorio."""
        return self.client.connect()

    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
        """Reset environment for new episode."""
        super().reset(seed=seed)
        
        self._total_steps = 0
        self._episode_start_time = time.time()
        self._last_player_health = 100
        self._episode_reward = 0.0
        self._current_action = ACTION_NOOP

        # Cancel any pending job
        if self.client.is_connected():
            self.client.cancel()

        obs = self.capture.capture()
        return obs, {}

    def step(self, action: int | np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        Execute one step in the environment.
        
        Args:
            action: Action index (0-8) or array
            
        Returns:
            (observation, reward, terminated, truncated, info)
        """
        # Handle VecEnv array input
        if isinstance(action, (list, tuple, np.ndarray)):
            action = int(action[0]) if len(action) > 0 else ACTION_NOOP

        # Validate action
        if not 0 <= action < NUM_ACTIONS:
            logger.warning(f"Invalid action {action}, defaulting to NOOP")
            action = ACTION_NOOP

        self._total_steps += 1
        self._current_action = action

        # Execute action
        action_name = self._execute_action(action)

        # Capture new frame (visual observation)
        obs = self.capture.capture()

        # Compute reward from game state
        reward, reward_info = self._compute_reward()

        # Check done conditions
        terminated = self._total_steps >= self._max_episode_steps
        truncated = False

        info = {
            "action": action_name,
            "step": self._total_steps,
            "reward": reward,
            **reward_info,
        }

        return obs, reward, terminated, truncated, info

    def _execute_action(self, action: int) -> str:
        """
        Execute action via client.
        
        Args:
            action: Action index
            
        Returns:
            Action name for logging
        """
        action_names = {
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

        if not self.client.is_connected():
            return action_names.get(action, "unknown") + "_disconnected"

        if action == ACTION_NOOP:
            self.client.cancel()
            return "cancel"

        # Get player position for relative movement
        pos_resp = self.client._send_silent(
            'rcon.print(helpers.table_to_json(game.players[1].position))'
        )
        match = re.search(r'"x":([-\d.]+),"y":([-\d.]+)', pos_resp)
        if match:
            px, py = float(match.group(1)), float(match.group(2))
        else:
            px, py = 0.0, 0.0

        try:
            # Movement actions - walk to relative position
            if action == ACTION_MOVE_NORTH:
                self.client.start_job(1, {"x": px, "y": py - 10}, "nil")
                return "move_N"
            elif action == ACTION_MOVE_SOUTH:
                self.client.start_job(1, {"x": px, "y": py + 10}, "nil")
                return "move_S"
            elif action == ACTION_MOVE_EAST:
                self.client.start_job(1, {"x": px + 10, "y": py}, "nil")
                return "move_E"
            elif action == ACTION_MOVE_WEST:
                self.client.start_job(1, {"x": px - 10, "y": py}, "nil")
                return "move_W"

            # Build at cursor position
            elif action == ACTION_BUILD:
                self.client.start_job(1, {"x": px, "y": py}, "player.selected")
                return "build_at_cursor"

            # Mine at cursor
            elif action == ACTION_MINE:
                self.client.start_job(1, {"x": px, "y": py}, "player.selected")
                return "mine_at_cursor"

            # Research (via RCON direct)
            elif action == ACTION_RESEARCH:
                self.client.add_research("automation")
                return "research"

            # Craft (via RCON direct)
            elif action == ACTION_CRAFT:
                self.client.craft_gears()
                return "craft"

        except Exception as e:
            logger.error(f"Action {action} failed: {e}")
            return action_names.get(action, "unknown") + "_error"

        return "unknown"

    def _compute_reward(self) -> tuple[float, dict[str, float]]:
        """
        Compute reward from game state.
        
        Returns:
            (total_reward, reward_info_dict)
        """
        reward_info = {
            "survival_reward": 0.0,
            "health_reward": 0.0,
            "tech_reward": 0.0,
            "damage_penalty": 0.0,
        }

        if not self.client.is_connected():
            reward_info["survival_reward"] = 1.0
            return 1.0, reward_info

        try:
            state = self.client.get_game_state()
            if not state:
                reward_info["survival_reward"] = 1.0
                return 1.0, reward_info

            player = state.get("player", {})
            force = state.get("force", {})

            # 1. Survival reward (heaviest)
            reward_info["survival_reward"] = 1.0

            # 2. Health change
            health = player.get("health", 0)
            if health < self._last_player_health:
                damage = self._last_player_health - health
                reward_info["damage_penalty"] = -damage * 0.5
                reward_info["health_reward"] = -damage * 0.5
            self._last_player_health = health

            # 4. Technology progress
            techs = force.get("technologies", {})
            researched_count = sum(1 for t in techs.values() if t.get("researched"))
            if researched_count > self._last_tech_count:
                reward_info["tech_reward"] = (researched_count - self._last_tech_count) * 10.0
            self._last_tech_count = researched_count

        except Exception as e:
            logger.error(f"Reward computation error: {e}")

        # Sum weighted rewards
        total_reward = (
            reward_info["survival_reward"] * REWARD_WEIGHTS.survival +
            reward_info["health_reward"] * REWARD_WEIGHTS.health +
            reward_info["tech_reward"] * REWARD_WEIGHTS.tech +
            reward_info["damage_penalty"] * REWARD_WEIGHTS.damage_penalty
        )

        return total_reward, reward_info

    def close(self) -> None:
        """Close environment and cleanup."""
        self.client.close()
        logger.info("Environment closed")


# Test function
if __name__ == "__main__":
    from .screen_capture import FactorioScreenCapture
    from .config import CaptureConfig

    logging.basicConfig(level=logging.INFO)
    
    print("Testing FactorioEnv...")
    capture = FactorioScreenCapture(CaptureConfig())
    env = FactorioEnv(capture=capture)

    if env.connect():
        print("Environment connected!")
        obs, _ = env.reset()
        print(f"Initial obs shape: {obs.shape}")

        for action in [1, 5, 7, 8, 0]:
            obs, reward, done, truncated, info = env.step(action)
            print(f"Action: {info.get('action')}, Reward: {reward:.2f}")
            time.sleep(1)

        env.close()
        print("Test complete!")
    else:
        print("Could not connect to Factorio")