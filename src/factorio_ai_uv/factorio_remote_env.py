"""
Factorio Environment using Remote API for game state and actions.
Screen capture still used for visual observation (CNN input).
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .factorio_remote import FactorioRemoteAPI
from .config import FactorioScreenCapture, CAPTURE_SIZE, CAPTURE_REGION, CAPTURE_INTERVAL, MAX_EPISODE_STEPS


class FactorioRemoteEnv(gym.Env):
    """
    Gymnasium environment using Factorio Remote API for:
    - Real game state (player, inventory, entities, tech, production)
    - Reliable command execution (move, build, mine, research, craft)
    - Structured rewards from actual game data
    
    Screen capture still used for visual observation (CNN input).
    """

    metadata = {"render_modes": []}

    def __init__(self, capture: FactorioScreenCapture = None, rcon_password: str = "factorio_ai_password"):
        super().__init__()

        # Screen capture for visual observation
        if capture is None:
            capture = FactorioScreenCapture(
                region=CAPTURE_REGION,
                size=CAPTURE_SIZE,
                interval=CAPTURE_INTERVAL
            )
        self.capture = capture

        # Remote API client
        self.api = FactorioRemoteAPI(password=rcon_password)
        self.connected = False

        # Action space: 0=noop, 1=move_N, 2=move_S, 3=move_E, 4=move_W,
        # 5=build, 6=mine, 7=research, 8=craft
        self.action_space = spaces.Discrete(9)

        # Observation: (C, H, W) in [0, 1] float32
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(3, CAPTURE_SIZE[1], CAPTURE_SIZE[0]), dtype=np.float32
        )

        # Internal state
        self.total_ticks = 0
        self.prev_tick = 0
        self.last_player_health = 100
        self.last_items_produced = {}
        self.last_tech_count = 0
        self.last_entity_count = 0
        self.episode_start_tick = 0
        self.current_action = 0

        # Connect to Factorio
        self._connect()

    def _connect(self):
        """Connect to Factorio Remote API."""
        print("Connecting to Factorio Remote API...")
        self.connected = self.api.connect()
        if not self.connected:
            print("WARNING: Could not connect to Factorio Remote API. Falling back to placeholders.")
        else:
            print("Factorio Remote API connected successfully!")

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.total_ticks = 0
        self.prev_tick = 0
        self.last_player_health = 100
        self.last_items_produced = {}
        self.last_tech_count = 0
        self.last_entity_count = 0
        self.episode_start_tick = 0
        self.current_action = 0

        # Release any movement commands
        if self.connected:
            self.api.execute_command({"type": "stop"})

        obs = self.capture.capture()
        return obs, {}

    def step(self, action):
        self.total_ticks += 1

        # VecEnv passes action as list/array; extract scalar
        if isinstance(action, (list, tuple, np.ndarray)):
            action = int(action[0]) if len(action) > 0 else 0

        self.current_action = action

        # Execute action via Remote API
        action_name = self._execute_action(action)

        # Capture new frame (visual observation)
        obs = self.capture.capture()

        # Compute reward from REAL game state
        reward, reward_info = self._compute_reward()

        # Check done conditions
        done = self.total_ticks >= MAX_EPISODE_STEPS

        # Info dict
        info = {
            "action": action_name,
            "tick": self.total_ticks,
            "reward": reward,
            **reward_info,
        }

        return obs, reward, done, False, info

    def _execute_action(self, action: int) -> str:
        """Execute action via Remote API."""
        action_names = {
            0: "noop",
            1: "move_N",
            2: "move_S",
            3: "move_E",
            4: "move_W",
            5: "build",
            6: "mine",
            7: "research",
            8: "craft",
        }

        if not self.connected:
            return action_names.get(action, "unknown")

        cmd = None

        if action == 0:  # noop
            return "noop"

        elif action == 1:  # move North
            cmd = {"type": "move", "direction": "north"}
        elif action == 2:  # move South
            cmd = {"type": "move", "direction": "south"}
        elif action == 3:  # move East
            cmd = {"type": "move", "direction": "east"}
        elif action == 4:  # move West
            cmd = {"type": "move", "direction": "west"}

        elif action == 5:  # build - place burner miner on nearest resource
            cmd = {"type": "build", "entity": "burner-mining-drill"}
        elif action == 6:  # mine - mine nearest entity
            cmd = {"type": "mine", "target": {"name": "any", "position": {"x": 0, "y": 0}}}
        elif action == 7:  # research - pick first available tech
            cmd = {"type": "research", "tech": "automation"}
        elif action == 8:  # craft - craft basic item
            cmd = {"type": "craft", "recipe": "iron-gear-wheel", "count": 5}

        if cmd:
            result = self.api.execute_command(cmd)
            if result and result.get("success"):
                return f"{action_names.get(action, 'unknown')}_success"
            elif result:
                return f"{action_names.get(action, 'unknown')}_failed: {result.get('error')}"
            else:
                return f"{action_names.get(action, 'unknown')}_no_response"

        return "noop"

    def _compute_reward(self):
        """Compute reward from REAL game state via Remote API."""
        reward_info = {
            "survival_reward": 0.0,
            "health_reward": 0.0,
            "production_reward": 0.0,
            "tech_reward": 0.0,
            "entity_reward": 0.0,
            "damage_penalty": 0.0,
        }

        if not self.connected:
            # Fallback: survival only
            reward_info["survival_reward"] = 1.0
            return 1.0, reward_info

        try:
            state = self.api.get_game_state()
            if not state or "error" in state:
                reward_info["survival_reward"] = 1.0
                return 1.0, reward_info

            player = state.get("player", {})
            force = state.get("force", {})
            nearby_entities = state.get("nearby_entities", [])

            # 1. Survival reward (heaviest)
            reward_info["survival_reward"] = 1.0

            # 2. Health change
            health = player.get("health", 100)
            if health < self.last_player_health:
                damage = self.last_player_health - health
                reward_info["damage_penalty"] = -damage * 0.5
                reward_info["health_reward"] = -damage * 0.5
            self.last_player_health = health

            # 3. Production increase
            production = force.get("production", {})
            new_production = 0
            for item, count in production.items():
                prev = self.last_items_produced.get(item, 0)
                if count > prev:
                    new_production += count - prev
            self.last_items_produced = production.copy()
            reward_info["production_reward"] = new_production * 0.1

            # 4. Technology progress
            techs = force.get("technologies", {})
            researched_count = sum(1 for t in techs.values() if t.get("researched"))
            if researched_count > self.last_tech_count:
                reward_info["tech_reward"] = (researched_count - self.last_tech_count) * 10.0
            self.last_tech_count = researched_count

            # 5. Entity count (base building)
            entity_count = len([e for e in state.get("nearby_entities", []) if e.get("force") == "player"])
            if entity_count > self.last_entity_count:
                reward_info["entity_reward"] = (entity_count - self.last_entity_count) * 2.0
            self.last_entity_count = entity_count

        except Exception as e:
            print(f"Reward computation error: {e}")

        # Sum weighted rewards
        total_reward = (
            reward_info["survival_reward"] * 1.0 +
            reward_info["health_reward"] * 1.0 +
            reward_info["production_reward"] * 0.5 +
            reward_info["tech_reward"] * 1.0 +
            reward_info["entity_reward"] * 0.3 +
            reward_info["damage_penalty"] * 1.0
        )

        return total_reward, reward_info

    def close(self):
        """Clean up connection."""
        if self.connected:
            try:
                self.api.execute_command({"type": "stop"})
            except Exception:
                pass
            self.api.close()
            self.connected = False


# Test function
if __name__ == "__main__":
    from main import FactorioScreenCapture, CAPTURE_SIZE, CAPTURE_REGION, CAPTURE_INTERVAL

    print("Testing FactorioRemoteEnv...")
    capture = FactorioScreenCapture(region=CAPTURE_REGION, size=CAPTURE_SIZE, interval=CAPTURE_INTERVAL)
    env = FactorioRemoteEnv(capture=capture, rcon_password="factorio_ai_password")

    if env.connected:
        print("Environment connected!")
        obs, _ = env.reset()
        print(f"Initial obs shape: {obs.shape}")

        # Test a few actions
        for action in [1, 5, 7, 0]:
            obs, reward, done, truncated, info = env.step(action)
            print(f"Action: {info.get('action')}, Reward: {reward:.2f}, Info: {info}")
            import time
            time.sleep(1)

        env.close()
        print("Test complete!")
    else:
        print("Could not connect to Factorio Remote API")