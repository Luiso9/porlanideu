"""
Factorio Environment using Kruise_Kontrol_Remote for actions and RCON for game state.
Screen capture used for visual observation (CNN input).
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import factorio_rcon
import time
import re
import json

from .config import FactorioScreenCapture, CAPTURE_SIZE, CAPTURE_REGION, CAPTURE_INTERVAL, MAX_EPISODE_STEPS


class FactorioKKEnv(gym.Env):
    """
    Gymnasium environment using:
    - Kruise_Kontrol_Remote for reliable character actions (start_job, cancel, is_active)
    - RCON for game state queries (player, inventory, tech, production)
    - Screen capture for visual observation (CNN input)
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        capture: FactorioScreenCapture = None,
        rcon_host: str = '127.0.0.1',
        rcon_port: int = 27015,
        rcon_password: str = 'factorio_ai_password'
    ):
        super().__init__()

        # Screen capture for visual observation
        if capture is None:
            capture = FactorioScreenCapture(
                region=CAPTURE_REGION,
                size=CAPTURE_SIZE,
                interval=CAPTURE_INTERVAL
            )
        self.capture = capture

        # RCON client
        self.rcon_host = rcon_host
        self.rcon_port = rcon_port
        self.rcon_password = rcon_password
        self.client: factorio_rcon.RCONClient = None
        self.connected = False

        # Action space: 0=noop, 1=move_N, 2=move_S, 3=move_E, 4=move_W,
        # 5=build_at_cursor, 6=mine_at_cursor, 7=research, 8=craft
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
        self.pending_job = False

        # Connect to Factorio
        self._connect()

    def _connect(self):
        """Connect to Factorio RCON."""
        print("Connecting to Factorio RCON...")
        try:
            self.client = factorio_rcon.RCONClient(
                self.rcon_host, self.rcon_port, self.rcon_password
            )
            # Test connection
            resp = self.client.send_command('/c rcon.print("test")')
            if resp is not None:
                # Test Kruise_Kontrol_Remote
                resp = self.client.send_command(
                    '/c local ok = pcall(function() return remote.interfaces["kruise_kontrol_updated"] end); rcon.print(serpent.block({ok=ok}))'
                )
                if resp and 'ok = true' in resp:
                    self.connected = True
                    print("Factorio RCON + Kruise_Kontrol_Remote connected!")
                    return
            print("WARNING: Kruise_Kontrol_Remote not available")
        except Exception as e:
            print(f"RCON connection error: {e}")
        self.connected = False

    def _to_lua(self, obj):
        """Convert Python object to Lua literal string."""
        if isinstance(obj, dict):
            parts = []
            for k, v in obj.items():
                key = f'["{k}"]' if not k.isidentifier() else k
                parts.append(f"{key} = {self._to_lua(v)}")
            return "{" + ", ".join(parts) + "}"
        elif isinstance(obj, list):
            return "{" + ", ".join(self._to_lua(v) for v in obj) + "}"
        elif isinstance(obj, str):
            # Escape single quotes and backslashes
            escaped = obj.replace("\\", "\\\\").replace("'", "\\'")
            return f"'{escaped}'"
        elif isinstance(obj, bool):
            return "true" if obj else "false"
        elif obj is None:
            return "nil"
        elif isinstance(obj, (int, float)):
            return str(obj)
        else:
            return str(obj)

    def _send_lua(self, code: str) -> str:
        """Execute Lua code via RCON and return response (silent)."""
        if not self.connected:
            return ""
        try:
            # Use load() to execute silently - avoids console spam
            # [==[...]==] is a long string literal that handles quotes properly
            silent_wrapper = f"local f,err=load([==[{code}]==]); if f then return f() else error(err) end"
            cmd = f'/c {silent_wrapper}'
            return self.client.send_command(cmd) or ""
        except Exception as e:
            print(f"RCON error: {e}")
            return ""

    def _call_kk(self, method: str, *args) -> str:
        """Call Kruise_Kontrol_Remote method."""
        args_lua = ", ".join(self._to_lua(a) for a in args)
        code = f'local ok, result = pcall(function() return remote.call("kruise_kontrol_updated", "{method}", {args_lua}) end); rcon.print(serpent.block({{ok=ok, result=result}}))'
        return self._send_lua(code)

    def _get_game_state(self):
        """Get game state via direct Lua query."""
        if not self.connected:
            return None
        code = '''
        local player = game.players[1]
        if not player or not player.valid then return nil end
        local char = player.character
        local inv = player.get_main_inventory()
        local surf = player.surface
        local force = player.force
        
        local state = {
            tick = game.tick,
            player = {
                position = {x = player.position.x, y = player.position.y},
                health = char and char.health or 0,
                inventory = inv and inv.get_contents() or {},
            },
            force = {
                technologies = {}
            }
        }
        for name, tech in pairs(force.technologies) do
            state.force.technologies[name] = {researched = tech.researched, enabled = tech.enabled}
        end
        rcon.print(helpers.table_to_json(state))
        '''
        return self._send_lua(code)

    def _parse_state(self, text: str):
        """Parse JSON output from Lua."""
        try:
            import json, re
            # Find the first complete JSON object in the text
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                return json.loads(match.group(0))
        except Exception as e:
            print(f"Parse error: {e}")
        return None

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
        self.pending_job = False

        # Cancel any pending job
        if self.connected:
            self._call_kk("cancel", 1)

        obs = self.capture.capture()
        return obs, {}

    def step(self, action):
        self.total_ticks += 1

        # VecEnv passes action as list/array; extract scalar
        if isinstance(action, (list, tuple, np.ndarray)):
            action = int(action[0]) if len(action) > 0 else 0

        self.current_action = action

        # Execute action via Kruise_Kontrol_Remote
        action_name = self._execute_action(action)

        # Capture new frame (visual observation)
        obs = self.capture.capture()

        # Compute reward from game state
        reward, reward_info = self._compute_reward()

        # Check done conditions
        done = self.total_ticks >= MAX_EPISODE_STEPS

        info = {
            "action": action_name,
            "tick": self.total_ticks,
            "reward": reward,
            **reward_info,
        }

        return obs, reward, done, False, info

    def _execute_action(self, action: int) -> str:
        """Execute action via Kruise_Kontrol_Remote."""
        if not self.connected:
            return "noop (not connected)"

        action_names = {
            0: "noop",
            1: "move_N",
            2: "move_S",
            3: "move_E",
            4: "move_W",
            5: "build_at_cursor",
            6: "mine_at_cursor",
            7: "research",
            8: "craft",
        }

        if action == 0:
            self._call_kk("cancel", 1)
            return "cancel"

        # Get player position for relative movement (silent)
        pos_resp = self._send_lua('rcon.print(helpers.table_to_json(game.players[1].position))')
        match = re.search(r'"x":([-\d.]+),"y":([-\d.]+)', pos_resp)
        if match:
            px, py = float(match.group(1)), float(match.group(2))
        else:
            px, py = 0, 0

        # Movement actions - walk to relative position
        if action == 1:  # North
            self._call_kk("start_job", 1, {"x": px, "y": py - 10}, "nil")
            return "move_N"
        elif action == 2:  # South
            self._call_kk("start_job", 1, {"x": px, "y": py + 10}, "nil")
            return "move_S"
        elif action == 3:  # East
            self._call_kk("start_job", 1, {"x": px + 10, "y": py}, "nil")
            return "move_E"
        elif action == 4:  # West
            self._call_kk("start_job", 1, {"x": px - 10, "y": py}, "nil")
            return "move_W"

        # Build at cursor position
        elif action == 5:
            self._call_kk("start_job", 1, {"x": px, "y": py}, "player.selected")
            return "build_at_cursor"

        # Mine at cursor
        elif action == 6:
            self._call_kk("start_job", 1, {"x": px, "y": py}, "player.selected")
            return "mine_at_cursor"

        # Research (via RCON direct, silent)
        elif action == 7:
            self._send_lua("game.players[1].force.add_research('automation')")
            return "research"

        # Craft (via RCON direct, silent)
        elif action == 8:
            self._send_lua("game.players[1].begin_crafting{recipe='iron-gear-wheel', count=5, silent=true}")
            return "craft"

        return "unknown"

    def _compute_reward(self):
        """Compute reward from game state."""
        reward_info = {
            "survival_reward": 0.0,
            "health_reward": 0.0,
            "tech_reward": 0.0,
            "damage_penalty": 0.0,
        }

        if not self.connected:
            reward_info["survival_reward"] = 1.0
            return 1.0, reward_info

        try:
            state_text = self._get_game_state()
            state = self._parse_state(state_text)
            if not state:
                reward_info["survival_reward"] = 1.0
                return 1.0, reward_info

            player = state.get("player", {})
            force = state.get("force", {})

            # 1. Survival reward (heaviest)
            reward_info["survival_reward"] = 1.0

            # 2. Health change
            health = player.get("health", 0)
            if health < self.last_player_health:
                damage = self.last_player_health - health
                reward_info["damage_penalty"] = -damage * 0.5
                reward_info["health_reward"] = -damage * 0.5
            self.last_player_health = health

            # 4. Technology progress
            techs = force.get("technologies", {})
            researched_count = sum(1 for t in techs.values() if t.get("researched"))
            if researched_count > self.last_tech_count:
                reward_info["tech_reward"] = (researched_count - self.last_tech_count) * 10.0
            self.last_tech_count = researched_count

        except Exception as e:
            print(f"Reward computation error: {e}")

        # Sum weighted rewards
        total_reward = (
            reward_info["survival_reward"] * 1.0 +
            reward_info["health_reward"] * 1.0 +
            reward_info["tech_reward"] * 1.0 +
            reward_info["damage_penalty"] * 1.0
        )

        return total_reward, reward_info

    def _parse_state(self, text: str):
        """Parse JSON output from Lua."""
        try:
            import json, re
            # Find the first complete JSON object in the text
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                return json.loads(match.group(0))
        except Exception as e:
            print(f"Parse error: {e}")
        return None

    def close(self):
        """Clean up connection."""
        if self.connected:
            try:
                self._call_kk("cancel", 1)
            except Exception:
                pass
            try:
                self.client.close()
            except Exception:
                pass
            self.connected = False


# Test function
if __name__ == "__main__":
    from main import FactorioScreenCapture, CAPTURE_SIZE, CAPTURE_REGION, CAPTURE_INTERVAL

    print("Testing FactorioKKEnv...")
    capture = FactorioScreenCapture(region=CAPTURE_REGION, size=CAPTURE_SIZE, interval=CAPTURE_INTERVAL)
    env = FactorioKKEnv(capture=capture)

    if env.connected:
        print("Environment connected!")
        obs, _ = env.reset()
        print(f"Initial obs shape: {obs.shape}")

        # Test a few actions
        for action in [1, 5, 7, 8, 0]:
            obs, reward, done, truncated, info = env.step(action)
            print(f"Action: {info.get('action')}, Reward: {reward:.2f}")
            import time
            time.sleep(1)

        env.close()
        print("Test complete!")
    else:
        print("Could not connect to Factorio")