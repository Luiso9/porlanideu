"""
Factorio RCON client with Kruise_Kontrol_Remote integration.
"""

import json
import logging
import re
import time
from typing import Any, Optional

import factorio_rcon

from .config import CaptureConfig, REWARD_WEIGHTS
from .exceptions import (
    ConnectionError,
    RCONError,
    ModNotFoundError,
    ParseError,
    ActionError,
    ScreenCaptureError,
)

logger = logging.getLogger(__name__)


class FactorioClient:
    """
    Factorio RCON client with Kruise_Kontrol_Remote integration.
    
    Handles:
    - RCON connection with retries
    - Kruise_Kontrol_Remote actions (movement, building, mining)
    - Game state queries (position, health, inventory, tech)
    - Silent command execution (no console spam)
    """

    # Lua code templates for common operations
    _LUA_GET_POSITION = 'rcon.print(helpers.table_to_json(game.players[1].position))'
    _LUA_GET_STATE = '''
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

    _LUA_ADD_RESEARCH = "game.players[1].force.add_research('automation')"
    _LUA_CRAFT_GEARS = "game.players[1].begin_crafting{recipe='iron-gear-wheel', count=5, silent=true}"

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 27015,
        password: str = "factorio_ai_password",
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ) -> None:
        """
        Initialize Factorio client.
        
        Args:
            host: RCON host address
            port: RCON port
            password: RCON password
            max_retries: Maximum connection retries
            retry_delay: Delay between retries (seconds)
        """
        self.host = host
        self.port = port
        self.password = password
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        self._client: Optional[factorio_rcon.RCONClient] = None
        self._connected = False
        self._kk_available = False

    def connect(self) -> bool:
        """
        Connect to Factorio RCON and verify Kruise_Kontrol_Remote.
        
        Returns:
            True if connection successful
            
        Raises:
            ConnectionError: If connection fails after retries
            ModNotFoundError: If Kruise_Kontrol_Remote not available
        """
        for attempt in range(self.max_retries):
            try:
                self._client = factorio_rcon.RCONClient(
                    self.host, self.port, self.password
                )
                # Test basic connection
                resp = self._send_raw('rcon.print("test")')
                if resp is None:
                    raise RCONError("No response from server")
                
                # Check Kruise_Kontrol_Remote
                kk_resp = self._send_raw(
                    '/c local ok = pcall(function() return remote.interfaces["kruise_kontrol_updated"] end); '
                    'rcon.print(serpent.block({ok=ok}))'
                )
                if kk_resp and 'ok = true' in kk_resp:
                    self._kk_available = True
                    logger.info("Kruise_Kontrol_Remote connected")
                else:
                    raise ModNotFoundError("Kruise_Kontrol_Remote not available")
                
                self._connected = True
                logger.info("Connected to Factorio RCON + Kruise_Kontrol_Remote")
                return True
                
            except (factorio_rcon.RCONConnectError, factorio_rcon.RCONSendError) as e:
                logger.warning(f"Connection attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (2 ** attempt))  # Exponential backoff
                else:
                    raise ConnectionError(f"Failed after {self.max_retries} attempts") from e
        
        return False

    def _send_raw(self, command: str) -> Optional[str]:
        """
        Send raw RCON command.
        
        Args:
            command: RCON command string
            
        Returns:
            Response string or None
        """
        if not self._client:
            return None
        try:
            return self._client.send_command(command) or ""
        except (factorio_rcon.RCONSendError, factorio_rcon.RCONReceiveError) as e:
            logger.error(f"RCON send error: {e}")
            self._connected = False
            return None

    def _send_silent(self, code: str) -> Optional[str]:
        """
        Execute Lua code silently (no console output).
        
        Args:
            code: Lua code to execute
            
        Returns:
            Response string or None
        """
        # Use load() to execute silently - avoids console spam
        silent_wrapper = f"local f,err=load([==[{code}]==]); if f then return f() else error(err) end"
        return self._send_raw(f'/c {silent_wrapper}')

    def _call_kk(self, method: str, *args) -> Optional[str]:
        """
        Call Kruise_Kontrol_Remote method.
        
        Args:
            method: KK method name
            *args: Arguments to pass
            
        Returns:
            Response string or None
        """
        args_lua = ", ".join(self._to_lua(a) for a in args)
        code = (
            f'local ok, result = pcall(function() '
            f'return remote.call("kruise_kontrol_updated", "{method}", {args_lua}) '
            f'end); rcon.print(serpent.block({{ok=ok, result=result}}))'
        )
        return self._send_raw(f'/c {code}')

    def _to_lua(self, obj: Any) -> str:
        """Convert Python object to Lua literal string."""
        if isinstance(obj, dict):
            parts = []
            for k, v in obj.items():
                key = f'["{k}"]' if not str(k).isidentifier() else k
                parts.append(f"{key} = {self._to_lua(v)}")
            return "{" + ", ".join(parts) + "}"
        elif isinstance(obj, list):
            return "{" + ", ".join(self._to_lua(v) for v in obj) + "}"
        elif isinstance(obj, str):
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

    # Public API
    
    def is_connected(self) -> bool:
        """Check if connected."""
        return self._connected and self._kk_available

    def get_game_state(self) -> Optional[dict]:
        """
        Get current game state.
        
        Returns:
            Game state dict or None if failed
        """
        if not self._connected:
            return None
        
        try:
            resp = self._send_silent(self._LUA_GET_STATE)
            if not resp:
                return None
            
            # Parse JSON from response
            match = re.search(r'\{.*\}', resp, re.DOTALL)
            if not match:
                raise ParseError(f"No JSON found in response: {resp[:200]}")
            
            return json.loads(match.group(0))
            
        except json.JSONDecodeError as e:
            raise ParseError(f"Failed to parse game state: {e}") from e

    def get_player_position(self) -> tuple[float, float]:
        """
        Get player position.
        
        Returns:
            (x, y) tuple
        """
        resp = self._send_silent(self._LUA_GET_POSITION)
        if not resp:
            return (0.0, 0.0)
        
        match = re.search(r'"x":([-\d.]+),"y":([-\d.]+)', resp)
        if match:
            return (float(match.group(1)), float(match.group(2)))
        return (0.0, 0.0)

    def start_job(self, player_index: int, position: dict, entity: str = "nil") -> bool:
        """
        Start Kruise_Kontrol job (move/build/mine).
        
        Args:
            player_index: Player index (usually 1)
            position: {"x": float, "y": float}
            entity: "nil" for movement, or "player.selected" for build/mine
            
        Returns:
            True if job started
        """
        resp = self._call_kk("start_job", 1, position, entity)
        if not resp:
            return False
        
        try:
            match = re.search(r'ok\s*=\s*true', resp)
            return match is not None
        except Exception:
            return False

    def cancel(self, player_index: int = 1) -> bool:
        """Cancel current Kruise_Kontrol job."""
        resp = self._call_kk("cancel", 1)
        if not resp:
            return False
        try:
            match = re.search(r'ok\s*=\s*true', resp)
            return match is not None
        except Exception:
            return False

    def is_active(self, player_index: int = 1) -> bool:
        """Check if Kruise_Kontrol is currently active."""
        resp = self._call_kk("is_active", 1)
        if not resp:
            return False
        try:
            match = re.search(r'result\s*=\s*true', resp)
            return match is not None
        except Exception:
            return False

    def get_description(self, player_index: int = 1) -> Optional[str]:
        """Get Kruise_Kontrol description."""
        resp = self._call_kk("get_description", 1)
        if not resp:
            return None
        try:
            match = re.search(r'result\s*=\s*"([^"]*)"', resp)
            return match.group(1) if match else None
        except Exception:
            return None

    def add_research(self, tech_name: str) -> bool:
        """Add research via RCON."""
        resp = self._send_silent(f"game.players[1].force.add_research('{tech_name}')")
        return resp is not None

    def craft_gears(self, count: int = 5) -> bool:
        """Craft iron gear wheels."""
        resp = self._send_silent(
            f"game.players[1].begin_crafting{{recipe='iron-gear-wheel', count={count}, silent=true}}"
        )
        return resp is not None

    def close(self) -> None:
        """Close connection."""
        if self._connected:
            try:
                self.cancel(1)
            except Exception:
                pass
            try:
                if self._client:
                    self._client.close()
            except Exception:
                pass
            self._connected = False
            self._kk_available = False
            logger.info("Disconnected from Factorio")