"""
Factorio Remote API Client
Connects to Factorio via RCON and calls the factorio_ai_interface remote interface.
"""

import json
import socket
import struct
import time
from typing import Any, Dict, Optional, List


class FactorioRCONClient:
    """RCON client for Factorio (Source RCON protocol)."""

    def __init__(self, host: str = '127.0.0.1', port: int = 27015, password: str = ''):
        self.host = host
        self.port = port
        self.password = password
        self.sock: Optional[socket.socket] = None
        self.request_id = 0

    def connect(self, timeout: float = 5.0) -> bool:
        """Connect to Factorio RCON server."""
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.settimeout(timeout)
            self.sock.connect((self.host, self.port))

            # Authenticate
            if not self._send_packet(3, self.password):
                return False

            # Check auth response
            response = self._receive_packet()
            if response is None or response[0] == -1:
                print("RCON authentication failed")
                return False

            print(f"Connected to Factorio RCON at {self.host}:{self.port}")
            return True

        except Exception as e:
            print(f"RCON connection error: {e}")
            return False

    def _send_packet(self, packet_type: int, body: str) -> bool:
        """Send RCON packet."""
        if not self.sock:
            return False

        self.request_id += 1
        body_bytes = body.encode('utf-8')
        packet = struct.pack('<ii', len(body_bytes) + 10, self.request_id) + \
                 struct.pack('<i', packet_type) + \
                 body_bytes + b'\x00\x00'

        try:
            self.sock.sendall(packet)
            return True
        except Exception as e:
            print(f"RCON send error: {e}")
            return False

    def _receive_packet(self) -> Optional[tuple]:
        """Receive RCON packet. Returns (request_id, type, body) or None."""
        if not self.sock:
            return None

        try:
            # Read header (size + request_id + type)
            header = self._recv_exact(12)
            if not header:
                return None

            size, request_id, packet_type = struct.unpack('<iii', header)
            body = self._recv_exact(size - 10)
            if body is None:
                return None

            # Remove null terminators
            body = body.rstrip(b'\x00')
            return (request_id, packet_type, body.decode('utf-8', errors='replace'))

        except Exception as e:
            print(f"RCON receive error: {e}")
            return None

    def _recv_exact(self, n: int) -> Optional[bytes]:
        """Receive exactly n bytes."""
        data = b''
        while len(data) < n:
            chunk = self.sock.recv(n - len(data))
            if not chunk:
                return None
            data += chunk
        return data

    def execute_command(self, command: str) -> Optional[str]:
        """Execute RCON command (like /c ...). Returns response body."""
        if not self._send_packet(2, command):
            return None

        response = self._receive_packet()
        if response:
            return response[2]
        return None

    def close(self):
        """Close connection."""
        if self.sock:
            self.sock.close()
            self.sock = None


class FactorioRemoteAPI:
    """High-level client for factorio_ai_interface remote interface."""

    def __init__(self, host: str = '127.0.0.1', port: int = 27015, password: str = ''):
        self.rcon = FactorioRCONClient(host, port, password)
        self.connected = False

    def connect(self) -> bool:
        """Connect and verify interface exists."""
        self.connected = self.rcon.connect()
        if self.connected:
            # Test if our interface is available
            result = self.call("factorio_ai", "get_game_state")
            if result:
                print("factorio_ai_interface detected and accessible")
                return True
            else:
                print("WARNING: factorio_ai_interface not found. Is the mod enabled?")
        return False

    def call(self, interface: str, method: str, *args) -> Optional[Any]:
        """Call a remote interface method."""
        if not self.connected:
            return None

        # Build Lua code to call remote interface
        lua_args = json.dumps(list(args)) if args else "{}"
        lua_code = f"""
            local result = remote.call("{interface}", "{method}", table.unpack({lua_args}))
            rcon.print(serpent.block(result))
        """

        response = self.rcon.execute_command(f"/c {lua_code}")
        if response is None:
            return None

        # Parse serpent output (Lua table serialization)
        return self._parse_serpent(response)

    def _parse_serpent(self, text: str) -> Any:
        """Parse serpent.lua output to Python object."""
        # Serpent output looks like: {key = "value", nested = {1, 2, 3}}
        # We'll do a simple conversion
        try:
            # Replace Lua syntax with JSON-compatible
            text = text.strip()

            # Handle serpent's output format
            # This is a simplified parser - for production, use a proper serpent parser
            text = text.replace('{', '[').replace('}', ']')
            text = text.replace('=', ':')
            # Fix keys without quotes
            import re
            text = re.sub(r'(\w+)\s*:', r'"\1":', text)
            text = text.replace('true', 'True').replace('false', 'False')
            text = text.replace('nil', 'null')

            return eval(text)
        except Exception as e:
            print(f"Parse error: {e}")
            print(f"Raw response: {text[:500]}")
            return None

    # Convenience methods
    def get_game_state(self) -> Optional[Dict]:
        """Get full game state."""
        return self.call("factorio_ai", "get_game_state")

    def execute_command(self, cmd: Dict) -> Optional[Dict]:
        """Execute AI command."""
        return self.call("factorio_ai", "execute_command", cmd)

    def get_entity_info(self, params: Dict) -> Optional[Dict]:
        """Query entities."""
        return self.call("factorio_ai", "get_entity_info", params)

    def close(self):
        """Close connection."""
        self.rcon.close()
        self.connected = False


# Example usage
if __name__ == "__main__":
    api = FactorioRemoteAPI(password="your_rcon_password")
    if api.connect():
        state = api.get_game_state()
        print(json.dumps(state, indent=2))
    api.close()