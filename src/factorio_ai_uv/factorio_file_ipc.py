"""
File-based IPC for Factorio AI communication.
Works in single-player without RCON.
Mod writes game state to JSON file, Python reads it.
Python writes commands to JSON file, mod reads and executes.
"""

import json
import time
import os
from pathlib import Path
from typing import Any, Dict, Optional, List
import threading


class FactorioFileIPC:
    """
    File-based communication with Factorio mod.
    
    Mod (Lua) writes game_state.json every tick
    Python reads game_state.json for observations/rewards
    Python writes commands.json for actions
    Mod reads commands.json and executes on next tick
    """

    def __init__(
        self,
        state_file: str = "factorio_game_state.json",
        command_file: str = "factorio_commands.json",
        factorio_script_output: str = None  # Auto-detect
    ):
        # Use Factorio's script-output directory for mod files
        if factorio_script_output is None:
            # Default: %APPDATA%\Factorio\script-output
            import os
            factorio_script_output = os.path.join(
                os.environ.get('APPDATA', ''), 'Factorio', 'script-output'
            )
        
        self.base_dir = Path(factorio_script_output)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        self.state_file = self.base_dir / state_file
        self.command_file = self.base_dir / command_file
        self.response_file = self.base_dir / "factorio_responses.json"
        
        # Ensure files exist
        self.state_file.touch(exist_ok=True)
        self.command_file.touch(exist_ok=True)
        self.response_file.touch(exist_ok=True)
        
        self._lock = threading.Lock()
        self.last_state = {}
        self.command_id = 0

    def read_state(self) -> Optional[Dict]:
        """Read latest game state from file."""
        try:
            with self._lock:
                if not self.state_file.exists():
                    return None
                
                content = self.state_file.read_text(encoding='utf-8')
                if not content.strip():
                    return None
                
                state = json.loads(content)
                self.last_state = state
                return state
        except (json.JSONDecodeError, IOError) as e:
            return None

    def write_command(self, command: Dict) -> int:
        """Write command to file for mod to execute."""
        self.command_id += 1
        cmd = {
            "id": self.command_id,
            "timestamp": time.time(),
            "command": command
        }
        
        try:
            with self._lock:
                # Read existing commands
                commands = []
                if self.command_file.exists():
                    content = self.command_file.read_text(encoding='utf-8')
                    if content.strip():
                        commands = json.loads(content)
                
                commands.append(cmd)
                
                # Keep only last 10 commands
                if len(commands) > 10:
                    commands = commands[-10:]
                
                self.command_file.write_text(json.dumps(commands))
                return self.command_id
        except Exception as e:
            print(f"Write command error: {e}")
            return 0

    def read_response(self, command_id: int) -> Optional[Dict]:
        """Read response for a specific command."""
        try:
            if not self.response_file.exists():
                return None
            
            content = self.response_file.read_text(encoding='utf-8')
            if not content.strip():
                return None
            
            responses = json.loads(content)
            for resp in responses:
                if resp.get("id") == command_id:
                    return resp
            return None
        except Exception as e:
            print(f"Read response error: {e}")
            return None

    def wait_for_response(self, command_id: int, timeout: float = 5.0) -> Optional[Dict]:
        """Wait for command response."""
        start = time.time()
        while time.time() - start < timeout:
            resp = self.read_response(command_id)
            if resp:
                return resp
            time.sleep(0.05)
        return None

    def execute_command(self, command: Dict, wait: bool = True) -> Optional[Dict]:
        """Execute command and optionally wait for response."""
        cmd_id = self.write_command(command)
        if cmd_id == 0:
            return None
        
        if wait:
            return self.wait_for_response(cmd_id)
        return {"id": cmd_id, "status": "sent"}

    def get_game_state(self) -> Optional[Dict]:
        """Get current game state."""
        return self.read_state()


# Lua mod code to add to control.lua for file IPC
LUA_FILE_IPC_CODE = r'''
-- Add this to your control.lua for file-based IPC

local ipc_state_file = "factorio_game_state.json"
local ipc_command_file = "factorio_commands.json"
local ipc_response_file = "factorio_responses.json"

-- Write game state every tick (or every N ticks)
local STATE_WRITE_INTERVAL = 1  -- every tick

script.on_event(defines.events.on_tick, function(event)
    if event.tick % STATE_WRITE_INTERVAL == 0 then
        write_game_state()
    end
    
    -- Check for commands
    read_and_execute_commands()
end)

function write_game_state()
    local player = game.players[1]
    if not player or not player.valid then return end
    
    local state = {
        tick = game.tick,
        player = {
            position = {x = player.position.x, y = player.position.y},
            health = player.health,
            inventory = player.get_main_inventory().get_contents(),
        },
        force = {
            technologies = {},
            production = player.force.items_produced,
        }
    }
    
    -- Technologies
    for name, tech in pairs(player.force.technologies) do
        state.force.technologies[name] = {
            researched = tech.researched,
            enabled = tech.enabled,
        }
    end
    
    -- Nearby entities
    local nearby = {}
    local entities = player.surface.find_entities_filtered{
        area = {{player.position.x - 50, player.position.y - 50}, {player.position.x + 50, player.position.y + 50}},
        force = "player"
    }
    for _, ent in ipairs(entities) do
        if ent.valid then
            table.insert(nearby, {
                name = ent.name,
                position = {x = ent.position.x, y = ent.position.y},
                health = ent.health,
            })
        end
    end
    state.nearby_entities = nearby
    
    -- Write to file
    local json = helpers.table_to_json(state)
    game.write_file(ipc_state_file, json, false, player.force.index)
end

function read_and_execute_commands()
    local content = game.read_file("factorio_commands.json", player.force.index)
    if not content or content == "" then return end
    
    local commands = helpers.json_to_table(content)
    if not commands then return end
    
    local responses = {}
    
    for _, cmd_entry in ipairs(commands) do
        local cmd = cmd_entry.command
        local cmd_id = cmd_entry.id
        local response = {id = cmd_id, success = false, error = "unknown_command"}
        
        if cmd.type == "move" then
            local dir_map = {
                north = defines.direction.north,
                south = defines.direction.south,
                east = defines.direction.east,
                west = defines.direction.west
            }
            if dir_map[cmd.direction] then
                game.players[1].walking_state = {walking = true, direction = dir_map[cmd.direction]}
                response = {id = cmd_id, success = true}
            end
        elseif cmd.type == "stop" then
            game.players[1].walking_state = {walking = false}
            response = {id = cmd_id, success = true}
        elseif cmd.type == "build" then
            local entity = game.players[1].surface.create_entity{
                name = cmd.entity,
                position = cmd.position or game.players[1].position,
                force = game.players[1].force
            }
            if entity then
                response = {id = cmd_id, success = true, entity = entity.name}
            else
                response = {id = cmd_id, success = false, error = "build_failed"}
            end
        elseif cmd.type == "research" then
            if game.players[1].force.technologies[cmd.tech] then
                game.players[1].force.add_research(cmd.tech)
                response = {id = cmd_id, success = true}
            else
                response = {id = cmd_id, success = false, error = "tech_not_found"}
            end
        end
        
        table.insert(responses, response)
    end
    
    -- Write responses
    if #responses > 0 then
        local json = helpers.table_to_json(responses)
        game.write_file("factorio_responses.json", json, false, game.players[1].force.index)
    end
    
    -- Clear commands file
    game.write_file("factorio_commands.json", "[]", false, game.players[1].force.index)
end
'''