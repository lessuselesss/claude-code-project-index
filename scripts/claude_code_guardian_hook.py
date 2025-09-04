#!/usr/bin/env python3
"""
Claude Code Guardian Hook
Integrates Real-Time Guardian with Claude Code's hook system.

This hook intercepts code being written by Claude and performs real-time
similarity checking to prevent code duplication.
"""

import json
import sys
import os
import asyncio
from pathlib import Path
from typing import Dict, Any

# Import the guardian system
try:
    from realtime_guardian import RealtimeGuardian, GuardianConfig, GuardianMode, ReviewAction
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent))
    from realtime_guardian import RealtimeGuardian, GuardianConfig, GuardianMode, ReviewAction


class ClaudeCodeGuardianHook:
    """Hook that integrates with Claude Code to provide real-time duplicate prevention."""
    
    def __init__(self):
        self.guardian = None
        self.config = self._load_config()
        
    def _load_config(self) -> GuardianConfig:
        """Load guardian configuration from project or global settings."""
        # Try project-specific config first
        config_paths = [
            Path.cwd() / 'guardian-config.json',
            Path.home() / '.claude-code-project-index' / 'guardian-config.json',
            Path.cwd() / 'project-index.toml'  # Future unified config
        ]
        
        for config_path in config_paths:
            if config_path.exists() and config_path.suffix == '.json':
                try:
                    with open(config_path) as f:
                        data = json.load(f)
                        return GuardianConfig(
                            mode=GuardianMode(data.get('mode', 'advisory')),
                            similarity_threshold=data.get('similarity_threshold', 0.85),
                            max_matches=data.get('max_matches', 5),
                            embedding_model=data.get('embedding_model', 'nomic-embed-text'),
                            embedding_endpoint=data.get('embedding_endpoint', 'http://localhost:11434')
                        )
                except Exception as e:
                    print(f"Warning: Could not load config from {config_path}: {e}", file=sys.stderr)
        
        # Default configuration
        return GuardianConfig()
    
    async def _initialize_guardian(self):
        """Initialize guardian if not already done."""
        if not self.guardian:
            self.guardian = RealtimeGuardian(self.config)
            try:
                await self.guardian.initialize()
            except Exception as e:
                print(f"Warning: Could not initialize Guardian: {e}", file=sys.stderr)
                self.guardian = None
    
    async def on_code_write_attempt(self, code: str, file_path: str = None, context: Dict = None) -> Dict[str, Any]:
        """
        Hook called when Claude attempts to write code.
        
        Returns:
            Dict with keys:
            - allow: bool - Whether to allow the write
            - modified_code: str - Potentially modified code
            - context_injection: str - Additional context for Claude
            - user_message: str - Message to show to user
        """
        if self.config.mode == GuardianMode.DISABLED:
            return {"allow": True, "modified_code": code}
        
        await self._initialize_guardian()
        if not self.guardian:
            return {"allow": True, "modified_code": code}
        
        try:
            # Check for similar code
            matches = await self.guardian.check_code_similarity(code, file_path)
            
            if not matches:
                # No similar code found - proceed normally
                return {"allow": True, "modified_code": code}
            
            if self.config.mode == GuardianMode.BLOCKING:
                return await self._handle_blocking_mode(code, matches)
            else:  # Advisory mode
                return self._handle_advisory_mode(code, matches)
                
        except Exception as e:
            print(f"Warning: Guardian check failed: {e}", file=sys.stderr)
            return {"allow": True, "modified_code": code}
    
    async def _handle_blocking_mode(self, code: str, matches) -> Dict[str, Any]:
        """Handle blocking mode - stop and require user decision."""
        action = await self.guardian.handle_blocking_mode(matches)
        
        if action == ReviewAction.USE_EXISTING:
            # User chose to use existing function
            best_match = matches[0]
            return {
                "allow": False,
                "user_message": f"✅ Using existing function: {best_match.function_name} from {best_match.file_path}",
                "suggested_action": "use_existing",
                "existing_function": {
                    "name": best_match.function_name,
                    "file_path": best_match.file_path,
                    "line_number": best_match.line_number
                }
            }
        
        elif action == ReviewAction.MODIFY_EXISTING:
            return {
                "allow": False, 
                "user_message": f"🔧 Modify existing function: {matches[0].function_name}",
                "suggested_action": "modify_existing",
                "existing_function": {
                    "name": matches[0].function_name,
                    "file_path": matches[0].file_path,
                    "line_number": matches[0].line_number,
                    "code": matches[0].code_snippet
                }
            }
        
        elif action == ReviewAction.PROCEED_NEW:
            return {
                "allow": True,
                "modified_code": code,
                "user_message": "✅ Proceeding with new implementation as requested"
            }
        
        elif action == ReviewAction.REFACTOR_BOTH:
            return {
                "allow": False,
                "user_message": "🔄 Refactoring needed - create shared utility function",
                "suggested_action": "refactor_both",
                "existing_functions": [
                    {
                        "name": match.function_name,
                        "file_path": match.file_path,
                        "code": match.code_snippet
                    } for match in matches[:2]  # Top 2 matches
                ]
            }
        
        else:  # CANCEL
            return {
                "allow": False,
                "user_message": "❌ Operation cancelled by user",
                "suggested_action": "cancel"
            }
    
    def _handle_advisory_mode(self, code: str, matches) -> Dict[str, Any]:
        """Handle advisory mode - inject context and let Claude decide."""
        context_injection = self.guardian.create_advisory_context(matches)
        
        return {
            "allow": True,
            "modified_code": code,
            "context_injection": context_injection,
            "user_message": f"💡 Found {len(matches)} similar function(s) - context injected for Claude's consideration"
        }
    
    async def on_code_written(self, code: str, file_path: str, success: bool) -> None:
        """Hook called after Claude has successfully written code."""
        if success and self.guardian:
            # Could track successful writes, learn from patterns, etc.
            # For now, just update any caches if needed
            pass


# Hook entry points for Claude Code integration
async def user_prompt_submit_hook(prompt: str, context: Dict = None) -> Dict[str, Any]:
    """
    Claude Code hook: Called when user submits a prompt.
    This allows us to set up the guardian before Claude starts writing.
    """
    hook = ClaudeCodeGuardianHook()
    
    # Check if this prompt might result in code being written
    code_indicators = ['function', 'def ', 'class ', 'const ', 'let ', 'var ', 'async ', 'export']
    might_write_code = any(indicator in prompt.lower() for indicator in code_indicators)
    
    if might_write_code:
        await hook._initialize_guardian()
        return {
            "guardian_active": True,
            "mode": hook.config.mode.value,
            "threshold": hook.config.similarity_threshold
        }
    
    return {"guardian_active": False}


async def tool_call_hook(tool_name: str, args: Dict, result: Any = None) -> Dict[str, Any]:
    """
    Claude Code hook: Called during tool execution.
    We intercept Write/Edit/MultiEdit tools to check code similarity.
    """
    if tool_name not in ['Write', 'Edit', 'MultiEdit']:
        return {}
    
    hook = ClaudeCodeGuardianHook()
    
    # Extract code content from tool args
    code_content = None
    file_path = args.get('file_path')
    
    if tool_name == 'Write':
        code_content = args.get('content', '')
    elif tool_name == 'Edit':
        code_content = args.get('new_string', '')
    elif tool_name == 'MultiEdit':
        edits = args.get('edits', [])
        if edits:
            # Check the new strings being added
            code_content = '\n'.join(edit.get('new_string', '') for edit in edits)
    
    if code_content and len(code_content.strip()) > 50:  # Only check substantial code
        response = await hook.on_code_write_attempt(code_content, file_path)
        
        if not response.get('allow', True):
            # Guardian blocked the write
            return {
                "block_tool": True,
                "reason": response.get('user_message', 'Guardian blocked code write'),
                "suggested_action": response.get('suggested_action'),
                "existing_function": response.get('existing_function'),
                "existing_functions": response.get('existing_functions')
            }
        
        elif response.get('context_injection'):
            # Advisory mode - inject context
            return {
                "inject_context": response['context_injection'],
                "user_message": response.get('user_message')
            }
    
    return {}


def main():
    """Test the hook system."""
    async def test_hook():
        hook = ClaudeCodeGuardianHook()
        
        # Test code that might be similar to existing functions
        test_code = """
def authenticate_user(username, password):
    \"\"\"Authenticate a user with username and password.\"\"\"
    # Hash the password
    hashed = hash_password(password)
    
    # Check against database
    user = db.get_user(username)
    if user and user.password_hash == hashed:
        return user
    return None
"""
        
        result = await hook.on_code_write_attempt(test_code, "test.py")
        print(json.dumps(result, indent=2))
    
    asyncio.run(test_hook())


if __name__ == '__main__':
    main()