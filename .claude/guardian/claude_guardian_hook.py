#!/usr/bin/env python3
"""
Claude Code Guardian Hook - Real-Time Code Duplication Prevention
Integrates with Claude Code's hook system to prevent duplicate code in real-time.

Installation:
1. Copy this file to your project: .claude/hooks/claude_guardian_hook.py
2. Make executable: chmod +x .claude/hooks/claude_guardian_hook.py
3. Configure hooks in Claude Code using /hooks command:

PreToolUse Hook Configuration:
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Write|Edit|MultiEdit",
        "hooks": [
          {
            "type": "command",
            "command": "$CLAUDE_PROJECT_DIR/.claude/hooks/claude_guardian_hook.py"
          }
        ]
      }
    ]
  }
}

The hook will automatically:
- Check similarity before Claude writes code
- In Advisory mode: Inject similar functions as context
- In Blocking mode: Stop and require user decision
"""

import json
import sys
import asyncio
import os
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add scripts directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))

try:
    from realtime_guardian import RealtimeGuardian, GuardianConfig, GuardianMode, ReviewAction
except ImportError:
    print("❌ Error: Could not import Guardian system", file=sys.stderr)
    print("   Make sure realtime_guardian.py is in the scripts/ directory", file=sys.stderr)
    sys.exit(1)


class ClaudeCodeGuardianHook:
    """Main hook class for Claude Code integration."""
    
    def __init__(self):
        self.guardian = None
        self.config = None
        
    def load_config(self) -> GuardianConfig:
        """Load guardian configuration."""
        # Try project-specific config first
        config_paths = [
            Path(os.environ.get('CLAUDE_PROJECT_DIR', Path.cwd())) / 'guardian-config.json',
            Path.home() / '.claude' / 'guardian-config.json',
            Path.cwd() / 'guardian-config.json'
        ]
        
        for config_path in config_paths:
            if config_path.exists():
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
        
        # Default configuration - advisory mode
        return GuardianConfig(mode=GuardianMode.ADVISORY)
    
    async def initialize_guardian(self):
        """Initialize the guardian system."""
        if self.guardian is None:
            self.config = self.load_config()
            self.guardian = RealtimeGuardian(self.config)
            
            try:
                project_root = Path(os.environ.get('CLAUDE_PROJECT_DIR', Path.cwd()))
                await self.guardian.initialize(project_root)
            except Exception as e:
                print(f"Warning: Could not initialize Guardian: {e}", file=sys.stderr)
                self.guardian = None
                return False
        return True
    
    def extract_code_content(self, tool_input: Dict) -> Optional[str]:
        """Extract code content from tool input."""
        if 'content' in tool_input:
            # Write tool
            return tool_input['content']
        elif 'new_string' in tool_input:
            # Edit tool
            return tool_input['new_string']
        elif 'edits' in tool_input:
            # MultiEdit tool - combine all new_strings
            edits = tool_input['edits']
            if isinstance(edits, list):
                code_parts = []
                for edit in edits:
                    if isinstance(edit, dict) and 'new_string' in edit:
                        code_parts.append(edit['new_string'])
                return '\n'.join(code_parts)
        
        return None
    
    def should_check_similarity(self, code: str, file_path: str = None) -> bool:
        """Determine if we should check similarity for this code."""
        if not code or len(code.strip()) < 50:  # Skip trivial code
            return False
        
        # Skip test files if configured
        if file_path and self.config:
            ignore_patterns = getattr(self.config, 'ignore_patterns', ['test_*', '*_test.py', 'tests/*'])
            for pattern in ignore_patterns:
                if pattern in file_path:
                    return False
        
        # Look for function/class definitions
        code_indicators = ['def ', 'class ', 'function ', 'const ', 'let ', 'async ']
        return any(indicator in code.lower() for indicator in code_indicators)
    
    async def handle_advisory_mode(self, matches: List) -> Dict[str, Any]:
        """Handle advisory mode - inject context without blocking."""
        if not matches:
            return {"continue": True}
        
        context = self.guardian.create_advisory_context(matches)
        
        return {
            "continue": True,
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "additionalContext": context
            },
            "systemMessage": f"💡 Found {len(matches)} similar function(s) - context injected for Claude's consideration"
        }
    
    def handle_blocking_mode_output(self, matches: List) -> Dict[str, Any]:
        """Handle blocking mode - create interactive prompt for user."""
        if not matches:
            return {"continue": True}
        
        # Create blocking message for user
        message = "🚫 GUARDIAN MODE: Similar code detected!\n\n"
        message += f"Found {len(matches)} similar function(s) with >{self.config.similarity_threshold*100:.0f}% similarity:\n\n"
        
        for i, match in enumerate(matches, 1):
            message += f"{i}. {match.function_name} ({match.file_path}:{match.line_number})\n"
            message += f"   Similarity: {match.similarity_score:.1%}\n"
            message += f"   Signature: {match.signature}\n"
            
            if match.documentation:
                message += f"   Doc: {match.documentation}\n"
            
            # Show code preview
            message += "   Code preview:\n"
            for line in match.code_snippet.split('\n')[:4]:
                message += f"     {line}\n"
            if len(match.code_snippet.split('\n')) > 4:
                message += "     ...\n"
            message += "\n"
        
        message += "Choose an action:\n"
        message += "1. Use existing function\n"
        message += "2. Modify existing to be more generic\n"
        message += "3. Proceed with new implementation\n"
        message += "4. Refactor both into shared utility\n"
        message += "5. Cancel operation\n"
        
        return {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "ask",
                "permissionDecisionReason": message
            },
            "systemMessage": f"🚫 Guardian blocked similar code - {len(matches)} match(es) found"
        }
    
    async def process_hook_input(self, input_data: Dict) -> Dict[str, Any]:
        """Main hook processing logic."""
        tool_name = input_data.get('tool_name', '')
        tool_input = input_data.get('tool_input', {})
        
        # Only process code-writing tools
        if tool_name not in ['Write', 'Edit', 'MultiEdit']:
            return {"continue": True}
        
        # Extract code content
        code_content = self.extract_code_content(tool_input)
        if not code_content:
            return {"continue": True}
        
        file_path = tool_input.get('file_path', '')
        
        # Check if we should analyze this code
        if not self.should_check_similarity(code_content, file_path):
            return {"continue": True}
        
        # Initialize guardian
        if not await self.initialize_guardian():
            return {"continue": True}  # Fail gracefully
        
        if self.config.mode == GuardianMode.DISABLED:
            return {"continue": True}
        
        try:
            # Check for similar code
            matches = await self.guardian.check_code_similarity(code_content, file_path)
            
            if self.config.mode == GuardianMode.ADVISORY:
                return await self.handle_advisory_mode(matches)
            elif self.config.mode == GuardianMode.BLOCKING:
                return self.handle_blocking_mode_output(matches)
            
        except Exception as e:
            print(f"Warning: Guardian similarity check failed: {e}", file=sys.stderr)
            return {"continue": True}  # Fail gracefully
        
        return {"continue": True}


async def main():
    """Main hook entry point."""
    try:
        # Read JSON input from stdin
        input_data = json.load(sys.stdin)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON input: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Verify this is a PreToolUse hook
    if input_data.get('hook_event_name') != 'PreToolUse':
        sys.exit(0)  # Not our event, allow to continue
    
    # Process the hook
    hook = ClaudeCodeGuardianHook()
    result = await hook.process_hook_input(input_data)
    
    # Output JSON result
    print(json.dumps(result))


if __name__ == '__main__':
    asyncio.run(main())