#!/usr/bin/env python3
"""
Real-Time Code Duplication Prevention System
Integrates with Claude Code to prevent duplicate code and promote reuse.

Two modes:
1. Guardian Mode: Blocks Claude when similarity exceeds threshold, requires manual review
2. Advisory Mode: Feeds similar functions to Claude for intelligent decision-making
"""

__version__ = "0.1.0"

import json
import os
import sys
import asyncio
import hashlib
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, NamedTuple
from dataclasses import dataclass
from enum import Enum

# Import existing utilities
try:
    from find_ollama import OllamaManager
    from index_utils import extract_python_signatures, extract_javascript_signatures, extract_shell_signatures
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent))
    from find_ollama import OllamaManager
    from index_utils import extract_python_signatures, extract_javascript_signatures, extract_shell_signatures


class GuardianMode(Enum):
    BLOCKING = "blocking"    # Guardian mode - blocks and requires review
    ADVISORY = "advisory"    # Assistant mode - injects context, lets Claude decide
    DISABLED = "disabled"    # No real-time checking


class ReviewAction(Enum):
    USE_EXISTING = "use_existing"
    MODIFY_EXISTING = "modify_existing"  
    PROCEED_NEW = "proceed_new"
    REFACTOR_BOTH = "refactor_both"
    CANCEL = "cancel"


@dataclass
class SimilarityMatch:
    """Represents a similar function found in the index."""
    file_path: str
    function_name: str
    similarity_score: float
    signature: str
    documentation: str
    line_number: int
    code_snippet: str


@dataclass
class GuardianConfig:
    """Configuration for the real-time guardian system."""
    mode: GuardianMode = GuardianMode.ADVISORY
    similarity_threshold: float = 0.85
    max_matches: int = 5
    embedding_model: str = "nomic-embed-text"
    embedding_endpoint: str = "http://localhost:11434"
    cache_embeddings: bool = True
    advisory_context_template: str = """
## Similar Functions Found

The following functions in your codebase are similar to what you're about to write:

{matches}

Consider whether you can:
- Reuse an existing function directly
- Extend/modify an existing function to be more generic
- Build upon the existing patterns
- Create new logic if the use case is genuinely different
"""


class RealtimeGuardian:
    """Real-time code duplication prevention system."""
    
    def __init__(self, config: GuardianConfig = None):
        self.config = config or GuardianConfig()
        self.project_index = None
        self.embedding_cache = {}
        self.ollama_manager = None
        
    async def initialize(self, project_root: Path = None):
        """Initialize the guardian with project index and embedding service."""
        if not project_root:
            project_root = self._find_project_root()
        
        # Load project index
        index_path = project_root / 'PROJECT_INDEX.json'
        if index_path.exists():
            with open(index_path) as f:
                self.project_index = json.load(f)
        else:
            raise FileNotFoundError(f"PROJECT_INDEX.json not found at {index_path}")
        
        # Initialize embedding service
        if self.config.mode != GuardianMode.DISABLED:
            self.ollama_manager = OllamaManager(self.config.embedding_endpoint)
            self.ollama_manager.default_model = self.config.embedding_model
    
    def _find_project_root(self) -> Path:
        """Find project root by looking for PROJECT_INDEX.json."""
        current = Path.cwd()
        while current != current.parent:
            if (current / 'PROJECT_INDEX.json').exists():
                return current
            current = current.parent
        raise FileNotFoundError("Could not find PROJECT_INDEX.json in current directory or parents")
    
    async def check_code_similarity(self, code: str, file_path: str = None) -> List[SimilarityMatch]:
        """Check if the provided code is similar to existing functions."""
        if self.config.mode == GuardianMode.DISABLED:
            return []
        
        # Extract function information from the code
        functions = self._extract_functions_from_code(code, file_path)
        if not functions:
            return []
        
        # Generate embeddings for new functions
        all_matches = []
        for func_info in functions:
            matches = await self._find_similar_functions(func_info)
            all_matches.extend(matches)
        
        # Sort by similarity score and return top matches
        all_matches.sort(key=lambda x: x.similarity_score, reverse=True)
        return all_matches[:self.config.max_matches]
    
    def _extract_functions_from_code(self, code: str, file_path: str = None) -> List[Dict]:
        """Extract function information from code snippet."""
        # Determine language from file path or code content
        language = self._detect_language(code, file_path)
        
        # Extract functions based on language
        if language == 'python':
            # Create a temporary content structure for the parser
            temp_signatures = extract_python_signatures(code)
            functions = []
            for name, details in temp_signatures.get('functions', {}).items():
                functions.append({
                    'name': name,
                    'signature': details.get('signature', ''),
                    'doc': details.get('doc', ''),
                    'code': code,  # For now, include full code - could be more precise
                    'language': language
                })
            return functions
        elif language in ['javascript', 'typescript']:
            temp_signatures = extract_javascript_signatures(code)
            functions = []
            for name, details in temp_signatures.get('functions', {}).items():
                functions.append({
                    'name': name,
                    'signature': details.get('signature', ''),
                    'doc': details.get('doc', ''),
                    'code': code,
                    'language': language
                })
            return functions
        elif language == 'shell':
            temp_signatures = extract_shell_signatures(code)
            functions = []
            for name, details in temp_signatures.get('functions', {}).items():
                functions.append({
                    'name': name,
                    'signature': details.get('signature', ''),
                    'doc': details.get('doc', ''),
                    'code': code,
                    'language': language
                })
            return functions
        
        return []
    
    def _detect_language(self, code: str, file_path: str = None) -> str:
        """Detect programming language from file path or code content."""
        if file_path:
            ext = Path(file_path).suffix.lower()
            lang_map = {
                '.py': 'python',
                '.js': 'javascript',
                '.ts': 'typescript',
                '.jsx': 'javascript',
                '.tsx': 'typescript',
                '.sh': 'shell',
                '.bash': 'shell'
            }
            if ext in lang_map:
                return lang_map[ext]
        
        # Basic heuristics based on code content
        if 'def ' in code and ':' in code:
            return 'python'
        elif 'function ' in code or '=>' in code:
            return 'javascript'
        elif 'function ' in code and '()' in code:
            return 'shell'
        
        return 'unknown'
    
    async def _find_similar_functions(self, func_info: Dict) -> List[SimilarityMatch]:
        """Find similar functions in the project index."""
        if not self.project_index or not self.ollama_manager:
            return []
        
        # Create embedding text for the new function
        embedding_text = self._create_embedding_text(func_info)
        
        # Generate embedding for new function
        success, new_embedding, error = self.ollama_manager.generate_embedding(
            embedding_text, self.config.embedding_model
        )
        
        if not success:
            print(f"Warning: Could not generate embedding: {error}", file=sys.stderr)
            return []
        
        # Compare against existing functions
        matches = []
        
        # Handle both compressed and original index formats
        files_data = self.project_index.get('files', {}) or self._expand_compressed_index()
        
        for file_path, file_info in files_data.items():
            if not isinstance(file_info, dict):
                continue
                
            # Check functions
            for func_name, func_data in file_info.get('functions', {}).items():
                if isinstance(func_data, dict) and 'embedding' in func_data:
                    similarity = self._cosine_similarity(new_embedding, func_data['embedding'])
                    
                    if similarity >= self.config.similarity_threshold:
                        matches.append(SimilarityMatch(
                            file_path=file_path,
                            function_name=func_name,
                            similarity_score=similarity,
                            signature=func_data.get('signature', ''),
                            documentation=func_data.get('doc', ''),
                            line_number=func_data.get('line', 0),
                            code_snippet=self._get_function_code_snippet(file_path, func_name, func_data.get('line', 0))
                        ))
            
            # Check class methods
            for class_name, class_data in file_info.get('classes', {}).items():
                if isinstance(class_data, dict):
                    for method_name, method_data in class_data.get('methods', {}).items():
                        if isinstance(method_data, dict) and 'embedding' in method_data:
                            similarity = self._cosine_similarity(new_embedding, method_data['embedding'])
                            
                            if similarity >= self.config.similarity_threshold:
                                matches.append(SimilarityMatch(
                                    file_path=file_path,
                                    function_name=f"{class_name}.{method_name}",
                                    similarity_score=similarity,
                                    signature=method_data.get('signature', ''),
                                    documentation=method_data.get('doc', ''),
                                    line_number=method_data.get('line', 0),
                                    code_snippet=self._get_function_code_snippet(file_path, method_name, method_data.get('line', 0))
                                ))
        
        return matches
    
    def _expand_compressed_index(self) -> Dict:
        """Expand compressed index format to access embeddings."""
        # If using compressed format, we need embeddings to have been added
        # This is a simplified expansion - in practice, we'd use the full expansion logic
        return {}
    
    def _create_embedding_text(self, func_info: Dict) -> str:
        """Create text representation for embedding generation."""
        text = f"Function: {func_info['name']}\n"
        if func_info.get('signature'):
            text += f"Signature: {func_info['signature']}\n"
        if func_info.get('doc'):
            text += f"Documentation: {func_info['doc']}\n"
        return text
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        if len(vec1) != len(vec2):
            return 0.0
        
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = sum(a * a for a in vec1) ** 0.5
        magnitude2 = sum(a * a for a in vec2) ** 0.5
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        return dot_product / (magnitude1 * magnitude2)
    
    def _get_function_code_snippet(self, file_path: str, func_name: str, line_number: int) -> str:
        """Get code snippet for a function from the actual file."""
        try:
            # Try to read the actual file to get code snippet
            full_path = Path(file_path)
            if full_path.exists():
                with open(full_path) as f:
                    lines = f.readlines()
                    
                # Get a few lines around the function definition
                start = max(0, line_number - 1)
                end = min(len(lines), line_number + 10)  # Show ~10 lines
                
                snippet_lines = lines[start:end]
                return ''.join(snippet_lines).strip()
        except Exception:
            pass
        
        return f"# Function: {func_name} (line {line_number})"
    
    async def handle_blocking_mode(self, matches: List[SimilarityMatch]) -> ReviewAction:
        """Handle blocking mode - require manual review."""
        if not matches:
            return ReviewAction.PROCEED_NEW
        
        print("\n🚫 GUARDIAN MODE: Similar code detected!")
        print(f"Found {len(matches)} similar function(s) with >{self.config.similarity_threshold*100:.0f}% similarity:")
        
        for i, match in enumerate(matches, 1):
            print(f"\n{i}. {match.function_name} ({match.file_path}:{match.line_number})")
            print(f"   Similarity: {match.similarity_score:.1%}")
            print(f"   Signature: {match.signature}")
            if match.documentation:
                print(f"   Doc: {match.documentation}")
            
            # Show code snippet
            print("   Code preview:")
            for line in match.code_snippet.split('\n')[:5]:
                print(f"     {line}")
            if len(match.code_snippet.split('\n')) > 5:
                print("     ...")
        
        # Interactive review
        print(f"\nChoose an action:")
        print("1. Use existing function")
        print("2. Modify existing to be more generic")  
        print("3. Proceed with new implementation")
        print("4. Refactor both into shared utility")
        print("5. Cancel operation")
        
        while True:
            try:
                choice = input("\nEnter your choice (1-5): ").strip()
                action_map = {
                    '1': ReviewAction.USE_EXISTING,
                    '2': ReviewAction.MODIFY_EXISTING,
                    '3': ReviewAction.PROCEED_NEW,
                    '4': ReviewAction.REFACTOR_BOTH,
                    '5': ReviewAction.CANCEL
                }
                
                if choice in action_map:
                    return action_map[choice]
                else:
                    print("Invalid choice. Please enter 1-5.")
            except KeyboardInterrupt:
                return ReviewAction.CANCEL
    
    def create_advisory_context(self, matches: List[SimilarityMatch]) -> str:
        """Create context injection for advisory mode."""
        if not matches:
            return ""
        
        matches_text = ""
        for i, match in enumerate(matches, 1):
            matches_text += f"\n{i}. **{match.function_name}** ({match.file_path}:{match.line_number})\n"
            matches_text += f"   Similarity: {match.similarity_score:.1%}\n"
            matches_text += f"   Signature: `{match.signature}`\n"
            
            if match.documentation:
                matches_text += f"   Documentation: {match.documentation}\n"
            
            matches_text += "   Code:\n"
            for line in match.code_snippet.split('\n')[:8]:  # Show more lines in advisory mode
                matches_text += f"   ```\n   {line}\n   ```\n"
            
            if len(match.code_snippet.split('\n')) > 8:
                matches_text += "   ...\n"
            matches_text += "\n"
        
        return self.config.advisory_context_template.format(matches=matches_text)


# CLI Interface for testing
async def main():
    """Test the real-time guardian system."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test Real-Time Code Guardian')
    parser.add_argument('--mode', choices=['blocking', 'advisory'], default='advisory')
    parser.add_argument('--threshold', type=float, default=0.85)
    parser.add_argument('--code', required=True, help='Code to check for similarity')
    parser.add_argument('--file', help='File path context')
    
    args = parser.parse_args()
    
    config = GuardianConfig(
        mode=GuardianMode.BLOCKING if args.mode == 'blocking' else GuardianMode.ADVISORY,
        similarity_threshold=args.threshold
    )
    
    guardian = RealtimeGuardian(config)
    
    try:
        await guardian.initialize()
        matches = await guardian.check_code_similarity(args.code, args.file)
        
        if config.mode == GuardianMode.BLOCKING:
            action = await guardian.handle_blocking_mode(matches)
            print(f"\nAction taken: {action.value}")
        else:
            context = guardian.create_advisory_context(matches)
            print(context)
            
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    asyncio.run(main())