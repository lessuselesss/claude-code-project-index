#!/usr/bin/env python3
"""
Test script for the Real-Time Guardian System
Tests both blocking and advisory modes with sample code.
"""

import asyncio
import json
from pathlib import Path
from realtime_guardian import RealtimeGuardian, GuardianConfig, GuardianMode

async def test_guardian_modes():
    """Test both guardian modes with sample code."""
    
    # Test code that should trigger similarity detection
    test_codes = [
        {
            "name": "Authentication Function",
            "code": """
def authenticate_user(username, password):
    \"\"\"Authenticate a user with username and password.\"\"\"
    # Hash the password
    hashed = hash_password(password)
    
    # Check against database
    user = db.get_user(username)
    if user and user.password_hash == hashed:
        return user
    return None
""",
            "file_path": "test_auth.py"
        },
        {
            "name": "Data Processing Function", 
            "code": """
def process_user_data(data):
    \"\"\"Process user data with validation.\"\"\"
    if not data:
        return None
    
    # Validate required fields
    required = ['name', 'email']
    for field in required:
        if field not in data:
            raise ValueError(f"Missing required field: {field}")
    
    # Clean and normalize
    cleaned = {}
    for key, value in data.items():
        cleaned[key.lower()] = str(value).strip()
    
    return cleaned
""",
            "file_path": "test_data.py"
        }
    ]
    
    print("🧪 Testing Real-Time Guardian System\n")
    
    # Test Advisory Mode
    print("=" * 60)
    print("📝 TESTING ADVISORY MODE")  
    print("=" * 60)
    
    advisory_config = GuardianConfig(
        mode=GuardianMode.ADVISORY,
        similarity_threshold=0.75  # Lower threshold for testing
    )
    
    advisory_guardian = RealtimeGuardian(advisory_config)
    
    try:
        await advisory_guardian.initialize()
        print("✅ Advisory Guardian initialized successfully\n")
        
        for test_case in test_codes:
            print(f"Testing: {test_case['name']}")
            print("-" * 40)
            
            matches = await advisory_guardian.check_code_similarity(
                test_case['code'], 
                test_case['file_path']
            )
            
            if matches:
                print(f"🔍 Found {len(matches)} similar function(s):")
                for i, match in enumerate(matches, 1):
                    print(f"  {i}. {match.function_name} ({match.similarity_score:.1%} similar)")
                    print(f"     File: {match.file_path}:{match.line_number}")
                
                # Show advisory context
                context = advisory_guardian.create_advisory_context(matches)
                print(f"\n📋 Advisory Context Generated:")
                print(context[:500] + "..." if len(context) > 500 else context)
            else:
                print("✨ No similar functions found")
            
            print("\n" + "=" * 40 + "\n")
    
    except Exception as e:
        print(f"❌ Advisory mode test failed: {e}")
    
    # Test Blocking Mode (simulated - no actual user input)
    print("=" * 60)
    print("🚫 TESTING BLOCKING MODE (SIMULATED)")
    print("=" * 60)
    
    blocking_config = GuardianConfig(
        mode=GuardianMode.BLOCKING,
        similarity_threshold=0.75
    )
    
    blocking_guardian = RealtimeGuardian(blocking_config)
    
    try:
        await blocking_guardian.initialize()
        print("✅ Blocking Guardian initialized successfully\n")
        
        for test_case in test_codes:
            print(f"Testing: {test_case['name']}")
            print("-" * 40)
            
            matches = await blocking_guardian.check_code_similarity(
                test_case['code'],
                test_case['file_path'] 
            )
            
            if matches:
                print(f"🚫 WOULD BLOCK: Found {len(matches)} similar function(s)")
                print("   User would see:")
                
                for i, match in enumerate(matches, 1):
                    print(f"\n   {i}. {match.function_name} ({match.file_path}:{match.line_number})")
                    print(f"      Similarity: {match.similarity_score:.1%}")
                    print(f"      Signature: {match.signature}")
                    
                    if match.documentation:
                        print(f"      Doc: {match.documentation}")
                    
                    # Show code preview
                    print("      Code preview:")
                    for line in match.code_snippet.split('\n')[:3]:
                        print(f"        {line}")
                    if len(match.code_snippet.split('\n')) > 3:
                        print("        ...")
                
                print(f"\n   🎯 User would choose from:")
                print(f"      1. Use existing function")
                print(f"      2. Modify existing to be more generic") 
                print(f"      3. Proceed with new implementation")
                print(f"      4. Refactor both into shared utility")
                print(f"      5. Cancel operation")
            else:
                print("✅ WOULD ALLOW: No similar functions found")
            
            print("\n" + "=" * 40 + "\n")
    
    except Exception as e:
        print(f"❌ Blocking mode test failed: {e}")


async def test_performance():
    """Test performance of similarity checking."""
    print("⚡ PERFORMANCE TEST")
    print("=" * 60)
    
    config = GuardianConfig(similarity_threshold=0.8)
    guardian = RealtimeGuardian(config)
    
    try:
        await guardian.initialize()
        
        # Test with a larger code sample
        large_code = """
def complex_data_processor(raw_data, config_options=None):
    \"\"\"
    Process complex data with multiple transformation steps.
    Supports various input formats and configuration options.
    \"\"\"
    import time
    start_time = time.time()
    
    if not raw_data:
        return {"error": "No data provided", "processed": []}
    
    # Default configuration
    default_config = {
        'normalize_keys': True,
        'validate_types': True,
        'remove_nulls': True,
        'sort_output': False
    }
    
    if config_options:
        default_config.update(config_options)
    
    processed_items = []
    
    for item in raw_data:
        if not isinstance(item, dict):
            continue
            
        processed_item = {}
        
        # Normalize keys if requested
        if default_config['normalize_keys']:
            for key, value in item.items():
                normalized_key = key.lower().replace(' ', '_')
                processed_item[normalized_key] = value
        else:
            processed_item = item.copy()
        
        # Validate types if requested
        if default_config['validate_types']:
            for key, value in processed_item.items():
                if value is None and default_config['remove_nulls']:
                    continue
                # Type validation logic here
                processed_item[key] = str(value) if value is not None else None
        
        # Remove nulls if requested
        if default_config['remove_nulls']:
            processed_item = {k: v for k, v in processed_item.items() if v is not None}
        
        if processed_item:
            processed_items.append(processed_item)
    
    # Sort output if requested
    if default_config['sort_output'] and processed_items:
        processed_items.sort(key=lambda x: str(x.get('id', '')))
    
    end_time = time.time()
    
    return {
        "processed": processed_items,
        "count": len(processed_items),
        "processing_time": end_time - start_time,
        "config_used": default_config
    }
"""
        
        import time
        start = time.time()
        matches = await guardian.check_code_similarity(large_code, "performance_test.py")
        end = time.time()
        
        print(f"⏱️  Similarity check took: {end - start:.3f} seconds")
        print(f"🔍 Found {len(matches)} similar functions")
        
        if matches:
            for match in matches[:3]:  # Show top 3
                print(f"   • {match.function_name}: {match.similarity_score:.1%} similar")
    
    except Exception as e:
        print(f"❌ Performance test failed: {e}")


def detect_nix():
    """Detect if we're on NixOS or have Nix package manager."""
    return (
        Path("/etc/NIXOS").exists() or 
        Path("/nix/store").exists() or
        Path.home().joinpath(".nix-profile").exists()
    )

def check_and_setup_ollama():
    """Check if Ollama is available and set it up if needed."""
    from find_ollama import OllamaManager
    
    manager = OllamaManager()
    is_running, error = manager.check_ollama_running()
    
    if is_running:
        print("✅ Ollama is running")
        return True
    
    print(f"❌ Ollama not running: {error}")
    
    # Check if we can detect Nix
    if detect_nix():
        print("🔍 Detected Nix/NixOS system")
        print("\n🚀 To start Ollama temporarily:")
        print("   nix run nixpkgs#ollama -- serve")
        print("\n💡 In another terminal, pull the embedding model:")
        print("   nix run nixpkgs#ollama -- pull nomic-embed-text")
        print("\n🔧 For permanent installation:")
        print("   Add 'ollama' to your system packages or home-manager")
        return False
    
    # Show general installation guide
    print("\n🚀 To install Ollama:")
    print("   curl -fsSL https://ollama.com/install.sh | sh")
    print("   ollama serve")
    print("\n💡 Then pull the embedding model:")
    print("   ollama pull nomic-embed-text")
    
    return False

async def setup_embeddings():
    """Set up embeddings with proper Ollama integration."""
    from find_ollama import OllamaManager
    
    # Check if Ollama is running
    if not check_and_setup_ollama():
        return False
    
    manager = OllamaManager()
    
    # Check if embedding model is available
    is_available, error = manager.is_model_available("nomic-embed-text")
    if not is_available:
        print(f"📥 Downloading embedding model: {error}")
        success, error = manager.pull_model("nomic-embed-text", show_progress=True)
        if not success:
            print(f"❌ Failed to download model: {error}")
            return False
    
    # Check if embeddings exist in index
    index_path = Path.cwd() / 'PROJECT_INDEX.json'
    try:
        with open(index_path) as f:
            index = json.load(f)
            
        has_embeddings = False
        files_data = index.get('files', {})
        
        for file_info in files_data.values():
            if isinstance(file_info, dict):
                for func_data in file_info.get('functions', {}).values():
                    if isinstance(func_data, dict) and 'embedding' in func_data:
                        has_embeddings = True
                        break
                if has_embeddings:
                    break
        
        if not has_embeddings:
            print("🧠 Generating embeddings for project functions...")
            import subprocess
            result = subprocess.run([
                "python3", "scripts/append_embeddings_to_index.py"
            ], cwd=Path.cwd(), capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"❌ Failed to generate embeddings: {result.stderr}")
                return False
            
            print("✅ Embeddings generated successfully")
        else:
            print("✅ Embeddings already exist")
        
        return True
    
    except Exception as e:
        print(f"❌ Error setting up embeddings: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Starting Real-Time Guardian Tests\n")
    
    # Check if PROJECT_INDEX.json exists
    index_path = Path.cwd() / 'PROJECT_INDEX.json'
    if not index_path.exists():
        print("❌ PROJECT_INDEX.json not found!")
        print("   Please run: python project_index.py first")
        return
    
    # Set up Ollama and embeddings
    print("🔧 Setting up Ollama and embeddings...")
    embeddings_ready = asyncio.run(setup_embeddings())
    
    if not embeddings_ready:
        print("\n❌ Cannot run tests without embeddings")
        print("   Please follow the setup instructions above")
        return
    
    print("\n" + "="*60)
    print("🧪 RUNNING GUARDIAN TESTS")
    print("="*60)
    
    # Run tests
    asyncio.run(test_guardian_modes())
    print()
    asyncio.run(test_performance())
    
    print("\n✅ Guardian testing complete!")
    print("\n📋 Next steps:")
    print("   1. Configure guardian-config.json for your preferences")
    print("   2. Integrate with Claude Code hooks (see claude_code_guardian_hook.py)")
    print("   3. Test with real Claude Code workflows")


if __name__ == '__main__':
    main()