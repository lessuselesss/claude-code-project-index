# Real-Time Guardian Setup Guide

## 🚀 Quick Start

The Real-Time Guardian System prevents code duplication by checking similarity in real-time as Claude Code writes functions.

### Prerequisites

1. **PROJECT_INDEX.json** with embeddings:
   ```bash
   # Generate base index
   python3 scripts/project_index.py
   
   # Add neural embeddings (required for similarity detection)  
   python3 scripts/append_embeddings_to_index.py
   ```

2. **Ollama running locally** (for embeddings):
   ```bash
   # Install Ollama
   curl -fsSL https://ollama.com/install.sh | sh
   
   # Pull embedding model
   ollama pull nomic-embed-text
   ```

### Installation

1. **Copy Guardian scripts** to your project:
   ```bash
   cp scripts/realtime_guardian.py /your/project/scripts/
   cp scripts/claude_code_guardian_hook.py /your/project/scripts/
   cp guardian-config.json /your/project/
   ```

2. **Test the system**:
   ```bash
   cd /your/project
   python3 scripts/test_guardian.py
   ```

## 🛡️ Guardian Modes

### Advisory Mode (Recommended)
- **Non-blocking** - Claude Code continues normally  
- **Context injection** - Feeds similar functions to Claude
- **Smart decisions** - Claude chooses whether to reuse or create new

**Example Output:**
```
💡 Found 2 similar function(s) - context injected for Claude's consideration

## 🔍 Similar Functions Found

1. **authenticate_user** (auth/login.py:45)
   Similarity: 87%
   Signature: `def authenticate_user(username: str, password: str) -> User | None`
   Code:
   ```python
   def authenticate_user(username, password):
       hashed = hash_password(password)
       user = db.get_user(username)
       return user if user and user.password_hash == hashed else None
   ```
```

### Blocking Mode (Strict)
- **Blocks Claude** when similarity > threshold
- **Interactive review** - User decides how to proceed
- **5 options**: Use existing, modify existing, proceed new, refactor both, cancel

**Example Interaction:**
```
🚫 GUARDIAN MODE: Similar code detected!
Found 2 similar function(s) with >85% similarity:

1. authenticate_user (auth/login.py:45)  
   Similarity: 87%
   Signature: def authenticate_user(username: str, password: str) -> User | None
   Code preview:
     def authenticate_user(username, password):
         hashed = hash_password(password)
         ...

Choose an action:
1. Use existing function
2. Modify existing to be more generic
3. Proceed with new implementation  
4. Refactor both into shared utility
5. Cancel operation

Enter your choice (1-5): _
```

## ⚙️ Configuration

Edit `guardian-config.json`:

```json
{
  "mode": "advisory",              // "advisory", "blocking", or "disabled"
  "similarity_threshold": 0.85,    // 0.0-1.0 (85% similarity to trigger)
  "max_matches": 5,                // Max similar functions to show
  "embedding_model": "nomic-embed-text",
  "embedding_endpoint": "http://localhost:11434",
  "languages": ["python", "javascript", "typescript", "shell"],
  "ignore_patterns": ["test_*", "*_test.py", "tests/*"]
}
```

**Threshold Guidelines:**
- `0.95+` - Nearly identical code only
- `0.85` - Similar logic and structure (recommended)
- `0.75` - Related functionality
- `0.65` - Loose similarity

## 🔗 Claude Code Integration

### Option 1: Hook Integration (Automatic)

1. **Add to Claude Code hooks** (when supported):
   ```bash
   # Copy hook to Claude Code hooks directory
   cp scripts/claude_code_guardian_hook.py ~/.claude-code/hooks/
   ```

2. **Enable in Claude Code settings**:
   ```json
   {
     "hooks": {
       "user_prompt_submit": "guardian_hook.user_prompt_submit_hook",
       "tool_call": "guardian_hook.tool_call_hook"
     }
   }
   ```

### Option 2: Manual Integration (Current)

1. **Before coding sessions**, run:
   ```bash
   python3 scripts/realtime_guardian.py --mode advisory --code "your code here"
   ```

2. **Use advisory context** in your prompts:
   ```
   I want to write an authentication function.
   
   [Run guardian check first, then include results in prompt]
   
   Similar functions found: authenticate_user() in auth/login.py
   Please consider reusing or building upon this existing pattern.
   ```

## 🧪 Testing

### Basic Test
```bash
python3 scripts/test_guardian.py
```

### Manual Testing
```bash
# Test advisory mode
python3 scripts/realtime_guardian.py \
  --mode advisory \
  --threshold 0.8 \
  --code "def login_user(email, pwd): return auth.verify(email, pwd)"

# Test blocking mode  
python3 scripts/realtime_guardian.py \
  --mode blocking \
  --threshold 0.8 \
  --code "def authenticate(user, pass): return check_credentials(user, pass)"
```

## 📊 Expected Results

### Successful Setup
- ✅ Guardian detects similar functions (>85% similarity)
- ✅ Advisory mode injects context without blocking
- ✅ Blocking mode presents interactive choices
- ✅ Performance: <1 second for similarity checks
- ✅ Reduced code duplication in your project

### Troubleshooting

**"No embeddings found"**
```bash
# Generate embeddings
python3 scripts/append_embeddings_to_index.py
```

**"Could not connect to Ollama"** 
```bash
# Start Ollama
ollama serve

# Verify model is available
ollama list | grep nomic-embed-text
```

**"PROJECT_INDEX.json not found"**
```bash
# Generate base index
python3 scripts/project_index.py
```

**"Guardian check too slow"**
- Reduce `max_matches` in config
- Use faster embedding model
- Enable `cache_embeddings`

## 🎯 Best Practices

### Advisory Mode Tips
- Start with advisory mode for non-disruptive experience
- Review Claude's decisions and provide feedback
- Gradually train Claude on your project patterns

### Blocking Mode Tips  
- Use for critical codebases where duplication must be prevented
- Set appropriate thresholds (0.85 recommended)
- Create shared utilities when refactoring is suggested

### Performance Tips
- Keep PROJECT_INDEX.json updated
- Use incremental embedding updates (future feature)
- Cache frequently accessed embeddings

### Team Usage
- Share `guardian-config.json` in version control
- Establish team conventions for similarity thresholds
- Use blocking mode for code reviews

## 🚀 Next Steps

1. **Test on your codebase** - Start with advisory mode
2. **Tune thresholds** - Adjust based on your project's needs  
3. **Integrate workflows** - Add to development process
4. **Provide feedback** - Help improve the system
5. **Extend languages** - Add support for more programming languages

## 🤝 Contributing

Found issues or want to improve the Guardian?

- **Report bugs**: Include config, error messages, and code samples
- **Suggest features**: New similarity algorithms, better UI, language support
- **Performance improvements**: Faster embedding, smarter caching
- **Integration**: IDE plugins, CI/CD workflows, team features

---

**The Guardian System represents a new paradigm in AI-assisted development - preventing code duplication before it happens rather than detecting it after. This is the future of intelligent development tools.**