#!/usr/bin/env bash
# Guardian System - One-Command Setup
# Real-Time Code Duplication Prevention for Claude Code
#
# Usage:
#   curl -fsSL https://guardian.dev/install.sh | bash
#   # OR
#   ./install-guardian.sh
#
# What it does in 5 minutes:
# 1. ✅ Detects OS and installs Ollama (NixOS/macOS/Linux)
# 2. ✅ Sets up Guardian scripts 
# 3. ✅ Generates PROJECT_INDEX.json automatically
# 4. ✅ Downloads embedding model with progress
# 5. ✅ Configures Claude Code hooks seamlessly
# 6. ✅ Validates everything works
# 7. ✅ Shows you similar code examples

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m'

# Configuration
GUARDIAN_VERSION="0.1.0"
SCRIPTS_DIR="./.claude/guardian"

# Logging
log_info() { echo -e "${BLUE}ℹ${NC} $1"; }
log_success() { echo -e "${GREEN}✅${NC} $1"; }
log_warning() { echo -e "${YELLOW}⚠${NC} $1"; }
log_error() { echo -e "${RED}❌${NC} $1"; }
log_step() { echo -e "\n${BOLD}${BLUE}🚀 $1${NC}"; }

progress_dots() {
    local pid=$1
    local message="$2"
    local dots=0
    while kill -0 $pid 2>/dev/null; do
        dots=$((dots + 1))
        printf "\r$message"
        for ((i=1; i<=((dots % 4)); i++)); do printf "."; done
        printf "   "
        sleep 1
    done
    printf "\r\033[K"
}

# OS Detection
detect_os() {
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        if [ -f /etc/nixos/configuration.nix ] || [ -d /nix/store ] || command -v nix &> /dev/null; then
            echo "nixos"
        elif [ -f /etc/debian_version ]; then
            echo "debian"
        else
            echo "linux"
        fi
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        echo "macos"
    else
        echo "unknown"
    fi
}

# Prerequisites check
check_prereqs() {
    log_step "Checking prerequisites"
    
    # Python 3
    if ! command -v python3 &> /dev/null; then
        log_error "Python 3 is required but not installed"
        exit 1
    fi
    log_success "Python 3 found"
    
    # Project detection
    if [ ! -d ".git" ] && [ ! -f "package.json" ] && [ ! -f "requirements.txt" ] && [ ! -f "Cargo.toml" ]; then
        log_warning "Not in a recognized project directory - Guardian will still work"
    else
        log_success "Project directory detected"
    fi
    
    # Disk space (need ~200MB)
    available=$(df . 2>/dev/null | tail -1 | awk '{print $4}' || echo "999999")
    if [ "$available" -lt 200000 ]; then
        log_warning "Low disk space detected - Guardian needs ~200MB for models"
    fi
}

# Install Ollama based on OS
install_ollama() {
    log_step "Setting up Ollama (AI embedding service)"
    
    local os=$(detect_os)
    
    case $os in
        "nixos")
            log_info "🔍 NixOS detected"
            if ! pgrep -f "ollama" > /dev/null; then
                log_info "Starting Ollama with Nix..."
                echo "💡 For permanent installation, add 'ollama' to your system packages"
                nix run nixpkgs#ollama -- serve &
                OLLAMA_PID=$!
                sleep 5
            else
                log_success "Ollama is already running"
            fi
            ;;
        "macos")
            if ! command -v ollama &> /dev/null; then
                if command -v brew &> /dev/null; then
                    log_info "Installing Ollama via Homebrew..."
                    brew install ollama &
                    progress_dots $! "🍺 Installing via Homebrew"
                else
                    log_info "Installing Ollama directly..."
                    curl -fsSL https://ollama.com/install.sh | sh
                fi
            else
                log_success "Ollama already installed"
            fi
            ;;
        "debian"|"linux")
            if ! command -v ollama &> /dev/null; then
                log_info "Installing Ollama..."
                curl -fsSL https://ollama.com/install.sh | sh
            else
                log_success "Ollama already installed"
            fi
            ;;
        *)
            log_error "Unsupported OS: $os"
            echo "Please install Ollama manually:"
            echo "• NixOS: nix run nixpkgs#ollama"  
            echo "• Others: https://ollama.com"
            exit 1
            ;;
    esac
    
    # Start Ollama if needed
    if ! pgrep -f "ollama" > /dev/null; then
        log_info "Starting Ollama service..."
        ollama serve &
        OLLAMA_PID=$!
        sleep 3
    fi
    
    # Verify connection
    local retries=10
    while [ $retries -gt 0 ]; do
        if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
            log_success "Ollama service is ready"
            break
        fi
        retries=$((retries - 1))
        if [ $retries -eq 0 ]; then
            log_error "Failed to connect to Ollama service"
            exit 1
        fi
        printf "⏳ Waiting for Ollama ($retries)...\r"
        sleep 2
    done
    printf "\033[K"
}

# Download embedding model
setup_embedding_model() {
    log_step "Setting up embedding model"
    
    if ollama list 2>/dev/null | grep -q "nomic-embed-text"; then
        log_success "Embedding model already available"
        return
    fi
    
    log_info "📥 Downloading nomic-embed-text model (~275MB)..."
    ollama pull nomic-embed-text &
    local pull_pid=$!
    progress_dots $pull_pid "📥 Downloading embedding model"
    
    wait $pull_pid
    if [ $? -eq 0 ]; then
        log_success "Embedding model ready"
    else
        log_error "Failed to download embedding model"
        exit 1
    fi
}

# Install Guardian scripts
install_guardian() {
    log_step "Installing Guardian system"
    
    mkdir -p "$SCRIPTS_DIR"
    
    local required_scripts=(
        "realtime_guardian.py"
        "claude_guardian_hook.py"
    )
    
    local helper_scripts=(
        "project_index.py"
        "index_utils.py"
        "find_ollama.py"
        "append_embeddings_to_index.py"
        "test_guardian.py"
    )
    
    # Copy scripts from current directory
    if [ -d "./scripts" ]; then
        log_info "Installing Guardian scripts..."
        for script in "${required_scripts[@]}" "${helper_scripts[@]}"; do
            if [ -f "./scripts/$script" ]; then
                cp "./scripts/$script" "$SCRIPTS_DIR/"
                chmod +x "$SCRIPTS_DIR/$script"
            else
                log_warning "Script $script not found"
            fi
        done
        log_success "Guardian scripts installed"
    else
        log_error "Guardian scripts not found - please run from Guardian repository"
        exit 1
    fi
}

# Create configuration
setup_config() {
    log_step "Creating smart configuration"
    
    # Detect project type for better defaults
    local project_type="general"
    local languages='["python", "javascript", "typescript", "shell"]'
    local ignore_patterns='["test_*", "*_test.py", "tests/*", "spec/*", "__tests__/*"]'
    
    if [ -f "package.json" ]; then
        project_type="nodejs"
        languages='["javascript", "typescript", "jsx", "tsx"]'
        ignore_patterns='["test_*", "*.test.js", "*.spec.js", "__tests__/*", "node_modules/*"]'
    elif [ -f "requirements.txt" ] || [ -f "pyproject.toml" ]; then
        project_type="python"
        languages='["python"]'
        ignore_patterns='["test_*", "*_test.py", "tests/*", "__pycache__/*", "*.pyc"]'
    elif [ -f "Cargo.toml" ]; then
        project_type="rust"
        languages='["rust"]'
        ignore_patterns='["tests/*", "target/*", "*.test.rs"]'
    elif [ -f "go.mod" ]; then
        project_type="go"
        languages='["go"]'
        ignore_patterns='["*_test.go", "tests/*", "testdata/*"]'
    fi
    
    cat > "guardian-config.json" << EOF
{
  "mode": "advisory",
  "similarity_threshold": 0.85,
  "max_matches": 5,
  "embedding_model": "nomic-embed-text",
  "embedding_endpoint": "http://localhost:11434",
  "languages": $languages,
  "ignore_patterns": $ignore_patterns,
  "project_type": "$project_type"
}
EOF
    
    log_success "Configuration created for $project_type project"
}

# Generate project index
create_index() {
    log_step "Analyzing your codebase"
    
    log_info "🔍 Scanning files and extracting functions..."
    python3 "$SCRIPTS_DIR/project_index.py" &
    local index_pid=$!
    progress_dots $index_pid "🔍 Building project index"
    
    wait $index_pid
    if [ $? -eq 0 ] && [ -f "PROJECT_INDEX.json" ]; then
        # Count functions found
        local func_count=$(python3 -c "
import json
try:
    with open('PROJECT_INDEX.json') as f:
        data = json.load(f)
    count = 0
    for file_data in data.get('f', {}).values():
        if isinstance(file_data, list) and len(file_data) > 1:
            count += len(file_data[1])
    print(count)
except:
    print(0)
" 2>/dev/null)
        log_success "Project indexed ($func_count functions found)"
    else
        log_error "Failed to generate project index"
        exit 1
    fi
}

# Add embeddings
generate_embeddings() {
    log_step "Generating neural embeddings"
    
    log_info "🧠 Creating semantic embeddings for similarity detection..."
    python3 "$SCRIPTS_DIR/append_embeddings_to_index.py" &
    local embed_pid=$!
    progress_dots $embed_pid "🧠 Generating embeddings"
    
    wait $embed_pid
    log_success "Neural embeddings generated"
}

# Configure Claude Code hooks
setup_hooks() {
    log_step "Integrating with Claude Code"
    
    mkdir -p ".claude"
    
    local hook_config='{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Write|Edit|MultiEdit",
        "hooks": [
          {
            "type": "command",
            "command": "$CLAUDE_PROJECT_DIR/.claude/guardian/claude_guardian_hook.py"
          }
        ]
      }
    ]
  }
}'
    
    if [ -f ".claude/settings.json" ]; then
        log_info "Merging with existing Claude settings..."
        # Simple append for now - in production, do proper JSON merge
        echo "$hook_config" > ".claude/guardian-hooks.json"
        log_warning "Created separate hook file: .claude/guardian-hooks.json"
        log_info "Merge this with your .claude/settings.json manually"
    else
        echo "$hook_config" > ".claude/settings.json"
        log_success "Claude Code hooks configured"
    fi
    
    chmod +x "$SCRIPTS_DIR/claude_guardian_hook.py"
}

# Test everything works
validate_system() {
    log_step "Validating installation"
    
    local checks=0
    local total=6
    
    # Check files exist
    [ -f "PROJECT_INDEX.json" ] && { log_success "✓ Project index"; checks=$((checks+1)); } || log_error "✗ Project index missing"
    [ -f "guardian-config.json" ] && { log_success "✓ Configuration"; checks=$((checks+1)); } || log_error "✗ Configuration missing"
    [ -x "$SCRIPTS_DIR/claude_guardian_hook.py" ] && { log_success "✓ Guardian hook"; checks=$((checks+1)); } || log_error "✗ Hook script missing"
    
    # Check services
    curl -s http://localhost:11434/api/tags > /dev/null 2>&1 && { log_success "✓ Ollama service"; checks=$((checks+1)); } || log_error "✗ Ollama not responding"
    ollama list 2>/dev/null | grep -q "nomic-embed-text" && { log_success "✓ Embedding model"; checks=$((checks+1)); } || log_error "✗ Model missing"
    
    # Test Guardian
    if python3 "$SCRIPTS_DIR/test_guardian.py" 2>/dev/null | grep -q "Guardian"; then
        log_success "✓ Guardian system working"
        checks=$((checks+1))
    else
        log_warning "✗ Guardian test had issues"
    fi
    
    echo
    if [ $checks -eq $total ]; then
        log_success "🎉 All systems operational!"
        return 0
    else
        log_warning "⚠️ $checks/$total checks passed"
        return 1
    fi
}

# Show example
demo_guardian() {
    log_step "Testing similarity detection"
    
    log_info "Looking for similar patterns in your code..."
    if python3 "$SCRIPTS_DIR/test_guardian.py" 2>/dev/null | head -10; then
        echo
        log_success "✨ Guardian detected code patterns successfully!"
    else
        log_info "No similar patterns found yet (normal for small/new projects)"
    fi
}

# Cleanup
cleanup() {
    if [ -n "${OLLAMA_PID:-}" ]; then
        kill $OLLAMA_PID 2>/dev/null || true
    fi
}
trap cleanup EXIT

# Main installation
main() {
    echo -e "${BOLD}${BLUE}"
    echo "🛡️  Guardian System Installer v$GUARDIAN_VERSION"
    echo "   Real-Time Code Duplication Prevention for Claude Code"
    echo -e "${NC}"
    echo
    echo "⚡ Setting up in ~5 minutes with zero configuration required"
    echo
    
    check_prereqs
    install_ollama
    setup_embedding_model
    install_guardian
    setup_config
    create_index
    generate_embeddings
    setup_hooks
    
    echo
    if validate_system; then
        demo_guardian
        
        echo
        echo -e "${GREEN}${BOLD}🎉 Guardian is now protecting your code!${NC}"
        echo
        echo "✨ What's ready:"
        echo "• 🛡️  Real-time duplicate detection in Claude Code"  
        echo "• 🧠 AI-powered similarity analysis"
        echo "• 📝 Advisory mode: Claude gets smart context automatically"
        echo "• ⚡ Sub-second response time"
        echo
        echo "🔧 Quick commands:"
        echo "• Test: python3 $SCRIPTS_DIR/test_guardian.py"
        echo "• Update index: python3 $SCRIPTS_DIR/project_index.py"
        echo "• Configure: edit guardian-config.json"
        echo
        echo "📖 How it works:"
        echo "• Write code in Claude Code as normal"
        echo "• Guardian automatically detects similar functions"
        echo "• Claude gets context about existing patterns"  
        echo "• You decide: reuse, extend, or create new"
        echo
        echo -e "${BLUE}Start coding - Guardian is watching for duplicates! 🚀${NC}"
        echo
        echo "💡 Try writing a function similar to existing code to see Guardian in action"
    else
        log_error "Installation completed with issues"
        echo "Please check the errors above and run the installer again"
        exit 1
    fi
}

main "$@"