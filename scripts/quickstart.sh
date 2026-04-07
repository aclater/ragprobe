#!/usr/bin/env bash
# Quickstart: Run ragprobe eval pipeline end-to-end
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

echo "=== ragprobe quickstart ==="
echo ""

check_prereqs() {
    echo "Checking prerequisites..."
    
    if ! command -v node &> /dev/null; then
        echo "ERROR: node not found. Install from https://nodejs.org/" >&2
        exit 1
    fi
    
    if ! command -v npm &> /dev/null; then
        echo "ERROR: npm not found. Install from https://nodejs.org/" >&2
        exit 1
    fi
    
    if ! command -v python3 &> /dev/null; then
        echo "ERROR: python3 not found. Install Python 3.11+." >&2
        exit 1
    fi
    
    echo "  node: $(node --version)"
    echo "  npm: $(npm --version)"
    echo "  python: $(python3 --version)"
    echo "OK"
}

setup() {
    echo ""
    echo "Setting up..."
    
    cd "$REPO_ROOT"
    
    if [[ ! -d node_modules ]]; then
        echo "  Installing npm packages..."
        npm install --silent
    else
        echo "  npm packages already installed"
    fi
    
    if ! python3 -c "import ragas_eval" 2>/dev/null; then
        echo "  Installing ragprobe package..."
        python3 -m pip install -e . --quiet
    else
        echo "  ragprobe package already installed"
    fi
    
    if [[ ! -f targets.yaml ]]; then
        if [[ -f targets.yaml.example ]]; then
            echo "  Copying targets.yaml.example to targets.yaml"
            cp targets.yaml.example targets.yaml
            echo ""
            echo "IMPORTANT: Edit targets.yaml with your ragpipe URL and token."
            echo "  vi targets.yaml"
            exit 0
        else
            echo "ERROR: targets.yaml.example not found." >&2
            exit 1
        fi
    fi
    
    echo "OK"
}

run_eval() {
    echo ""
    echo "Running promptfoo eval..."
    
    cd "$REPO_ROOT"
    npx promptfoo eval "$@"
}

show_help() {
    cat <<EOF
ragprobe quickstart script

Usage: $0 [options]

Options:
  --view         Open promptfoo results in browser after eval
  --latest       Show latest stored scores
  --help         Show this help

Environment:
  RAGPROBE_TARGET_URL    ragpipe URL (if not using targets.yaml)
  RAGPROBE_TARGETS_FILE  Path to targets.yaml (default: ./targets.yaml)

Examples:
  # Full setup and run
  $0
  
  # Run eval and open browser to view results
  $0 --view
  
  # Show previously stored scores
  $0 --latest
  
  # Run Ragas eval against specific target
  RAGPROBE_TARGET_URL=http://localhost:8090 python scripts/run_ragas_eval.py --store
  
  # Compare agentic vs baseline
  python scripts/compare_targets.py --baseline baseline --target crag-v1
EOF
}

main() {
    if [[ "${1:-}" == "--help" ]]; then
        show_help
        exit 0
    fi
    
    check_prereqs
    setup
    
    if [[ "${1:-}" == "--view" ]]; then
        shift
        run_eval "$@"
        echo ""
        echo "Opening results in browser..."
        npx promptfoo view
    elif [[ "${1:-}" == "--latest" ]]; then
        if [[ -z "${2:-}" ]]; then
            echo "ERROR: --latest requires a target label" >&2
            echo "Usage: $0 --latest <target-label>" >&2
            exit 1
        fi
        python3 scripts/run_ragas_eval.py --latest --target "$2"
    else
        run_eval "$@"
    fi
    
    echo ""
    echo "Done. Run 'npx promptfoo view' to open the interactive browser UI."
}

main "$@"