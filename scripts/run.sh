#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"
task="${1:-help}"
if [[ $# -gt 0 ]]; then shift; fi
case "$task" in
  train|play|check|live)
    script="$task"
    [[ "$task" == check ]] && script="check_environment"
    [[ "$task" == live ]] && script="live_sim"
    exec conda run --no-capture-output -n "${SPOT_CONDA_ENV:-unitree_sim_env}" python -u "scripts/$script.py" "$@"
    ;;
  test)
    exec conda run --no-capture-output -n "${SPOT_CONDA_ENV:-unitree_sim_env}" python -m unittest discover -s tests -v "$@"
    ;;
  *)
    echo "Usage: bash scripts/run.sh {check|test|train|play|live} [arguments]"
    ;;
esac
