#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"
mkdir -p work/runtime_tmp outputs/classroom_demo
export TMPDIR="$project_dir/work/runtime_tmp"
case "${1:-}" in
  start)
    checkpoint="policies/early_teacher/model_0.pt"
    profile="policies/early_teacher/robot.json"
    ;;
  fertig)
    checkpoint="policies/rough_student/model_final.pt"
    profile="policies/rough_student/flat.json"
    ;;
  *) echo "Aufruf: bash scripts/demo_classroom.sh {start|fertig}"; exit 2 ;;
esac
exec bash scripts/run.sh play --checkpoint "$checkpoint" --robot "$profile" \
  --num-envs 200 --overview --command 0.6 0 0 --steps 30000 \
  --output "outputs/classroom_demo/${1}_metrics.json"
