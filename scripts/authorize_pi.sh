#!/usr/bin/env bash
# Run on the training PC: enter the Pi password only at ssh-copy-id's prompt.
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$project_root"
pi_target="${1:?Aufruf: bash scripts/authorize_pi.sh BENUTZER@HOST}"
if [[ ! -f work/pi_access_ed25519 || ! -f work/pi_access_ed25519.pub || ! -f work/pi_known_hosts ]]; then
    echo "Pi-Zugangsschlüssel oder bekannte Hostkennung fehlen unter work/." >&2
    exit 1
fi
ssh-copy-id -i work/pi_access_ed25519.pub \
    -o StrictHostKeyChecking=yes \
    -o UserKnownHostsFile=work/pi_known_hosts \
    -o ConnectTimeout=8 \
    "$pi_target"
ssh -i work/pi_access_ed25519 -o IdentitiesOnly=yes -o BatchMode=yes \
    -o StrictHostKeyChecking=yes -o UserKnownHostsFile=work/pi_known_hosts \
    -o ConnectTimeout=8 "$pi_target" \
    'printf "Pi-Zugang bereit: "; whoami; hostname'
