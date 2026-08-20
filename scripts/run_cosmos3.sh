#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="${COSMOS3_MODEL:-nvidia/Cosmos3-Nano}"

# Local guardrail weights are gated separately from the model. Keep the
# validated no-guardrail default while allowing callers to override it.
export SGLANG_DISABLE_COSMOS3_GUARDRAILS="${SGLANG_DISABLE_COSMOS3_GUARDRAILS:-1}"

exec bash "$ROOT_DIR/scripts/serve_sglang_diffusion.sh" "$MODEL" "$@"
