#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${TRIPOSPLAT_PYTHON:-/private/tmp/triposplat-onnx-venv/bin/python}"
OFFICIAL_REPO="${TRIPOSPLAT_OFFICIAL_REPO:-/private/tmp/triposplat-official-root}"
WEIGHTS="${TRIPOSPLAT_FLOW_WEIGHTS:-/private/tmp/triposplat-weights/diffusion_models/triposplat_fp16.safetensors}"
SERVICE_PORT="${TRIPOSPLAT_MPS_PORT:-8765}"
VITE_PORT="${TRIPOSPLAT_VITE_PORT:-5173}"
TOKEN="${TRIPOSPLAT_MPS_TOKEN:-$(openssl rand -hex 24)}"
HOSTED_RUNNER="${TRIPOSPLAT_RUNNER_URL:-}"
LOG="$(mktemp -t triposplat-mps.XXXXXX.log)"
SERVICE_PID=""
ALLOW_ORIGIN_ARGS=()

cleanup() {
  if [[ -n "${SERVICE_PID}" ]]; then
    kill "${SERVICE_PID}" 2>/dev/null || true
    wait "${SERVICE_PID}" 2>/dev/null || true
  fi
  rm -f "${LOG}"
}
trap cleanup EXIT INT TERM

for required in "${PYTHON}" "${OFFICIAL_REPO}/triposplat.py" "${WEIGHTS}"; do
  if [[ ! -e "${required}" ]]; then
    echo "Missing required Mac MPS dependency: ${required}" >&2
    exit 1
  fi
done

cd "${ROOT}"
if [[ -n "${HOSTED_RUNNER}" ]]; then
  HOSTED_ORIGIN="$("${PYTHON}" -c 'import sys; from urllib.parse import urlsplit; u=urlsplit(sys.argv[1]); print(f"{u.scheme}://{u.netloc}")' "${HOSTED_RUNNER}")"
  ALLOW_ORIGIN_ARGS=(--allow-origin "${HOSTED_ORIGIN}")
fi
"${PYTHON}" scripts/triposplat/run_mac_mps_flow_service.py \
  --triposplat-repo "${OFFICIAL_REPO}" \
  --weights "${WEIGHTS}" \
  --port "${SERVICE_PORT}" \
  --token "${TOKEN}" \
  "${ALLOW_ORIGIN_ARGS[@]}" >"${LOG}" 2>&1 &
SERVICE_PID="$!"

for _ in {1..120}; do
  if ! kill -0 "${SERVICE_PID}" 2>/dev/null; then
    echo "Mac MPS service failed to start:" >&2
    sed -n '1,160p' "${LOG}" >&2
    exit 1
  fi
  if curl --silent --fail "http://127.0.0.1:${SERVICE_PORT}/v1/health" >/dev/null; then
    break
  fi
  sleep 0.25
done

if ! curl --silent --fail "http://127.0.0.1:${SERVICE_PORT}/v1/health" >/dev/null; then
  echo "Mac MPS service did not become ready:" >&2
  sed -n '1,160p' "${LOG}" >&2
  exit 1
fi

if [[ -n "${HOSTED_RUNNER}" ]]; then
  SEPARATOR="?"
  [[ "${HOSTED_RUNNER}" == *"?"* ]] && SEPARATOR="&"
  RUNNER_URL="${HOSTED_RUNNER}${SEPARATOR}mpsService=http://127.0.0.1:${SERVICE_PORT}/&mpsToken=${TOKEN}"
else
  RUNNER_URL="http://127.0.0.1:${VITE_PORT}/e2e-web.html?mpsService=http://127.0.0.1:${SERVICE_PORT}/&mpsToken=${TOKEN}"
fi
echo
echo "Exact 20-step Mac MPS service is ready."
echo "Open this authenticated runner:"
echo "${RUNNER_URL}"
echo

if [[ -n "${HOSTED_RUNNER}" ]]; then
  if command -v open >/dev/null 2>&1; then
    open "${RUNNER_URL}"
  fi
  echo "Keep this terminal open while generating. Press Ctrl-C to stop the service."
  wait "${SERVICE_PID}"
  exit 0
fi

pnpm exec vite --host 127.0.0.1 --port "${VITE_PORT}" --strictPort
