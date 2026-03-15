#!/usr/bin/env bash
set -euo pipefail

OUT_ROOT="${1:-benchmarks/results/gpu_h100_validation_$(date +%Y%m%d_%H%M%S)}"
BASELINE_PATH="${OUT_ROOT}/baselines/cuda_h100_standard.json"

if ! command -v uv >/dev/null 2>&1; then
  echo "error: uv not found in PATH" >&2
  exit 1
fi

mkdir -p "${OUT_ROOT}/baselines"

echo "Verifying Torch CUDA visibility..."
uv run python -c 'import torch
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("cuda_count", torch.cuda.device_count())
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available to Torch in this environment.")
print("device_names", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])'

echo "Inspecting detected dagzoo hardware tier..."
uv run dagzoo hardware --device cuda

echo "Running primary H100 smoke benchmark..."
uv run dagzoo benchmark \
  --preset cuda_h100 \
  --suite smoke \
  --hardware-policy cuda_tiered_v1 \
  --out-dir "${OUT_ROOT}/cuda_h100_smoke"

echo "Running primary H100 standard benchmark..."
uv run dagzoo benchmark \
  --preset cuda_h100 \
  --suite standard \
  --hardware-policy cuda_tiered_v1 \
  --out-dir "${OUT_ROOT}/cuda_h100_standard" \
  --save-baseline "${BASELINE_PATH}"

declare -a FEATURE_RUNS=(
  "filter_smoke:configs/preset_filter_benchmark_smoke.yaml"
  "missingness_mar_smoke:configs/preset_missingness_mar.yaml"
  "shift_smoke:configs/preset_shift_benchmark_smoke.yaml"
  "noise_smoke:configs/preset_noise_benchmark_smoke.yaml"
  "many_class_smoke:configs/preset_many_class_benchmark_smoke.yaml"
  "mechanism_baseline_smoke:configs/preset_mechanism_baseline_benchmark_smoke.yaml"
  "mechanism_gp_smoke:configs/preset_mechanism_gp_benchmark_smoke.yaml"
  "mechanism_piecewise_smoke:configs/preset_mechanism_piecewise_benchmark_smoke.yaml"
)

for run_spec in "${FEATURE_RUNS[@]}"; do
  run_name="${run_spec%%:*}"
  config_path="${run_spec#*:}"
  echo "Running GPU validation preset: ${run_name}"
  uv run dagzoo benchmark \
    --config "${config_path}" \
    --preset custom \
    --device cuda \
    --suite smoke \
    --hardware-policy none \
    --no-memory \
    --out-dir "${OUT_ROOT}/${run_name}"
done

echo "GPU validation artifacts written under: ${OUT_ROOT}"
echo "Primary H100 baseline artifact: ${BASELINE_PATH}"
