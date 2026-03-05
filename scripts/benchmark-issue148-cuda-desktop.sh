#!/usr/bin/env bash
set -euo pipefail

REPS=3
SUITE="standard"
ROOT=""
INCLUDE_CPU=1
RUN_PROFILES=1
GEN_PROFILE_DATASETS=400
NUM_DATASETS_OVERRIDE=""
WARMUP_OVERRIDE=""
ALLOW_CPU_FALLBACK=0
DRY_RUN=0
DESKTOP_DEVICE="cuda"

usage() {
  cat <<'USAGE'
Usage: scripts/benchmark-issue148-cuda-desktop.sh [options]

Runs issue #148 prioritization benchmark/profiling matrix for CUDA desktop hosts.

Options:
  --root <path>                 Artifact root directory (default: /tmp/issue148_cuda_desktop_<timestamp>)
  --reps <n>                    Number of benchmark repetitions per scenario (default: 3)
  --suite <smoke|standard|full> Benchmark suite for main matrix (default: standard)
  --num-datasets <n>            Optional override for benchmark dataset count
  --warmup <n>                  Optional override for benchmark warmup count
  --gen-profile-datasets <n>    Dataset count for generate write/no-write profiling (default: 400)
  --skip-cpu                    Skip CPU control scenarios
  --skip-profiles               Skip cProfile passes
  --allow-cpu-fallback          Allow running even when CUDA is unavailable
  --dry-run                     Print commands without executing
  -h, --help                    Show this help message
USAGE
}

err() {
  echo "error: $*" >&2
  exit 1
}

run_cmd() {
  if [[ "$DRY_RUN" -eq 1 ]]; then
    printf '+ '
    printf '%q ' "$@"
    printf '\n'
    return 0
  fi
  "$@"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --root)
      [[ $# -ge 2 ]] || err "--root requires a value"
      ROOT="$2"
      shift 2
      ;;
    --reps)
      [[ $# -ge 2 ]] || err "--reps requires a value"
      REPS="$2"
      shift 2
      ;;
    --suite)
      [[ $# -ge 2 ]] || err "--suite requires a value"
      SUITE="$2"
      shift 2
      ;;
    --num-datasets)
      [[ $# -ge 2 ]] || err "--num-datasets requires a value"
      NUM_DATASETS_OVERRIDE="$2"
      shift 2
      ;;
    --warmup)
      [[ $# -ge 2 ]] || err "--warmup requires a value"
      WARMUP_OVERRIDE="$2"
      shift 2
      ;;
    --gen-profile-datasets)
      [[ $# -ge 2 ]] || err "--gen-profile-datasets requires a value"
      GEN_PROFILE_DATASETS="$2"
      shift 2
      ;;
    --skip-cpu)
      INCLUDE_CPU=0
      shift
      ;;
    --skip-profiles)
      RUN_PROFILES=0
      shift
      ;;
    --allow-cpu-fallback)
      ALLOW_CPU_FALLBACK=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      err "unknown option: $1"
      ;;
  esac
done

if [[ -z "$ROOT" ]]; then
  ROOT="/tmp/issue148_cuda_desktop_$(date +%Y%m%d_%H%M%S)"
fi

DAGZOO_BIN=".venv/bin/dagzoo"
PY_BIN=".venv/bin/python"

[[ -x "$DAGZOO_BIN" ]] || err "missing executable: $DAGZOO_BIN"
[[ -x "$PY_BIN" ]] || err "missing executable: $PY_BIN"

case "$SUITE" in
  smoke|standard|full) ;;
  *) err "--suite must be one of smoke|standard|full (got '$SUITE')" ;;
esac

[[ "$REPS" =~ ^[0-9]+$ ]] || err "--reps must be an integer"
[[ "$GEN_PROFILE_DATASETS" =~ ^[0-9]+$ ]] || err "--gen-profile-datasets must be an integer"

mkdir -p "$ROOT"/{configs,bench,profiles,generate,reports,logs}

echo "artifact_root=$ROOT"
if command -v git >/dev/null 2>&1; then
  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "+ git rev-parse HEAD > $ROOT/commit.txt"
  else
    git rev-parse HEAD > "$ROOT/commit.txt"
  fi
fi

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "+ $DAGZOO_BIN hardware | tee $ROOT/hardware.txt"
else
  "$DAGZOO_BIN" hardware | tee "$ROOT/hardware.txt"
fi

if [[ "$ALLOW_CPU_FALLBACK" -ne 1 ]]; then
  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "+ CUDA availability check via $PY_BIN"
  else
    "$PY_BIN" - <<'PY'
import sys
import torch

if not torch.cuda.is_available():
    print(
        "error: CUDA is not available. Re-run with --allow-cpu-fallback to continue on CPU.",
        file=sys.stderr,
    )
    raise SystemExit(2)
PY
  fi
fi

if [[ "$ALLOW_CPU_FALLBACK" -eq 1 ]]; then
  DESKTOP_DEVICE="auto"
fi

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "+ generate temporary scenario configs under $ROOT/configs"
else
  ROOT="$ROOT" INCLUDE_CPU="$INCLUDE_CPU" NUM_DATASETS_OVERRIDE="$NUM_DATASETS_OVERRIDE" WARMUP_OVERRIDE="$WARMUP_OVERRIDE" \
    "$PY_BIN" - <<'PY'
import copy
import os
from pathlib import Path

import yaml

root = Path(os.environ["ROOT"])
include_cpu = os.environ["INCLUDE_CPU"] == "1"
num_override = os.environ.get("NUM_DATASETS_OVERRIDE", "").strip()
warmup_override = os.environ.get("WARMUP_OVERRIDE", "").strip()

cfg_out = root / "configs"
cfg_out.mkdir(parents=True, exist_ok=True)


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def dump_yaml(path: Path, payload: dict) -> None:
    path.write_text(yaml.safe_dump(payload, sort_keys=False))


def apply_overrides(cfg: dict) -> None:
    if num_override:
        cfg.setdefault("benchmark", {})["num_datasets"] = int(num_override)
    if warmup_override:
        cfg.setdefault("benchmark", {})["warmup_datasets"] = int(warmup_override)


if Path("configs/benchmark_filter_enabled.yaml").exists():
    filter_template = load_yaml(Path("configs/benchmark_filter_enabled.yaml")).get("filter", {})
else:
    base_cpu = load_yaml(Path("configs/benchmark_cpu.yaml"))
    filter_template = copy.deepcopy(base_cpu.get("filter", {}))
    filter_template.update({
        "enabled": True,
        "threshold": 0.5,
        "max_attempts": 10,
        "n_jobs": -1,
    })

base_desktop = load_yaml(Path("configs/benchmark_cuda_desktop.yaml"))
apply_overrides(base_desktop)
dump_yaml(cfg_out / "cuda_desktop_filter_off.yaml", base_desktop)

for tag, n_jobs in (("all", -1), ("1", 1)):
    cfg = copy.deepcopy(base_desktop)
    cfg["filter"] = copy.deepcopy(filter_template)
    cfg["filter"]["enabled"] = True
    cfg["filter"]["n_jobs"] = n_jobs
    cfg.setdefault("benchmark", {})["preset_name"] = f"cuda_desktop_filter_on_njobs{tag}"
    apply_overrides(cfg)
    dump_yaml(cfg_out / f"cuda_desktop_filter_on_njobs{tag}.yaml", cfg)

if include_cpu:
    base_cpu = load_yaml(Path("configs/benchmark_cpu.yaml"))
    apply_overrides(base_cpu)
    dump_yaml(cfg_out / "cpu_filter_off.yaml", base_cpu)

    cpu_filter_on = copy.deepcopy(base_cpu)
    cpu_filter_on["filter"] = copy.deepcopy(filter_template)
    cpu_filter_on["filter"]["enabled"] = True
    cpu_filter_on["filter"]["n_jobs"] = -1
    cpu_filter_on.setdefault("benchmark", {})["preset_name"] = "cpu_filter_on_njobsall"
    apply_overrides(cpu_filter_on)
    dump_yaml(cfg_out / "cpu_filter_on_njobsall.yaml", cpu_filter_on)

    cpu_filter_on_1 = copy.deepcopy(cpu_filter_on)
    cpu_filter_on_1["filter"]["n_jobs"] = 1
    cpu_filter_on_1.setdefault("benchmark", {})["preset_name"] = "cpu_filter_on_njobs1"
    apply_overrides(cpu_filter_on_1)
    dump_yaml(cfg_out / "cpu_filter_on_njobs1.yaml", cpu_filter_on_1)
PY
fi

run_benchmark_scenario() {
  local scenario="$1"
  local config_path="$2"
  local device="$3"
  local rep="$4"

  local out_dir="$ROOT/bench/$scenario/rep${rep}"
  mkdir -p "$out_dir"

  local cmd=(
    "$DAGZOO_BIN" benchmark
    --config "$config_path"
    --preset custom
    --suite "$SUITE"
    --device "$device"
    --no-memory
    --hardware-policy none
    --out-dir "$out_dir"
  )

  if [[ -n "$NUM_DATASETS_OVERRIDE" ]]; then
    cmd+=(--num-datasets "$NUM_DATASETS_OVERRIDE")
  fi
  if [[ -n "$WARMUP_OVERRIDE" ]]; then
    cmd+=(--warmup "$WARMUP_OVERRIDE")
  fi

  run_cmd "${cmd[@]}"
}

echo "running benchmark matrix (reps=$REPS suite=$SUITE include_cpu=$INCLUDE_CPU)"
for rep in $(seq 1 "$REPS"); do
  run_benchmark_scenario "cuda_desktop_filter_off" "$ROOT/configs/cuda_desktop_filter_off.yaml" "$DESKTOP_DEVICE" "$rep"
  run_benchmark_scenario "cuda_desktop_filter_on_njobs_all" "$ROOT/configs/cuda_desktop_filter_on_njobsall.yaml" "$DESKTOP_DEVICE" "$rep"
  run_benchmark_scenario "cuda_desktop_filter_on_njobs1" "$ROOT/configs/cuda_desktop_filter_on_njobs1.yaml" "$DESKTOP_DEVICE" "$rep"

  if [[ "$INCLUDE_CPU" -eq 1 ]]; then
    run_benchmark_scenario "cpu_filter_off" "$ROOT/configs/cpu_filter_off.yaml" "cpu" "$rep"
    run_benchmark_scenario "cpu_filter_on_njobs_all" "$ROOT/configs/cpu_filter_on_njobsall.yaml" "cpu" "$rep"
    run_benchmark_scenario "cpu_filter_on_njobs1" "$ROOT/configs/cpu_filter_on_njobs1.yaml" "cpu" "$rep"
  fi
done

if [[ "$RUN_PROFILES" -eq 1 ]]; then
  echo "running profiling passes"

  run_cmd "$PY_BIN" -m cProfile -o "$ROOT/profiles/bench_cuda_desktop_filter_off.prof" -m dagzoo benchmark \
    --config "$ROOT/configs/cuda_desktop_filter_off.yaml" \
    --preset custom --suite smoke --device "$DESKTOP_DEVICE" --no-memory --hardware-policy none \
    --out-dir "$ROOT/bench/profile_cuda_desktop_filter_off"

  run_cmd "$PY_BIN" -m cProfile -o "$ROOT/profiles/bench_cuda_desktop_filter_on_njobs1.prof" -m dagzoo benchmark \
    --config "$ROOT/configs/cuda_desktop_filter_on_njobs1.yaml" \
    --preset custom --suite smoke --device "$DESKTOP_DEVICE" --no-memory --hardware-policy none \
    --out-dir "$ROOT/bench/profile_cuda_desktop_filter_on_njobs1"

  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "+ /usr/bin/time -p $PY_BIN -m cProfile -o $ROOT/profiles/generate_cuda_desktop_write.prof -m dagzoo generate ... > $ROOT/generate/cuda_desktop_write.log 2>&1"
    echo "+ /usr/bin/time -p $PY_BIN -m cProfile -o $ROOT/profiles/generate_cuda_desktop_nowrite.prof -m dagzoo generate ... > $ROOT/generate/cuda_desktop_nowrite.log 2>&1"
  else
    /usr/bin/time -p "$PY_BIN" -m cProfile -o "$ROOT/profiles/generate_cuda_desktop_write.prof" -m dagzoo generate \
      --config "$ROOT/configs/cuda_desktop_filter_off.yaml" \
      --device "$DESKTOP_DEVICE" \
      --num-datasets "$GEN_PROFILE_DATASETS" \
      --out "$ROOT/generate/cuda_desktop_write" \
      > "$ROOT/generate/cuda_desktop_write.log" 2>&1

    /usr/bin/time -p "$PY_BIN" -m cProfile -o "$ROOT/profiles/generate_cuda_desktop_nowrite.prof" -m dagzoo generate \
      --config "$ROOT/configs/cuda_desktop_filter_off.yaml" \
      --device "$DESKTOP_DEVICE" \
      --num-datasets "$GEN_PROFILE_DATASETS" \
      --no-dataset-write \
      --out "$ROOT/generate/cuda_desktop_nowrite" \
      > "$ROOT/generate/cuda_desktop_nowrite.log" 2>&1
  fi
fi

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "+ aggregate benchmark outputs into $ROOT/reports/priority_verdict.md"
  echo "done (dry-run)"
  exit 0
fi

ROOT="$ROOT" GEN_PROFILE_DATASETS="$GEN_PROFILE_DATASETS" "$PY_BIN" - <<'PY'
import json
import os
import re
import statistics
from pathlib import Path

root = Path(os.environ["ROOT"])
gen_profile_datasets = int(os.environ["GEN_PROFILE_DATASETS"])
bench_root = root / "bench"
report_root = root / "reports"
report_root.mkdir(parents=True, exist_ok=True)


def median_or_none(values):
    cleaned = [float(v) for v in values if isinstance(v, (int, float))]
    if not cleaned:
        return None
    return float(statistics.median(cleaned))


def parse_time_real(log_path: Path):
    if not log_path.exists():
        return None
    text = log_path.read_text(errors="ignore")
    matches = re.findall(r"^real\s+([0-9]+(?:\.[0-9]+)?)\s*$", text, flags=re.MULTILINE)
    if not matches:
        return None
    return float(matches[-1])

scenario_rows: dict[str, list[dict]] = {}
for summary_path in bench_root.glob("*/rep*/summary.json"):
    scenario = summary_path.parent.parent.name
    payload = json.loads(summary_path.read_text())
    preset_results = payload.get("preset_results")
    if not isinstance(preset_results, list) or not preset_results:
        continue
    pr = preset_results[0]

    dpm = pr.get("datasets_per_minute")
    gen_dpm = pr.get("generation_datasets_per_minute", dpm)
    write_dpm = pr.get("write_datasets_per_minute")

    attempts_dpm = pr.get("estimated_attempts_per_minute")
    if not isinstance(attempts_dpm, (int, float)):
        mean_attempts = pr.get("mean_attempts_per_dataset")
        if isinstance(gen_dpm, (int, float)) and isinstance(mean_attempts, (int, float)):
            attempts_dpm = float(gen_dpm) * float(mean_attempts)

    row = {
        "datasets_per_minute": float(dpm) if isinstance(dpm, (int, float)) else None,
        "generation_datasets_per_minute": float(gen_dpm) if isinstance(gen_dpm, (int, float)) else None,
        "write_datasets_per_minute": float(write_dpm) if isinstance(write_dpm, (int, float)) else None,
        "estimated_attempts_per_minute": float(attempts_dpm) if isinstance(attempts_dpm, (int, float)) else None,
        "mean_attempts_per_dataset": (
            float(pr.get("mean_attempts_per_dataset"))
            if isinstance(pr.get("mean_attempts_per_dataset"), (int, float))
            else None
        ),
        "filter_rejection_rate_attempt_level": (
            float(pr.get("filter_rejection_rate_attempt_level"))
            if isinstance(pr.get("filter_rejection_rate_attempt_level"), (int, float))
            else None
        ),
    }
    scenario_rows.setdefault(scenario, []).append(row)

scenario_summary: dict[str, dict] = {}
for scenario, rows in scenario_rows.items():
    scenario_summary[scenario] = {
        "reps": len(rows),
        "datasets_per_minute": median_or_none([r["datasets_per_minute"] for r in rows]),
        "generation_datasets_per_minute": median_or_none(
            [r["generation_datasets_per_minute"] for r in rows]
        ),
        "write_datasets_per_minute": median_or_none([r["write_datasets_per_minute"] for r in rows]),
        "estimated_attempts_per_minute": median_or_none(
            [r["estimated_attempts_per_minute"] for r in rows]
        ),
        "mean_attempts_per_dataset": median_or_none([r["mean_attempts_per_dataset"] for r in rows]),
        "filter_rejection_rate_attempt_level": median_or_none(
            [r["filter_rejection_rate_attempt_level"] for r in rows]
        ),
    }

desktop_off = scenario_summary.get("cuda_desktop_filter_off", {})
desktop_on_all = scenario_summary.get("cuda_desktop_filter_on_njobs_all", {})
desktop_on_1 = scenario_summary.get("cuda_desktop_filter_on_njobs1", {})

desktop_gen_off = desktop_off.get("generation_datasets_per_minute")
desktop_write_off = desktop_off.get("write_datasets_per_minute")
desktop_attempts_all = desktop_on_all.get("estimated_attempts_per_minute")
desktop_attempts_1 = desktop_on_1.get("estimated_attempts_per_minute")

def safe_ratio(n, d):
    if isinstance(n, (int, float)) and isinstance(d, (int, float)) and d > 0:
        return float(n) / float(d)
    return None

write_to_gen_ratio = safe_ratio(desktop_write_off, desktop_gen_off)

filter_overhead_pct = None
if isinstance(desktop_attempts_1, (int, float)) and isinstance(desktop_gen_off, (int, float)) and desktop_gen_off > 0:
    filter_overhead_pct = (1.0 - (desktop_attempts_1 / desktop_gen_off)) * 100.0

n_jobs_gain_pct = None
if isinstance(desktop_attempts_1, (int, float)) and isinstance(desktop_attempts_all, (int, float)) and desktop_attempts_all > 0:
    n_jobs_gain_pct = ((desktop_attempts_1 - desktop_attempts_all) / desktop_attempts_all) * 100.0

write_real = parse_time_real(root / "generate" / "cuda_desktop_write.log")
nowrite_real = parse_time_real(root / "generate" / "cuda_desktop_nowrite.log")

write_dpm_sync = None
nowrite_dpm_sync = None
write_penalty_pct = None
if isinstance(write_real, (int, float)) and write_real > 0:
    write_dpm_sync = (gen_profile_datasets / write_real) * 60.0
if isinstance(nowrite_real, (int, float)) and nowrite_real > 0:
    nowrite_dpm_sync = (gen_profile_datasets / nowrite_real) * 60.0
if isinstance(write_dpm_sync, (int, float)) and isinstance(nowrite_dpm_sync, (int, float)) and nowrite_dpm_sync > 0:
    write_penalty_pct = (1.0 - (write_dpm_sync / nowrite_dpm_sync)) * 100.0

gate_145_gt_147 = bool(isinstance(filter_overhead_pct, (int, float)) and filter_overhead_pct >= 20.0)
gate_146_gt_147 = bool(
    (isinstance(write_penalty_pct, (int, float)) and write_penalty_pct >= 20.0)
    or (isinstance(write_to_gen_ratio, (int, float)) and write_to_gen_ratio <= 1.2)
)
if isinstance(write_to_gen_ratio, (int, float)):
    gate_147_gt_66 = bool(write_to_gen_ratio >= 1.0)
else:
    gate_147_gt_66 = True

current_order_correct = gate_145_gt_147 and gate_146_gt_147 and gate_147_gt_66

if current_order_correct:
    recommended_order = ["#145", "#146", "#147", "#66"]
elif gate_145_gt_147 and not gate_146_gt_147:
    recommended_order = ["#145", "#147", "#66", "#146"]
elif gate_146_gt_147 and not gate_145_gt_147:
    recommended_order = ["#146", "#147", "#66", "#145"]
else:
    recommended_order = ["#147", "#66", "#146", "#145"]

if not gate_147_gt_66 and "#147" in recommended_order and "#66" in recommended_order:
    i147 = recommended_order.index("#147")
    i66 = recommended_order.index("#66")
    if i66 > i147:
        recommended_order[i147], recommended_order[i66] = recommended_order[i66], recommended_order[i147]

outputs = {
    "scenario_summary": scenario_summary,
    "derived": {
        "write_to_gen_ratio": write_to_gen_ratio,
        "filter_overhead_pct": filter_overhead_pct,
        "n_jobs_gain_pct": n_jobs_gain_pct,
        "write_dpm_sync": write_dpm_sync,
        "nowrite_dpm_sync": nowrite_dpm_sync,
        "write_penalty_pct": write_penalty_pct,
    },
    "gates": {
        "145_gt_147": gate_145_gt_147,
        "146_gt_147": gate_146_gt_147,
        "147_gt_66": gate_147_gt_66,
    },
    "current_order_correct_for_raw_generation": current_order_correct,
    "recommended_order": recommended_order,
    "notes": [
        "Desktop-only evidence. H100 evidence is intentionally absent in this run.",
        "Objective: raw generation datasets/minute.",
    ],
}

(report_root / "priority_inputs.json").write_text(json.dumps(outputs, indent=2, sort_keys=True))

lines = []
lines.append("# Issue #148 Prioritization Verdict (CUDA Desktop)")
lines.append("")
lines.append(f"Artifact root: `{root}`")
lines.append("")
lines.append("## Scenario Medians")
lines.append("")
lines.append("| Scenario | Reps | Gen DPM | Write DPM | Attempts DPM | Mean Attempts | Filter Rejection Rate |")
lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
for scenario in sorted(scenario_summary):
    s = scenario_summary[scenario]
    lines.append(
        "| "
        + scenario
        + " | "
        + str(s.get("reps"))
        + " | "
        + (f"{s['generation_datasets_per_minute']:.2f}" if isinstance(s.get("generation_datasets_per_minute"), (int, float)) else "n/a")
        + " | "
        + (f"{s['write_datasets_per_minute']:.2f}" if isinstance(s.get("write_datasets_per_minute"), (int, float)) else "n/a")
        + " | "
        + (f"{s['estimated_attempts_per_minute']:.2f}" if isinstance(s.get("estimated_attempts_per_minute"), (int, float)) else "n/a")
        + " | "
        + (f"{s['mean_attempts_per_dataset']:.3f}" if isinstance(s.get("mean_attempts_per_dataset"), (int, float)) else "n/a")
        + " | "
        + (f"{s['filter_rejection_rate_attempt_level']:.3f}" if isinstance(s.get("filter_rejection_rate_attempt_level"), (int, float)) else "n/a")
        + " |"
    )

lines.append("")
lines.append("## Derived Signals")
lines.append("")
lines.append(f"- write_to_gen_ratio (desktop filter-off): `{write_to_gen_ratio}`")
lines.append(f"- filter_overhead_pct (desktop n_jobs=1 vs filter-off): `{filter_overhead_pct}`")
lines.append(f"- n_jobs_gain_pct (desktop n_jobs=1 vs -1): `{n_jobs_gain_pct}`")
lines.append(f"- write_penalty_pct (generate write vs no-write): `{write_penalty_pct}`")
lines.append("")
lines.append("## Gates")
lines.append("")
lines.append(f"- `#145 > #147`: `{gate_145_gt_147}`")
lines.append(f"- `#146 > #147`: `{gate_146_gt_147}`")
lines.append(f"- `#147 > #66`: `{gate_147_gt_66}`")
lines.append("")
lines.append("## Verdict")
lines.append("")
lines.append(f"- Current #148 ordering correct for raw-generation objective: `{current_order_correct}`")
lines.append(f"- Recommended order from desktop evidence: `{', '.join(recommended_order)}`")
lines.append("- Note: H100 data is not included in this script run.")

(report_root / "priority_verdict.md").write_text("\n".join(lines) + "\n")

print(f"wrote: {report_root / 'priority_inputs.json'}")
print(f"wrote: {report_root / 'priority_verdict.md'}")
PY

echo "done"
echo "artifact_root=$ROOT"
echo "verdict_report=$ROOT/reports/priority_verdict.md"
