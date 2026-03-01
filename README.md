# cauchy-generator

High-throughput synthetic tabular data generation built around causal structure.
Use it to generate, benchmark, and stress-test tabular datasets with
deterministic seed behavior.

## Quick Start

Install:

```bash
uv tool install cauchy-generator
```

Generate a default batch:

```bash
cauchy-gen generate --config configs/default.yaml --num-datasets 10 --out data/run1
```

Run a smoke benchmark:

```bash
cauchy-gen benchmark --suite smoke --profile cpu --out-dir benchmarks/results/smoke_cpu
```

Inspect detected hardware profile:

```bash
cauchy-gen hardware
```

## Features at a Glance

- Missingness workflows (MCAR/MAR/MNAR) with deterministic masks.
- Meta-feature steering toward target metric bands.
- Curriculum staging for complexity progression.
- Many-class workflows within the current rollout envelope.
- Shift/drift controls for graph, mechanism, and noise.
- Benchmark guardrails for runtime and metadata checks.

## Documentation (End Users)

- [docs/usage-guide.md](docs/usage-guide.md): Primary workflow hub.
- [docs/how-it-works.md](docs/how-it-works.md): System flow and terminology.
- [docs/output-format.md](docs/output-format.md): Output schema and artifacts.
- Feature guides:
  [diagnostics](docs/features/diagnostics.md),
  [steering](docs/features/steering.md),
  [missingness](docs/features/missingness.md),
  [curriculum](docs/features/curriculum.md),
  [many-class](docs/features/many-class.md),
  [shift](docs/features/shift.md),
  [benchmark guardrails](docs/features/benchmark-guardrails.md)

## Python API

```python
from cauchy_generator import GeneratorConfig, generate_one

config = GeneratorConfig.from_yaml("configs/default.yaml")
bundle = generate_one(config, seed=42)
print(bundle.X_train.shape, bundle.y_train.shape)
```

For command-line and workflow details, use
[docs/usage-guide.md](docs/usage-guide.md).

## Roadmap and Development

- [docs/development/roadmap.md](docs/development/roadmap.md)
- [docs/development/backlog_decision_rules.md](docs/development/backlog_decision_rules.md)
- [docs/development/design-decisions.md](docs/development/design-decisions.md)
- [reference/literature_evidence_2026.md](reference/literature_evidence_2026.md)
