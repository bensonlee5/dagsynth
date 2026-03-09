#!/usr/bin/env python3
"""Seed keyed-RNG refactor tracker objects in Linear."""

from __future__ import annotations

import argparse
from pathlib import Path

from github_to_linear import (
    DEFAULT_LINEAR_ENDPOINT,
    LinearClient,
    LinearState as _LinearState,
    MigrationError as _MigrationError,
    load_linear_api_key,
)
from seed_linear_utils import (
    TicketSpec,
    ensure_labels as _ensure_labels,
    find_existing_project_issues,
    required_label_specs as _required_label_specs,
    seed_ticket_specs,
)


KEYED_RNG_EPIC_TITLE = "epic: keyed RNG refactor for semantic reproducibility"
LABEL_COLORS = {
    "documentation": "#0EA5E9",
    "rng": "#2563EB",
}

LinearState = _LinearState
MigrationError = _MigrationError


def _markdown_list(items: list[str]) -> str:
    return "\n".join(f"- {item}" for item in items)


def build_ticket_specs() -> list[TicketSpec]:
    return [
        TicketSpec(
            title="design(rng): define keyed RNG contract and migration plan for semantic reproducibility",
            state_name="Todo",
            labels=("rng",),
            parent_title=KEYED_RNG_EPIC_TITLE,
            description=(
                "## Summary\n\n"
                "Define the keyed RNG contract that will replace order-coupled ambient generator usage "
                "across dagzoo generation and runtime orchestration.\n\n"
                "## Why\n\n"
                "The repo currently mixes seed derivation, offset-based helper seeds, and direct "
                "`torch.Generator` draw order in ways that make semantic reproducibility harder to reason "
                "about as execution paths are regrouped or retried.\n\n"
                "## Scope\n\n"
                + _markdown_list(
                    [
                        "Inventory current RNG seams in `src/dagzoo` and bucket them by semantic stage.",
                        "Define the keyed namespace tree for layout, node spec, typed plan sampling, execution, noise, missingness, split permutation, postprocess, and benchmarks.",
                        "Define compatibility expectations for existing `SeedManager`, `offset_seed32`, and direct generator callsites.",
                        "State which seeded outputs must remain stable and which may intentionally change under the migration.",
                    ]
                )
                + "\n\n## Acceptance Criteria\n\n"
                + _markdown_list(
                    [
                        "The design note maps every current RNG seam to an explicit semantic key namespace.",
                        "The note defines migration ordering and compatibility constraints for the keyed RNG rollout.",
                        "The note explains how scalar, batched, retry, and replay paths preserve semantic reproducibility.",
                    ]
                )
                + "\n\n## Validation\n\n"
                + _markdown_list(
                    [
                        '`rg -n "SeedManager|offset_seed32|generator=" src/dagzoo` has been used to inventory the migration surface.',
                        "The design note cross-references the concrete runtime seams discovered in the inventory.",
                    ]
                )
            ),
        ),
        TicketSpec(
            title="refactor(rng): add keyed RNG primitives and hierarchical namespace helpers",
            state_name="Backlog",
            labels=("rng",),
            parent_title=KEYED_RNG_EPIC_TITLE,
            description=(
                "## Summary\n\n"
                "Extend `dagzoo.rng` with keyed substream primitives that deterministically derive device-aware "
                "torch generators from one base seed and a semantic key path.\n\n"
                "## Why\n\n"
                "The current seed substrate is good at deterministic derivation, but higher-level code still couples "
                "behavior to call order and ad hoc offset conventions.\n\n"
                "## Scope\n\n"
                + _markdown_list(
                    [
                        "Add keyed child/subkey derivation APIs on top of the existing seed substrate.",
                        "Support deterministic CPU/CUDA/MPS generator creation for the same `(seed, key)` contract.",
                        "Keep compatibility shims or documented delegation for existing `SeedManager` entrypoints where needed.",
                    ]
                )
                + "\n\n## Acceptance Criteria\n\n"
                + _markdown_list(
                    [
                        "The keyed RNG API returns stable streams for identical `(seed, key)` inputs.",
                        "Sibling key evaluation order does not affect produced streams.",
                        "Existing callers can migrate without introducing global RNG state or manual seed mutation.",
                    ]
                )
                + "\n\n## Validation\n\n"
                + _markdown_list(
                    [
                        "`./.venv/bin/pytest -q tests/test_rng.py tests/test_sampling.py tests/test_noise_sampling.py`",
                    ]
                )
            ),
        ),
        TicketSpec(
            title="refactor(execution): move typed plans and executor paths onto keyed RNG",
            state_name="Backlog",
            labels=("rng",),
            parent_title=KEYED_RNG_EPIC_TITLE,
            description=(
                "## Summary\n\n"
                "Migrate typed plan sampling and execution off ambient draw order so scalar helpers, node execution, "
                "and fixed-layout batched generation consume randomness through semantic keys.\n\n"
                "## Why\n\n"
                "Execution semantics are currently vulnerable to RNG perturbations from regrouping, batching, and "
                "path-specific delegation changes.\n\n"
                "## Scope\n\n"
                + _markdown_list(
                    [
                        "Migrate `execution_semantics`, scalar wrappers, converter execution, node pipeline, and fixed-layout batched execution onto keyed RNG derivation.",
                        "Preserve scalar-vs-batched plan equivalence and replay behavior where the contract requires it.",
                        "Ensure nested/product plans and converter groups use stable child-key derivation rather than shared draw order.",
                    ]
                )
                + "\n\n## Acceptance Criteria\n\n"
                + _markdown_list(
                    [
                        "Plan sampling and execution consume randomness through semantic keys rather than ambient generator ordering.",
                        "Regrouping parent reductions or converter groups does not perturb unrelated randomness.",
                        "Explicit-plan scalar vs batched equivalence remains covered.",
                    ]
                )
                + "\n\n## Validation\n\n"
                + _markdown_list(
                    [
                        "`./.venv/bin/pytest -q tests/test_execution_semantics.py tests/test_fixed_layout_batched.py tests/test_node_pipeline.py`",
                    ]
                )
            ),
        ),
        TicketSpec(
            title="refactor(runtime): move orchestration, noise, missingness, and postprocess onto keyed RNG",
            state_name="Backlog",
            labels=("rng",),
            parent_title=KEYED_RNG_EPIC_TITLE,
            description=(
                "## Summary\n\n"
                "Move dataset-level runtime orchestration to keyed RNG so retries, grouped generation, split "
                "permutation, noise selection, missingness, and postprocess remapping are keyed by semantic stage.\n\n"
                "## Why\n\n"
                "The current runtime mixes deterministic child-seed derivation with offset helpers and shared-stage "
                "generator usage, which makes semantic reproducibility fragile when orchestration changes.\n\n"
                "## Scope\n\n"
                + _markdown_list(
                    [
                        "Migrate layout/runtime orchestration, noise-family selection, missingness, split permutation, and postprocess RNG seams.",
                        "Preserve dataset-level replay and grouped-noise runtime behavior under the new keyed model.",
                        "Ensure retrying one stage does not perturb sibling-stage randomness.",
                    ]
                )
                + "\n\n## Acceptance Criteria\n\n"
                + _markdown_list(
                    [
                        "Dataset generation stages use explicit semantic namespaces rather than offset-only coupling.",
                        "Grouped runtime generation remains reproducible for fixed seed and config.",
                        "Metadata replay and emitted bundle reproducibility still satisfy the documented contract.",
                    ]
                )
                + "\n\n## Validation\n\n"
                + _markdown_list(
                    [
                        "`./.venv/bin/pytest -q tests/test_generate.py tests/test_postprocess.py tests/test_missingness_sampling.py tests/test_benchmark_suite.py`",
                    ]
                )
            ),
        ),
        TicketSpec(
            title="hardening(repro): prove keyed semantic reproducibility and update docs",
            state_name="Backlog",
            labels=("rng", "documentation"),
            parent_title=KEYED_RNG_EPIC_TITLE,
            description=(
                "## Summary\n\n"
                "Codify the new keyed RNG reproducibility contract in docs and end-to-end regression coverage.\n\n"
                "## Why\n\n"
                "This refactor changes the repo's internal randomness model. The new contract needs explicit docs and "
                "test coverage so later changes do not silently regress semantic reproducibility.\n\n"
                "## Scope\n\n"
                + _markdown_list(
                    [
                        "Update reproducibility docs in `docs/how-it-works.md` and `docs/development/design-decisions.md`.",
                        "Add regression coverage for order independence across regrouping, retries, scalar-vs-batched execution, and benchmark reproducibility checks.",
                        "Call out merge-time version bump and `CHANGELOG.md` requirements for `src/dagzoo` behavior changes.",
                    ]
                )
                + "\n\n## Acceptance Criteria\n\n"
                + _markdown_list(
                    [
                        "Repo docs describe the keyed RNG model and its semantic reproducibility guarantees.",
                        "Tests cover the intended order-independence and replay guarantees of the new keyed model.",
                        "The documented reproducibility contract is specific enough to guide future refactors.",
                    ]
                )
                + "\n\n## Validation\n\n"
                + _markdown_list(
                    [
                        "`./.venv/bin/pytest -q`",
                        "`./.venv/bin/ruff check scripts tests`",
                    ]
                )
            ),
        ),
    ]


def required_label_specs() -> dict[str, str]:
    return _required_label_specs(build_ticket_specs(), label_colors=LABEL_COLORS)


def ensure_labels(
    linear: LinearClient,
    *,
    team_id: str,
    existing_labels: dict[str, object],
) -> dict[str, object]:
    return _ensure_labels(
        linear,
        team_id=team_id,
        existing_labels=existing_labels,
        label_specs=required_label_specs(),
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--linear-api-key-file", required=True, type=Path)
    parser.add_argument("--project-slug", required=True)
    parser.add_argument("--endpoint", default=DEFAULT_LINEAR_ENDPOINT)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    api_key = load_linear_api_key(args.linear_api_key_file)
    linear = LinearClient(api_key, endpoint=args.endpoint, dry_run=args.dry_run)
    project_id, team_id, _team_key = linear.get_project(args.project_slug)
    states, labels = linear.get_team_metadata(team_id)
    states = linear.ensure_workflow_states(team_id, states)
    labels = ensure_labels(linear, team_id=team_id, existing_labels=labels)
    existing_issues = find_existing_project_issues(linear, project_id)

    created = seed_ticket_specs(
        linear,
        ticket_specs=build_ticket_specs(),
        states=states,
        labels=labels,
        existing_issues=existing_issues,
        team_id=team_id,
        project_id=project_id,
    )
    for spec in build_ticket_specs():
        issue = created[spec.title]
        print(f"SEEDED_ISSUE={issue['identifier']} {issue['url']} {spec.title}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
